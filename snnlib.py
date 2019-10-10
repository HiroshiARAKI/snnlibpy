import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from bindsnet.network import Network
from bindsnet.network.nodes import Nodes, Input, LIFNodes, IFNodes, IzhikevichNodes, SRM0Nodes, DiehlAndCookNodes, AdaptiveLIFNodes
from bindsnet.network.topology import Connection
from bindsnet.network.monitors import Monitor
from bindsnet.analysis.plotting import plot_spikes
from bindsnet.learning import PostPre
from bindsnet.encoding import poisson, PoissonEncoder
from bindsnet.datasets import MNIST

from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import os


class Spiking:
    """
    The Class to simulate Spiking Neural Networks.
    """

    LIF = LIFNodes
    IF = IFNodes
    IZHIKEVICH = IzhikevichNodes
    SRM = SRM0Nodes
    DIEHL_COOK = DiehlAndCookNodes
    ADAPTIVE_LIF = AdaptiveLIFNodes

    PROJECT_ROOT = os.getcwd()
    IMAGE_DIR = PROJECT_ROOT + '/images/'

    rest_voltage = -65  # mV. 静止膜電位
    reset_voltage = -65  # mV. リセット膜電位．通常は静止膜電位と一緒
    threshold = -40  # mV. 発火閾値
    refractory_period = 3  # ms. 不応期

    input_firing_rate = 100.  # Hz. 入力の最大発火率 (1sec.あたり何本のスパイクが出て欲しいか)

    gpu = torch.cuda.is_available()
    seed = 0

    np.random.seed(seed)
    torch.manual_seed(seed)

    def __init__(self, input_l, obs_time: int = 500, dt: float = 1.0):
        """
        The constructor of the class, 'Spiking'
        :param input_l:
        :param obs_time:
        """
        self.network: Network = Network()
        self.layer_index = 0
        self.input_l = input_l
        self.monitors = {}
        self.original = []
        self.input_layer_name = 'in'
        self.dt = dt
        self.train_loader = None
        self.test_loader = None
        self.batch = 1
        self.train_data_num = None
        self.test_data_num = None
        self.layer_names = []

        if self.gpu:
            print('GPU available.')
            self.network.to('cuda')
        else:
            print('Only CPU.')

        input_layer = Input(n=input_l, traces=True)
        self.network.add_layer(layer=input_layer, name=self.input_layer_name)
        self.layer_names.append(self.input_layer_name)

        self.pre = {
            'layer': input_layer,
            'name': self.input_layer_name
        }

        self.T = obs_time

        monitor = Monitor(
            obj=input_layer,
            state_vars=('s',),
            time=self.T
        )

        self.monitors[self.input_layer_name] = monitor

        self.network.add_monitor(monitor=monitor, name=self.input_layer_name)

    def add_layer(self, n: int, name='', node: Nodes = LIF):
        """
        Add a full connection layer that consists LIF neuron.
        :param n:
        :param name:
        :param node:
        :return:
        """

        layer = node(n=n,
                     traces=True,
                     rest=self.rest_voltage,
                     restet=self.reset_voltage,
                     thresh=self.threshold,
                     refrac=self.refractory_period,
                     )

        if name == '' or name is None:
            name = 'fc-' + str(self.layer_index)
            self.layer_index += 1

        self.network.add_layer(layer=layer, name=name)
        self.layer_names.append(name)

        connection = Connection(
            source=self.pre['layer'],
            target=layer,
            w=0.1 + 0.1 * torch.randn(self.pre['layer'].n, layer.n),
            update_rule=PostPre,
            nu=1e-3
        )

        self.network.add_connection(connection,
                                    source=self.pre['name'],
                                    target=name,)
        monitor = Monitor(
            obj=layer,
            state_vars=('s', 'v'),
            time=self.T
        )

        self.monitors[name] = monitor

        self.network.add_monitor(monitor=monitor, name=name)

        self.pre['layer'] = layer
        self.pre['name'] = name

    def load_MNIST(self, batch: int = 1):
        """
        Load MNIST dataset from pyTorch.
        :param batch:
        :return:
        """
        self.batch = batch
        self.train_data_num = 60000
        self.test_data_num = 10000

        train_data = MNIST(root=self.PROJECT_ROOT+'/data/mnist',
                           train=True,
                           download=True,
                           transform=transforms.ToTensor())
        test_data = MNIST(root=self.PROJECT_ROOT+'/data/mnist',
                          train=False,
                          download=True,
                          transform=transforms.ToTensor())

        self.train_loader = DataLoader(train_data,
                                       batch_size=batch,
                                       shuffle=True)
        self.test_loader = DataLoader(test_data,
                                      batch_size=batch,
                                      shuffle=False)

    def run(self, tr_size=None):
        """
        Let the Network run.
        :return:
        """
        if tr_size is None:
            tr_size = self.train_data_num
        else:
            tr_size = int(tr_size / self.batch)

        for i, data in tqdm(enumerate(self.train_loader)):
            for d in data['image']:
                # ポアソン分布に従ってスパイクに変換する
                # 第１引数は発火率になる
                poisson_img = poisson(d*self.input_firing_rate, time=self.T, dt=self.dt).reshape((self.T, 784))
                inputs_img = {'in': poisson_img}

                # self.plot_poisson_img(
                #     poisson_img, save=True,
                #     file_name='PoissonImg_'+str(data['label'].numpy()[0])+'.png')

                # run!
                self.network.run(inpts=inputs_img, time=self.T)

            if i >= tr_size:  # もし訓練データ数が指定の数に達したら終わり
                break

        print('Have finished running the network.')

    def plot_out_voltage(self, index: int, save: bool = False,
                         file_name: str = 'out_voltage.png', dpi: int = 300):
        """
        Plot a membrane potential of 'index'th neuron in the final layer.
        :param index:
        :param save:
        :param file_name:
        :param dpi:
        :return:
        """
        os.makedirs(self.IMAGE_DIR, exist_ok=True)

        voltage = self.monitors[self.pre['name']].get('v').numpy().reshape(self.T, self.pre['layer'].n).T[index]

        plt.title('Membrane Voltage at neuron[{0}] in final layer'.format(index))
        plt.plot(voltage)
        plt.xlabel('time [ms]')
        plt.ylabel('voltage [mV]')
        plt.ylim(self.reset_voltage-5, self.threshold+5)
        if not save:
            plt.show()
        else:
            plt.savefig(self.IMAGE_DIR+file_name, dpi=dpi)

        plt.close()

    def plot_spikes(self, save: bool = False,
                    file_name: str = 'spikes.png', dpi: int = 300):
        """
        Plot spike trains of all neurons as a scatter plot.
        :param save:
        :param file_name:
        :param dpi:
        :return:
        """

        self.make_image_dir()
        spikes = {}
        for m_name in self.monitors:
            spikes[m_name] = self.monitors[m_name].get('s')

        plt.ioff()
        plot_spikes(spikes)
        if not save:
            plt.show()
        else:
            plt.savefig(self.IMAGE_DIR+file_name, dpi=dpi)
        plt.close()

    def plot_poisson_img(self, image: torch.Tensor, save: bool = False,
                         file_name='poisson_img.png', dpi: int = 300):
        """
        Plot a poisson image.
        :param image:
        :param save:
        :param file_name:
        :param dpi:
        :return:
        """

        self.make_image_dir()

        result_img = np.zeros((28, 28))
        for dt_spike_img in image:
            result_img += dt_spike_img.numpy().reshape((28, 28))

        plt.imshow(result_img, cmap='winter')
        plt.colorbar().set_label('# of spikes')

        if not save:
            plt.show()
        else:
            plt.savefig(self.IMAGE_DIR + file_name, dpi=dpi)

        plt.close()

    def plot_output_weights_map(self, index: int, save: bool = False,
                                file_name: str = 'weight_map.png', dpi: int = 300):
        """
        Plot an afferent weight map of the last layer's [index]th neuron.
        :param index:
        :param save:
        :param file_name:
        :param dpi:
        :return:
        """
        self.make_image_dir()

        names = self.layer_names
        last = len(names) - 1

        # ネットワークの最終層の結合情報を取得
        weight: torch.Tensor = self.network.connections[(names[last-1], names[last])].w
        # ndarrayにして転置
        weight = weight.numpy().T
        # 欲しいマップを(28,28)の形にして画像としてカラーマップとして描画
        weight = weight[index].reshape(28, 28)

        # 必ずカラーマップの真ん中が0になるようにする
        wmax = weight.max()
        wmin = weight.min()
        abs_max = abs(wmax) if abs(wmax) > abs(wmin) else abs(wmin)

        plt.imshow(weight, cmap='coolwarm', vmax=abs_max, vmin=(-abs_max))
        plt.colorbar()
        if not save:
            plt.show()
        else:
            plt.savefig(self.IMAGE_DIR + file_name, dpi=dpi)
        plt.close()

    def get_train_batch(self, index) -> torch.Tensor:
        return self.train_loader[index]['data']

    def make_image_dir(self):
        return os.makedirs(self.IMAGE_DIR, exist_ok=True)
