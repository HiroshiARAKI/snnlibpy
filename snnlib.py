"""
snnlib.py

@description  A tiny library to use BindsNET usefully.
@author       HiroshiARAKI
@source       https://github.com/HiroshiARAKI/snnlibpy
@contact      araki@hirlab.net
@Website      https://hirlab.net
@update       2019.10.16
"""

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from bindsnet.network import Network
from bindsnet.network.nodes import (Nodes, Input, LIFNodes, IFNodes, IzhikevichNodes,
                                    SRM0Nodes, DiehlAndCookNodes, AdaptiveLIFNodes)
from bindsnet.network.topology import Connection
from bindsnet.network.monitors import Monitor
from bindsnet.analysis.plotting import plot_spikes
from bindsnet.learning import PostPre, NoOp
from bindsnet.encoding import poisson
from bindsnet.datasets import MNIST
from bindsnet.evaluation import all_activity, assign_labels, proportion_weighting

from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import os
from time import time


class Spiking:
    """
    The Class to simulate Spiking Neural Networks.
    """

    __version__ = '0.1.1'

    # ======= Constants ======= #

    LIF = LIFNodes
    IF = IFNodes
    IZHIKEVICH = IzhikevichNodes
    SRM = SRM0Nodes
    DIEHL_COOK = DiehlAndCookNodes
    ADAPTIVE_LIF = AdaptiveLIFNodes

    NO_STDP: str = 'No_STDP'
    SIMPLE_STDP: str = 'Simple_STDP'

    W_NORMAL_DIST: int = 0
    W_RANDOM: int = 1
    W_SIMPLE_RAND: int = 3

    PROJECT_ROOT: str = os.getcwd()
    IMAGE_DIR: str = PROJECT_ROOT + '/images/'

    DPI: int = 150  # 標準の保存グラフdpi (画質)

    rest_voltage = -65  # mV. 静止膜電位
    reset_voltage = -65  # mV. リセット膜電位．通常は静止膜電位と一緒
    threshold = -40  # mV. 発火閾値
    refractory_period = 3  # ms. 不応期

    input_firing_rate: float = 100  # Hz. 入力の最大発火率 (1sec.あたり何本のスパイクが出て欲しいか)

    # ======================== #

    gpu = torch.cuda.is_available()  # GPU is available?? -> if you wanna use gpu, you need to be cuda >= 9.0
    seed = 0

    np.random.seed(seed)

    if gpu:
        torch.cuda.manual_seed_all(seed)
    else:
        torch.manual_seed(seed)

    def __init__(self, input_l, obs_time: int = 500, dt: float = 1.0):
        """
        Constructor: Build SNN easily. Initialize many variables in backend.
        :param input_l:
        :param obs_time:
        """
        print('\033[31mYou Called Spiking Neural Networks Library "WBN".')
        print('=> WrappedBindsNET (This Library) :version. %s' % self.__version__)
        print('=> PyTorch :version. %s' % torch.__version__)
        print('=> TorchVision :version. %s\n' % torchvision.__version__)

        self.network: Network = Network()
        self.layer_index = 0
        self.input_l = input_l
        self.monitors = {}
        self.original = []
        self.input_layer_name = 'in'
        self.dt = dt
        self.train_data = None
        self.train_loader = None
        self.test_loader = None
        self.batch = 1
        self.train_data_num = None
        self.test_data_num = None
        self.label_num = 0
        self.layer_names = []

        self.assignments = None
        self.proportions = None
        self.rates = None
        self.accuracy = {'all': [], 'proportion': []}

        if self.gpu:
            print('GPU computing is available.')
            self.network.to('cuda')
        else:
            print('You use Only CPU computing.')

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

    def add_layer(self, n: int, name='', node: Nodes = LIF,
                  w=W_NORMAL_DIST, rule=SIMPLE_STDP, **kwargs):
        """
        Add a full connection layer that consists LIF neuron.
        :param n:
        :param name:
        :param node:
        :param w:
        :param rule:
        :param kwargs: nu (learning rate of STDP), mu, sigma, w_max and w_min are available
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

        if type(w) is int:
            if w is self.W_NORMAL_DIST:
                mu = 0.3 if 'mu' not in kwargs else kwargs['mu']
                sigma = 0.3 if 'sigma' not in kwargs else kwargs['sigma']

                w = self.weight_norm(self.pre['layer'].n, layer.n,
                                     mu=mu, sigma=sigma)
            if w is self.W_RANDOM:
                w_max = 0.5 if 'w_max' not in kwargs else kwargs['w_max']
                w_min = -0.5 if 'w_min' not in kwargs else kwargs['w_min']

                w = self.weight_rand(self.pre['layer'].n, layer.n,
                                     w_max=w_max, w_min=w_min)
            if w is self.W_SIMPLE_RAND:
                if 'scale' not in kwargs:
                    scale = 0.3
                else:
                    scale = kwargs['scale']
                w = self.weight_simple_rand(self.pre['layer'].n, layer.n, scale)

        self.network.add_layer(layer=layer, name=name)
        self.layer_names.append(name)

        if 'nu' not in kwargs:
            nu = (1e-3, 1e-3)
        else:
            nu = kwargs['nu']

        if rule == self.SIMPLE_STDP:
            l_rule = PostPre
        elif rule == self.NO_STDP:
            l_rule = NoOp
        else:
            l_rule = NoOp

        connection = Connection(
            source=self.pre['layer'],
            target=layer,
            w=w,
            update_rule=l_rule,
            nu=nu,
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

        print('-- Added', name, 'with the Learning rule,', rule)

    def add_inhibit_layer(self, n: int = None, name='', node: Nodes = LIF,
                          exc_w: float = 22.5, inh_w: float = -17.5):
        """
        Add an inhibitory layer behind the last layer.
        If you added this layer, you can add layers more behind a last normal layer, not an inhibitory layer.
        :param n:
        :param name:
        :param node:
        :param exc_w:
        :param inh_w:
        :return:
        """
        if n is None:
            n = self.pre['layer'].n

        layer = node(
            n=n,
            traces=False,
            rest=self.rest_voltage,
            restet=self.reset_voltage,
            thresh=self.threshold,
            refrac=self.refractory_period,
        )

        if name == '' or name is None:
            name = 'inh[' + self.pre['name'] + ']'
            self.layer_index += 1

        self.network.add_layer(layer=layer, name=name)

        n_neurons = self.pre['layer'].n

        #  最終層 - 即抑制層の接続
        w = exc_w * torch.diag(torch.ones(n_neurons))
        last_to_inh_conn = Connection(
            source=self.pre['layer'],
            target=layer,
            w=w,
            wmin=0,
            wmax=exc_w,
        )

        # 即抑制層 - 最終層の接続
        w = inh_w * (torch.ones(n_neurons, n_neurons) - torch.diag(torch.ones(n_neurons)))
        inh_to_last_conn = Connection(
            source=layer,
            target=self.pre['layer'],
            w=w,
            wmin=inh_w,
            wmax=0,
        )

        self.network.add_connection(last_to_inh_conn,
                                    source=self.pre['name'],
                                    target=name)
        self.network.add_connection(inh_to_last_conn,
                                    source=name,
                                    target=self.pre['name'])

    def load_MNIST(self, batch: int = 1):
        """
        Load MNIST dataset from pyTorch.
        :param batch:
        :return:
        """
        self.batch = batch
        self.train_data_num = 60000
        self.test_data_num = 10000
        self.label_num = 10

        self.train_data = MNIST(root=self.PROJECT_ROOT+'/data/mnist',
                                train=True,
                                download=True,
                                transform=transforms.ToTensor())
        test_data = MNIST(root=self.PROJECT_ROOT+'/data/mnist',
                          train=False,
                          download=True,
                          transform=transforms.ToTensor())

        self.train_loader = DataLoader(self.train_data,
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
        self.print_model()

        print()
        act_acc, pro_acc = (0., 0.)
        if tr_size is None:
            tr_size = int(self.train_data_num / self.batch)
        else:
            tr_size = int(tr_size / self.batch)

        progress = enumerate(self.train_loader)
        start = time()
        for i, data in progress:
            print('\033[31mProgress: %d / %d (%.4f seconds)\033[0m' % (i, tr_size, time() - start))

            spikes = torch.zeros(self.batch, self.T, self.pre['layer'].n)
            labels = torch.zeros(self.batch)

            for (j, d), l in zip(enumerate(tqdm(data['image'])), data['label']):  # batch loop
                # ポアソン分布に従ってスパイクに変換する
                poisson_img = poisson(d*self.input_firing_rate, time=self.T, dt=self.dt).reshape((self.T, 784))
                inputs_img = {'in': poisson_img}

                if self.gpu:
                    inputs_img = {k: v.cuda() for k, v in inputs_img.items()}

                # run!
                self.network.run(inpts=inputs_img, time=self.T)

                spikes[j] = self.monitors[self.pre['name']].get('s').squeeze()
                labels[j] = l

                self.network.reset_()

            # 1バッチ分の精度を計る
            act_acc, pro_acc = self.predict(spikes, labels)
            print(' -- Transition accuracy: %d, proportion weight accuracy: %d' % (act_acc, pro_acc))
            self.accuracy['all'].append(act_acc)
            self.accuracy['proportion'].append(pro_acc)

            if i >= tr_size:  # もし訓練データ数が指定の数に達したら終わり
                break

        print('\033[31mProgress: %d / %d data. (%.4f seconds)' % (tr_size, tr_size, time() - start))
        print('\nHave finished running the network.\033[0m')
        print(self.accuracy)
        plt.plot(self.accuracy['all'], label='all', c='b', marker='.')
        plt.plot(self.accuracy['proportion'], label='proportion', c='g', marker='.')
        plt.legend()
        plt.savefig(self.IMAGE_DIR+'result.png', dpi=self.DPI)

    def predict(self, spikes, labels):
        """
        Compute the accuracies with spike activities and proportion weighting.
        :param spikes:
        :param labels:
        :return:
        """
        # ラベル付
        self.assignments, self.proportions, self.rates = assign_labels(spikes,
                                                                       labels,
                                                                       self.label_num,
                                                                       self.rates)

        act_pred = all_activity(spikes, self.assignments, self.label_num)
        pro_pred = proportion_weighting(spikes, self.assignments, self.proportions, self.label_num)

        act_acc = torch.sum(labels.long() == act_pred).item() / self.batch
        pro_acc = torch.sum(labels.long() == pro_pred).item() / self.batch

        return act_acc, pro_acc

    def plot_out_voltage(self, index: int, save: bool = False,
                         file_name: str = 'out_voltage.png', dpi: int = DPI):
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

    def plot_spikes(self, save: bool = False, index: int = 0,
                    file_name: str = 'spikes.png', dpi: int = DPI):
        """
        Plot spike trains of all neurons as a scatter plot.
        :param save:
        :param index:
        :param file_name:
        :param dpi:
        :return:
        """

        self.make_image_dir()

        data = self.train_data[index]
        d = data['image']
        label = data['label']

        poisson_img = poisson(d * self.input_firing_rate, time=self.T, dt=self.dt).reshape((self.T, 784))
        inputs_img = {'in': poisson_img}

        self.network.run(inpts=inputs_img, time=self.T)

        spikes = {}
        for m_name in self.monitors:
            spikes[m_name] = self.monitors[m_name].get('s')

        plt.ioff()
        plot_spikes(spikes)
        if not save:
            plt.show()
        else:
            plt.savefig(self.IMAGE_DIR+'label_'+str(label)+file_name, dpi=dpi)
        plt.close()

    def plot_poisson_img(self, image: torch.Tensor, save: bool = False,
                         file_name='poisson_img.png', dpi: int = DPI):
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
                                file_name: str = 'weight_map.png', dpi: int = DPI):
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

    def print_model(self):
        print('\033[31m')
        print('=============================')
        print('Show your network information below.')
        layers: dict[str: Nodes] = self.network.layers
        print('Layers:')
        for l in layers:
            print(' '+l+'('+str(layers[l].n)+')', end='\n    |\n')
        print('  [END]')
        print('=============================')

    @staticmethod
    def weight_norm(n: int, m: int, mu: float = 0.3, sigma: float = 0.3) -> torch.Tensor:
        return mu + sigma * torch.randn(n, m)

    @staticmethod
    def weight_rand(n: int, m: int, w_max: float = 0.5, w_min: float = -0.5) -> torch.Tensor:
        x = torch.rand(n, m)
        x_max = x.max()
        x_min = x.min()
        return ((x - x_min) / (x_max - x_min)) * (w_max - w_min) + w_min

    @staticmethod
    def weight_simple_rand(n: int, m: int, scale: float = 0.3) -> torch.Tensor:
        return scale * torch.rand(n, m)

