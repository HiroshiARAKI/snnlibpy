import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from bindsnet.network import Network
from bindsnet.network.nodes import Nodes, Input, LIFNodes, IFNodes, IzhikevichNodes, SRM0Nodes, DiehlAndCookNodes, AdaptiveLIFNodes
from bindsnet.network.topology import Connection
from bindsnet.network.monitors import Monitor
from bindsnet.analysis.plotting import plot_spikes
from bindsnet.learning import PostPre
from bindsnet.encoding import poisson
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

    rest_voltage = -65
    reset_voltage = -65
    threshold = -40
    refractory_period = 3

    def __init__(self, input_l, obs_time: int = 500, dt: float = 1.0):
        """
        The constructor of the class, 'Spiking'
        :param input_l:
        :param obs_time:
        """
        self.network = Network()
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

        input_layer = Input(n=input_l, traces=True)
        self.network.add_layer(layer=input_layer, name=self.input_layer_name)

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

        connection = Connection(
            source=self.pre['layer'],
            target=layer,
            w=0.05 + 0.1 * torch.randn(self.pre['layer'].n, layer.n),
            update_rule=PostPre,
            nu=1e-4
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
                poisson_img = poisson(d * 255., time=self.T, dt=self.dt).reshape((self.T, 784))
                inputs_img = {'in': poisson_img}
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
        os.makedirs(self.IMAGE_DIR, exist_ok=True)
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

