import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from bindsnet.network import Network
from bindsnet.network.nodes import Input, LIFNodes
from bindsnet.network.topology import Connection
from bindsnet.network.monitors import Monitor
from bindsnet.analysis.plotting import plot_spikes, plot_voltages
from bindsnet.learning import PostPre
from bindsnet.encoding import poisson_loader, poisson
from bindsnet.datasets import MNIST

from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import os


class Spiking:
    """
    The Class to simulate Spiking Neural Networks.
    """

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

    def add_lif_layer(self, n: int, name=''):
        """
        Add a full connection layer that consists LIF neuron.
        :param n:
        :param name:
        :return:
        """
        layer = LIFNodes(n=n, traces=True)

        if name == '':
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

    def load_MNIST(self, batch: int = 10):
        """
        Load MNIST dataset from pyTorch.
        :param batch:
        :return:
        """
        train_data = MNIST(root=os.getcwd()+'/data/mnist', train=True, download=True, transform=transforms.ToTensor())
        self.train_loader = DataLoader(train_data,
                                       batch_size=batch,
                                       shuffle=True)

        test_data = MNIST(root=os.getcwd()+'~/tmp/mnist', train=False, download=True, transform=transforms.ToTensor())
        self.test_loader = DataLoader(test_data,
                                      batch_size=batch,
                                      shuffle=False)

    def train_STDP(self):
        """
        Let the Network learn with STDP learning that is implemented by bindsNet.
        :return:
        """
        for i, data in tqdm(enumerate(self.train_loader)):
            for d in data['image']:
                poisson_img = poisson(d * 127., time=self.T, dt=self.dt).reshape((self.T, 784))
                inputs_img = {'in': poisson_img}
                self.network.run(inpts=inputs_img, time=self.T)

        spikes = {}
        for m_name in self.monitors:
            spikes[m_name] = self.monitors[m_name].get('s')

        plt.ioff()
        plot_spikes(spikes)
        plt.show()
