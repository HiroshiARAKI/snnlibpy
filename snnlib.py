"""
snnlib.py

@description  A tiny library to use BindsNET usefully.
@author       HiroshiARAKI
@source       https://github.com/HiroshiARAKI/snnlibpy
@contact      araki@hirlab.net
@Website      https://hirlab.net
@update       2019.10.31
"""

import torch
from torchvision import __version__ as tv_ver
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from bindsnet.network import Network
from bindsnet.network.nodes import (Nodes, Input, LIFNodes, IFNodes, IzhikevichNodes,
                                    SRM0Nodes, DiehlAndCookNodes, AdaptiveLIFNodes)
from bindsnet.network.topology import Connection
from bindsnet.network.monitors import Monitor
from bindsnet.analysis.plotting import plot_spikes
from bindsnet.learning import PostPre, NoOp, WeightDependentPostPre
from bindsnet.encoding import PoissonEncoder
from bindsnet.datasets import MNIST
from bindsnet.evaluation import assign_labels

from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import os
from time import time


class Spiking:
    """
    The Class to simulate Spiking Neural Networks.
    """

    __version__ = '0.1.7'

    # ======= Constants ======= #

    LIF = LIFNodes
    IF = IFNodes
    IZHIKEVICH = IzhikevichNodes
    SRM = SRM0Nodes
    DIEHL_COOK = DiehlAndCookNodes
    ADAPTIVE_LIF = AdaptiveLIFNodes

    NO_STDP: str = 'No_STDP'
    SIMPLE_STDP: str = 'Simple_STDP'
    WEIGHT_DEPENDENT_STDP: str = 'Weight_dependent_STDP'

    W_NORMAL_DIST: int = 0     # initialize with Normal Distribution
    W_RANDOM: int = 1          # initialize with Uniform Distribution [sw_min, sw_max]
    W_SIMPLE_RAND: int = 3     # initialize Uniform Distribution[0, scale]

    PROJECT_ROOT: str = os.getcwd()
    IMAGE_DIR: str = PROJECT_ROOT + '/images/'

    DPI: int = 150          # the dpi value of plt.savefig()

    rest_voltage = -65      # [mV] resting potential
    reset_voltage = -65     # [mV] reset potential
    threshold = -40         # [mV] firing threshold
    refractory_period = 3   # [ms] refractory period

    intensity: float = 128  # [Hz] firing rate of input spikes

    seed = 0                # a seed of random

    # ======================== #

    def __init__(self, input_l, obs_time: int = 500, dt: float = 1.0,
                 input_shape=(1, 28, 28)):
        """
        Constructor: Build SNN easily. Initialize many variables in backend.
        :param input_l:
        :param obs_time:
        """
        print('You Called Spiking Neural Networks Library "WBN"!!')
        print('=> WrappedBindsNET (This Library) :version. %s' % self.__version__)
        print('=> PyTorch :version. %s' % torch.__version__)
        print('=> TorchVision :version. %s\n' % tv_ver)

        self.network: Network = Network()
        self.layer_index = 0
        self.input_l = input_l
        self.monitors = {}
        self.original = []
        self.input_layer_name = 'in'
        self.dt = dt
        self.train_data = None
        self.test_data = None
        self.train_loader = None
        self.test_loader = None
        self.batch = 1
        self.train_data_num = None
        self.test_data_num = None
        self.label_num = 0
        self.layer_names = []
        self.history = {'test_acc': [], 'train_acc': []}
        self.gpu = torch.cuda.is_available()  # Is GPU available?

        self.workers = self.gpu * 4 * torch.cuda.device_count()

        np.random.seed(self.seed)

        if self.gpu:
            torch.cuda.manual_seed_all(self.seed)
        else:
            torch.manual_seed(self.seed)

        self.assignments = None
        self.proportions = None
        self.rates = None

        input_layer = Input(n=input_l, traces=True, shape=input_shape)

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
                  w=W_NORMAL_DIST, rule=SIMPLE_STDP,
                  wmax: float = 1, wmin: float = -1, norm: float = 78.4,
                  **kwargs):
        """
        Add a full connection layer that consists LIF neuron.
        :param n:
        :param name:
        :param node:
        :param w:
        :param rule:
        :param wmax:
        :param wmin:
        :param norm:
        :param kwargs: nu (learning rate of STDP), mu, sigma, w_max and w_min are available
        :return:
        """

        layer = node(n=n,
                     traces=True,
                     rest=self.rest_voltage,
                     restet=self.reset_voltage,
                     thresh=self.threshold,
                     refrac=self.refractory_period,
                     tc_decay=kwargs.get('tc_decay', 100.0),
                     theta_plus=kwargs.get('theta_plus', 0.05),
                     tc_theta_decay=kwargs.get('tc_theta_decay', 1e7),
                     **kwargs
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
                w_max = 0.5 if 'sw_max' not in kwargs else kwargs['sw_max']
                w_min = -0.5 if 'sw_min' not in kwargs else kwargs['sw_min']

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
        elif rule == self.WEIGHT_DEPENDENT_STDP:
            l_rule = WeightDependentPostPre
        elif rule == self.NO_STDP:
            l_rule = NoOp
        else:
            l_rule = NoOp

        connection = Connection(
            source=self.pre['layer'],
            target=layer,
            w=w,
            wmax=wmax,
            wmin=wmin,
            update_rule=l_rule,
            nu=nu,
            norm=norm
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
                          exc_w: float = 22.5, inh_w: float = -100):
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

        print('-- Added', name, 'as an inhibitory layer')

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

        self.train_data = MNIST(PoissonEncoder(time=self.T, dt=self.dt),
                                None,
                                root=self.PROJECT_ROOT+'/data/mnist',
                                train=True,
                                download=True,
                                transform=transforms.Compose(
                                    [transforms.ToTensor(), transforms.Lambda(lambda x: x * self.intensity)]
                                ))
        self.test_data = MNIST(PoissonEncoder(time=self.T, dt=self.dt),
                               None,
                               root=self.PROJECT_ROOT+'/data/mnist',
                               train=False,
                               download=True,
                               transform=transforms.Compose(
                                   [transforms.ToTensor(), transforms.Lambda(lambda x: x * self.intensity)]
                               ))

        self.train_loader = DataLoader(self.train_data,
                                       batch_size=batch,
                                       shuffle=True,
                                       pin_memory=self.gpu,
                                       num_workers=self.workers)
        self.test_loader = DataLoader(self.test_data,
                                      batch_size=batch,
                                      shuffle=False,
                                      pin_memory=self.gpu,
                                      num_workers=self.workers)

    def run(self, tr_size=None,
            unsupervised: bool = True, alpha: float = 0.8, interval: int = 250,
            debug: bool = False, **kwargs):
        """
        Let the Network run simply.
        :param tr_size:
        :param unsupervised:
        :param alpha:
        :param interval:
        :param debug:
        :return:
        """

        if tr_size is None:
            tr_size = int(self.train_data_num / self.batch)
        else:
            tr_size = int(tr_size / self.batch)

        n_out_neurons = self.pre['layer'].n

        assignment = -torch.ones(n_out_neurons)
        spikes = torch.zeros(interval, self.T, n_out_neurons)
        labels = []
        proportion = torch.zeros(n_out_neurons, self.label_num)
        rates = None

        progress = tqdm(enumerate(self.train_loader))
        start = time()
        for i, data in progress:
            progress.set_description_str('\rProgress: %d / %d (%.4f seconds)' % (i, tr_size, time() - start))

            inputs_img = {'in': data['encoded_image'].view(self.T, self.batch, 1, 28, 28)}

            if self.gpu:
                inputs_img = {key: img.cuda() for key, img in inputs_img.items()}

            # run!
            self.network.run(inpts=inputs_img, time=self.T, input_time_dim=1)

            # assign labels
            if unsupervised and i % interval == 0 and i > 0:
                t_labels = torch.tensor(labels)  # to tensor
                # Get the assignment of output neurons
                assignment, proportion, rates = assign_labels(
                    spikes=spikes,
                    labels=t_labels,
                    n_labels=self.label_num,
                    rates=rates,
                    alpha=alpha
                )
                labels = []

            if unsupervised:
                # labels used by assigning
                labels.append(data['label'])
                # output spike trains
                spikes[i % interval] = self.monitors[self.pre['name']].get('s').squeeze()

            self.network.reset_()

            if i >= tr_size:  # if reach training size you specified, break for loop
                break

        print('Progress: %d / %d (%.4f seconds)' % (tr_size, tr_size, time() - start))

        # compute train. and test accuracies
        if unsupervised:
            print('Computing accuracies...')
            # assignment = assignment.argmax(0)

            if debug:
                print('\n[Neurons assignments]')
                print(assignment)

            self.stop_learning()

            self.history['train_acc'].append(self.calc_train_accuracy(assignment, tr_size))
            self.history['test_acc'].append(self.calc_test_accuracy(assignment, kwargs.get('ts_size', None)))
            print('\n*** Train accuracy is %4f ***' % self.history['train_acc'][-1])
            print('*** Test accuracy is %4f ***\n' % self.history['test_acc'][-1])

            self.start_learning()

        print('===< Have finished running the network >===\n')

    def calc_test_accuracy(self, assignment: torch.Tensor, ts_size: int = None) -> float:
        """
        Calculate test accuracy with the assignment.
        :param assignment:
        :param ts_size:
        :return:
        """
        labels_rate = torch.zeros(self.label_num).float()  # each firing rate of labels
        count_correct = 0

        if ts_size is None:
            ts_size = self.test_data_num

        progress = tqdm(enumerate(self.test_loader))
        print('\n===< Calculate Test accuracy >===')
        for i, data in progress:
            progress.set_description_str('\rCalculate Test accuracy ... %d / %d ' % (i, ts_size))
            inputs_img = {'in': data['encoded_image'].view(self.T, self.batch, 1, 28, 28)}

            if self.gpu:
                inputs_img = {key: img.cuda() for key, img in inputs_img.items()}
            # run!
            self.network.run(inpts=inputs_img, time=self.T)

            # output spike trains
            spikes: torch.Tensor = self.monitors[self.pre['name']].get('s')

            # sum of the number of spikes
            sum_spikes = spikes.sum(0)
            self.network.reset_()

            for b in range(self.batch):
                for l in range(self.label_num):
                    if l in assignment:
                        indices = torch.tensor([i for i, a in enumerate(assignment) if a == l])
                        count_assign = torch.sum(assignment == l)
                        labels_rate[l] += torch.sum(sum_spikes[b][indices]).float() / count_assign.float()

                # if actual prediction equals desired label, increment the count.
                if labels_rate.argmax() == data['label']:
                    count_correct += 1

                # initialize zeros
                labels_rate[:] = 0

            if i >= ts_size:
                break

        print('\r ... done!')
        return float(count_correct) / float(ts_size)

    def calc_train_accuracy(self, assignment: torch.Tensor, tr_size: int = None) -> float:
        """
        Calculate train accuracy with the assignment.
        :param assignment:
        :param tr_size:
        :return:
        """
        if tr_size is None:
            tr_size = self.train_data_num

        labels_rate = torch.zeros(self.label_num).float()  # each firing rate of labels
        count_correct = 0

        progress = tqdm(enumerate(self.train_loader))
        print('\n===< Calculate Training accuracy >===')
        for i, data in progress:
            progress.set_description_str('\rCalculate Training accuracy ... %d / %d ' % (i, tr_size))
            inputs_img = {'in': data['encoded_image'].view(self.T, self.batch, 1, 28, 28)}

            if self.gpu:
                inputs_img = {key: img.cuda() for key, img in inputs_img.items()}
            # run!
            self.network.run(inpts=inputs_img, time=self.T)

            # output spike trains
            spikes: torch.Tensor = self.monitors[self.pre['name']].get('s')

            # sum of the number of spikes
            sum_spikes = spikes.sum(0)
            self.network.reset_()

            for b in range(self.batch):
                for l in range(self.label_num):
                    if l in assignment:
                        indices = torch.nonzero(assignment == l).view(-1)
                        count_assign = torch.sum(assignment == l).float()
                        labels_rate[l] += torch.sum(sum_spikes[:, indices]).float() / count_assign

                # if actual prediction equals desired label, increment the count.
                if labels_rate.argmax() == data['label'][b]:
                    count_correct += 1

                # initialize zeros
                labels_rate[:] = 0

            if i >= tr_size:
                break

        print('\r ... done!')
        return float(count_correct) / float(tr_size)

    def test(self, data_num: int):
        """
        Calculate test accuracy with the label assignment used training data.
        :param data_num:
        :return accuracy:
        """
        # Stop learning
        for layer in self.network.layers:
            self.network.layers[layer].train(False)

        # the assignments of output neurons
        assignment = torch.zeros(self.label_num, self.pre['layer'].n)

        print('===< Calculate train spikes and assign labels >===')
        progress = tqdm(enumerate(self.train_loader))
        for i, data in progress:
            progress.set_description_str('\rAssign labels... %d / %d ' % (i, data_num))
            inputs_img = {'in': data['encoded_image'].view(self.T, self.batch, 1, 28, 28)}

            if self.gpu:
                inputs_img = {key: img.cuda() for key, img in inputs_img.items()}
            # run!
            self.network.run(inpts=inputs_img, time=self.T)

            # output spike trains
            spikes: torch.Tensor = self.monitors[self.pre['name']].get('s')

            # sum of the number of spikes
            sum_spikes = spikes.sum(0)

            max_n_fire = sum_spikes.argmax(1)
            labels = data['label']

            for j, l in enumerate(labels):
                assignment[l][max_n_fire[j]] += 1

            self.network.reset_()

            if i >= data_num:
                break

        # this result is assignment of output neurons
        assignment = assignment.argmax(0)

        # Calculate accuracy
        labels_rate = torch.zeros(self.label_num).float()  # each firing rate of labels
        count_correct = 0
        progress = tqdm(enumerate(self.test_loader))
        print('\n===< Calculate Test accuracy >===')
        for i, data in progress:
            progress.set_description_str('\rCalculate Test accuracy ... %d / %d ' % (i, self.test_data_num))
            inputs_img = {'in': data['encoded_image'].view(self.T, self.batch, 1, 28, 28)}

            if self.gpu:
                inputs_img = {key: img.cuda() for key, img in inputs_img.items()}
            # run!
            self.network.run(inpts=inputs_img, time=self.T)

            # output spike trains
            spikes: torch.Tensor = self.monitors[self.pre['name']].get('s')

            # sum of the number of spikes
            sum_spikes = spikes.sum(0)
            self.network.reset_()

            for b in range(self.batch):
                for l in range(self.label_num):
                    if l in assignment:
                        indices = torch.tensor([i for i, a in enumerate(assignment) if a == l])
                        count_assign = torch.sum(assignment == l)
                        labels_rate[l] += torch.sum(sum_spikes[b][indices]).float() / count_assign.float()

                # if actual prediction equals desired label, increment the count.
                if labels_rate.argmax() == data['label']:
                    count_correct += 1

                # initialize zeros
                labels_rate[:] = 0

        acc = float(count_correct) / float(self.test_data_num)
        self.history['test_acc'].append(acc)

        print('\n*** Test accuracy is %4f ***\n' % acc)

        # make learning rates be back
        for layer in self.network.layers:
            self.network.layers[layer].train(True)

        return acc

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
        self.stop_learning()

        data = self.train_data[index]
        label = data['label']

        inputs_img = {'in': data['encoded_image'].view(self.T, self.batch, 1, 28, 28)}
        if self.gpu:
            inputs_img = {key: img.cuda() for key, img in inputs_img.items()}

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

        self.start_learning()

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

    def plot_output_weight_map(self, index: int, save: bool = False,
                               file_name: str = 'weight_map.png', dpi: int = DPI,
                               c_max: float = 1.0, c_min: float = -1.0):
        """
        Plot an afferent weight map of the last layer's [index]th neuron.
        :param index:
        :param save:
        :param file_name:
        :param dpi:
        :param c_max: max of colormap
        :param c_min: min of colormap
        :return:
        """
        self.make_image_dir()

        names = self.layer_names
        last = len(names) - 1

        # Get the connection information of the last layer
        weight: torch.Tensor = self.network.connections[(names[last-1], names[last])].w
        # to ndarray and trans.
        weight = weight.numpy().T if not self.gpu else weight.cpu().numpy().T
        # to shape as (28,28)
        weight = weight[index].reshape(28, 28)

        # Set the center of a colormap zero.
        wmax = weight.max() if weight.max() > c_max else c_max
        wmin = weight.min() if weight.min() < c_min else c_min
        abs_max = abs(wmax) if abs(wmax) > abs(wmin) else abs(wmin)

        plt.imshow(weight, cmap='coolwarm', vmax=abs_max, vmin=(-abs_max))
        plt.colorbar()
        if not save:
            plt.show()
        else:
            plt.savefig(self.IMAGE_DIR + file_name, dpi=dpi)
        plt.close()

    def plot_weight_maps(self, f_shape: tuple = (3, 3), file_name: str = 'weight_maps.png',
                         dpi: int = DPI, c_max: float = 1.0, c_min: float = -1.0, save: bool = True):
        """
        Plot weight maps of output connection with the shape of [f_shape].
        :param f_shape:
        :param file_name:
        :param dpi:
        :param c_max:
        :param c_min:
        :param save:
        :return:
        """

        self.make_image_dir()

        names = self.layer_names
        last = len(names) - 1

        # Get the connection information of the last layer
        weight: torch.Tensor = self.network.connections[(names[last - 1], names[last])].w
        # to ndarray and trans.
        weight = weight.numpy().T if not self.gpu else weight.cpu().numpy().T

        # setting of figure
        fig, axes = plt.subplots(ncols=f_shape[0], nrows=f_shape[1])

        index = 0
        im = None
        for cols in axes:
            for ax in cols:
                print('Plot weight map {}/{}'.format(index+1, f_shape[0]*f_shape[1]))

                # to shape as (28,28)
                tmp_weight = weight[index].reshape(28, 28)

                # Set the center of a colormap zero.
                wmax = tmp_weight.max() if tmp_weight.max() > c_max else c_max
                wmin = tmp_weight.min() if tmp_weight.min() < c_min else c_min
                abs_max = abs(wmax) if abs(wmax) > abs(wmin) else abs(wmin)

                im = ax.imshow(tmp_weight, cmap='coolwarm', vmax=abs_max, vmin=(-abs_max))
                ax.set_title('map({})'.format(index))
                ax.tick_params(labelbottom=False,
                               labelleft=False,
                               labelright=False,
                               labeltop=False,
                               bottom=False,
                               left=False,
                               right=False,
                               top=False
                               )
                index += 1

        fig.subplots_adjust(right=0.8)
        cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
        fig.colorbar(im, cax=cbar_ax)

        if not save:
            plt.show()
        else:
            plt.savefig(self.IMAGE_DIR + file_name, dpi=dpi)
        plt.close()

    def plot(self, plt_type: str, **kwargs):
        """
        A Shortcut function about plotting
        :param plt_type:
        :param kwargs:
        :return:
        """
        if 'save' not in kwargs:
            kwargs['save'] = True
        if 'prefix' not in kwargs:
            kwargs['prefix'] = ''
        if 'range' not in kwargs:
            kwargs['range'] = 1
        if 'f_shape' not in kwargs:
            kwargs['f_shape'] = (3, 3)

        if plt_type == 'wmp':
            for i in range(kwargs['range']):
                self.plot_output_weight_map(index=i,
                                            save=kwargs['save'],
                                            file_name='{}_wmp_'.format(kwargs['prefix'])+str(i+1)+'.png')
        elif plt_type == 'sp':
            for i in range(kwargs['range']):
                self.plot_spikes(save=kwargs['save'],
                                 index=i)
        elif plt_type == 'history':
            plt.plot(self.history['train_acc'], label='train_acc', marker='.', c='b')
            plt.plot(self.history['test_acc'], label='test_acc', marker='.', c='g')
            plt.xlabel('epochs')
            plt.ylabel('accuracy')
            plt.legend()
            if kwargs['save']:
                plt.savefig(self.IMAGE_DIR + '{}_accuracies.png'.format(kwargs['prefix']), dpi=self.DPI)
            else:
                plt.show()
            plt.close()
        elif plt_type == 'wmps':
            self.plot_weight_maps(f_shape=kwargs['f_shape'],
                                  file_name='{}_weight_maps.png'.format(kwargs['prefix']),
                                  save=kwargs['save'])

        elif plt_type == 'p_img':
            pass
        elif plt_type == 'v':
            pass
        else:
            print('Not Found the plt_type.')

    def get_train_batch(self, index) -> torch.Tensor:
        return self.train_loader[index]['data']

    def make_image_dir(self):
        return os.makedirs(self.IMAGE_DIR, exist_ok=True)

    def print_model(self):
        print('=============================')
        print('Show your network information below.')
        layers: dict[str: Nodes] = self.network.layers
        print('Layers:')
        for l in layers:
            print(' '+l+'('+str(layers[l].n)+')', end='\n    |\n')
        print('  [END]')
        print('=============================')

    def to_gpu(self):
        """
        Set gpu to the network if available.
        :return:
        """
        if self.gpu:
            print('GPU computing is available.')
            self.network.to('cuda')
            return True
        else:
            print('You use Only CPU computing.')
            return False

    def stop_learning(self):
        """
        Stop learning
        :return:
        """
        for layer in self.network.layers:
            self.network.layers[layer].train(False)

    def start_learning(self):
        """
        (Re)start learning
        :return:
        """
        for layer in self.network.layers:
            self.network.layers[layer].train(True)

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

