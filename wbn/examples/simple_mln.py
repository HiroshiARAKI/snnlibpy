from ..snnlib import Spiking, PoissonEncoder
from typing import List

def DiehlCook_unsupervised_model(
        obs_time=250,
        num_of_exc=100,
        init_weights_scale=0.3,
        learning_rate=(1e-4, 1e-2),
        weight_norm=78.4,
        inh_w=-128,
        tr_size=60000,
        ts_size=10000,
        encoder=PoissonEncoder,
        intensity=128,
        gpu=False,
        epochs=5,
        plt_wmp=True,
        plt_history=True,
        plt_result_spikes=True,
        debug=True,
        **kwargs
):
    """
    Sample code: Diehl and Cook model using unsupervised STDP label assignment.
    (This is a backup model)
    :param obs_time:
    :param num_of_exc:
    :param init_weights_scale:
    :param learning_rate:
    :param weight_norm:
    :param inh_w:
    :param tr_size:
    :param ts_size:
    :param encoder:
    :param intensity:
    :param gpu:
    :param epochs:
    :param plt_wmp:
    :param plt_history:
    :param plt_result_spikes:
    :param debug:
    :return:
    """
    # Build SNNs and decide the number of input neurons and the simulation time.
    snn = Spiking(input_l=784, obs_time=obs_time)

    # Add a layer and give the num of neurons and the neuron model.
    snn.add_layer(n=num_of_exc,
                  node=snn.ADAPTIVE_LIF,  # or snn.LIF
                  w=snn.W_SIMPLE_RAND,  # initialize weights
                  scale=init_weights_scale,  # scale of random intensity
                  rule=snn.SIMPLE_STDP,  # learning rule
                  nu=learning_rate,  # learning rate
                  norm=weight_norm,              # L1 weight normalization term
                  wmax=kwargs.get('wmax', 1.0),
                  wmin=kwargs.get('wmin', -1.0),
                  )

    # Add an inhibitory layer
    snn.add_inhibit_layer(inh_w=inh_w)

    # Load dataset
    snn.load_MNIST(encoder=encoder, intensity=intensity, min_lim=kwargs.get('min_lim', 0))

    # Check your network architecture
    snn.print_model()

    # Gpu is available?? If available, make it use.
    if gpu:
        snn.to_gpu()
    else:
        snn.gpu = False

    # Plot weight maps before training
    if plt_wmp:
        snn.plot(plt_type='wmps', prefix='0', f_shape=kwargs.get('wmp_shape', (3, 3)))

    # Make my network run
    for i in range(epochs):
        snn.run(tr_size=tr_size,  # training data size
                unsupervised=True,  # do unsupervised learning?
                # alpha=0.8,           # assignment decay
                debug=debug,  # Do you wanna watch neuron's assignments?
                interval=250,  # interval of assignment
                ts_size=ts_size,        # If you have little time for experiments, be able to reduce test size
                )
        if plt_wmp:
            snn.plot(plt_type='wmps', prefix='{}'.format(i+1), f_shape=kwargs.get('wmp_shape', (3, 3)))  # plot maps

    # Plot test accuracy transition
    if plt_history:
        snn.plot(plt_type='history', prefix='result')

    # Plot weight maps after training
    if plt_wmp:
        snn.plot(plt_type='wmps', prefix='result', f_shape=kwargs.get('wmp_shape', (3, 3)))

    # Plot output spike trains after training
    if plt_result_spikes:
        snn.plot(plt_type='sp', range=10)


def MultiLayerNetwork_unsupervised_model(
        obs_time=250,
        layers: List[int] = None,
        init_weights_scale=0.3,
        learning_rate=(1e-4, 1e-2),
        weight_norm=78.4,
        inh_w=-128,
        tr_size=60000,
        ts_size=10000,
        gpu=False,
        epochs=5,
        plt_wmp=True,
        plt_history=True,
        plt_result_spikes=True,
        debug=True,
):
    """
    Simulate simple MultiLayer SNN with only full-connections.
    :param obs_time:
    :param layers: list[100, 100] is default
    :param init_weights_scale:
    :param learning_rate:
    :param weight_norm:
    :param inh_w:
    :param tr_size:
    :param ts_size:
    :param gpu:
    :param epochs:
    :param plt_wmp:
    :param plt_history:
    :param plt_result_spikes:
    :param debug:
    :return:
    """
    # Build SNNs and decide the number of input neurons and the simulation time.
    snn = Spiking(input_l=784, obs_time=obs_time)

    for num in layers:
        # Add a layer and give the num of neurons and the neuron model.
        snn.add_layer(n=num,
                      node=snn.ADAPTIVE_LIF,  # or snn.LIF
                      w=snn.W_SIMPLE_RAND,  # initialize weights
                      scale=init_weights_scale,  # scale of random intensity
                      rule=snn.SIMPLE_STDP,  # learning rule
                      nu=learning_rate,  # learning rate
                      norm=weight_norm,  # L1 weight normalization term
                      )

    # Add an inhibitory layer
    snn.add_inhibit_layer(inh_w=inh_w)

    # Load dataset
    snn.load_MNIST()

    # Check your network architecture
    snn.print_model()

    # Gpu is available?? If available, make it use.
    if gpu:
        snn.to_gpu()
    else:
        snn.gpu = False

    # Plot weight maps before training
    if plt_wmp:
        for l, n in layers:
            snn.plot(plt_type='wmps',
                     prefix='node{}-node{}_0'.format(l, l+1),
                     layer=l+1)

    # Make my network run
    for i in range(epochs):
        snn.run(tr_size=tr_size,  # training data size
                unsupervised=True,  # do unsupervised learning?
                # alpha=0.8,           # assignment decay
                debug=debug,  # Do you wanna watch neuron's assignments?
                interval=250,  # interval of assignment
                ts_size=ts_size,  # If you have little time for experiments, be able to reduce test size
                )
        if plt_wmp:
            for l, n in layers:
                snn.plot(plt_type='wmps',
                         prefix='node{}-node{}_{}'.format(l, l + 1, i + 1),
                         layer=l + 1)

    # Plot test accuracy transition
    if plt_history:
        snn.plot(plt_type='history', prefix='result')

    # Plot weight maps after training
    if plt_wmp:
        for l, n in layers:
            snn.plot(plt_type='wmps',
                     prefix='node{}-node{}_result'.format(l, l + 1),
                     layer=l + 1)

    # Plot output spike trains after training
    if plt_result_spikes:
        snn.plot(plt_type='sp', range=10)

