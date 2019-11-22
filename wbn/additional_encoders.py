from bindsnet.encoding import Encoder
import torch
import numpy as np


def fixed_frequency(datum: torch.Tensor, time: int, dt: float = 1.0):
    """
    Generates Fixed-frequency spike trains with an input image normalized as frequencies.
    :param datum: input image shape[batch, height, width]?
    :param time:
    :param dt:
    :return:
    """
    time = int(time / dt)
    shape = list(datum.shape)
    datum = np.copy(datum)

    periods = 1. / (datum*0.001 + 0.000001)  # transform frequencies to periods
    periods = periods.astype(int) + 1  # interval between two spike is periods, so needs plus one

    spike = np.arange(1, time + 1, 1)  # initialize spike [1, time+1] that means indices of array
    spikes = torch.tensor([
        spike for _ in range(784)  # also initialize all spikes of img with that initialized spike
    ]).numpy().T.reshape((time, shape[1], shape[2]))  # transpose and reshape

    # fixed-frequency spike trains generating process (this code is not smart??)
    spikes[spikes % periods != 0] = 2  # In all spikes, puts dummy number '2' into non-spike times decided each periods.
    spikes[spikes % periods == 0] = 1  # next, the number '1' into the spike-exsisting times
    spikes[spikes == 2] = 0            # and finally, changes dummy numbers to 0

    return torch.tensor(spikes)  # to tensor


def lif(datum: torch.Tensor, time: int, dt: float = 1.0, rest=-65, th=-40, ref=3, tc_decay=100):
    """
    Very simple LIF neuron spike generator.
    Membrane formula is below,
    v = v + I
    v = decay * (v - rest) + rest
    :param datum:
    :param time:
    :param dt:
    :param rest:
    :param th:
    :param ref:
    :param tc_decay:
    :return:
    """
    time = int(time / dt)
    shape = list(datum.shape)
    datum = np.copy(datum.squeeze())

    decay = np.exp(-1.0 / tc_decay)

    spikes = np.zeros((time, shape[1], shape[2]))
    v = np.full_like(datum, rest)
    refrac = np.zeros_like(v)

    for t in range(time):
        # makes neurons' membrane potentials outside a refractory period integrate input currents
        v[refrac == 0] += datum[refrac == 0]
        # and make them decay (leak)
        v = decay * (v - rest) + rest

        # neurons whose potential is higher than threshold are fired
        spikes[t][v > th] = 1
        # sets them to a refractory period
        refrac[v > th] = ref
        # and make them be back to resting potential
        v[v > th] = rest

        # makes refractory period counters decrement
        refrac[refrac > 0] -= 1

    return torch.tensor(spikes)  # to tensor


class FixedFrequencyEncoder(Encoder):
    def __init__(self, time: int, dt: float = 1.0, **kwargs):
        """
        Generates vey simple spike train by a fixed frequency, which means the all interval between two spikes are even.
        Then, each pixels values is used as a fixed frequency, so you have to normalize input images in advance.
        :param time:
        :param dt:
        :param kwargs:
        """
        super().__init__(time, dt=dt, **kwargs)
        self.enc = fixed_frequency


class LIFEncoder(Encoder):
    def __init__(self, time: int, dt: float = 1.0, rest=-65, th=-40, ref=3, tc_decay=100, **kwargs):
        """
        Generates vey simple spike train by LIF neuron, so the all interval between two spikes are also  even.
        Then, each pixels values is used as a input current, so you have to normalize input images in advance.
        WARNING: This encoder can be used when batch size is one (non-batch).
        :param time:
        :param dt:
        :param kwargs:
        """
        super().__init__(time, dt=dt, rest=rest, th=th, ref=ref, tc_decay=tc_decay, **kwargs)
        self.enc = lif
