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
    periods = periods.astype(int)

    spike = np.arange(1, time + 1, 1)  # initialize spike [1, time+1]
    spikes = torch.tensor([
        spike for _ in range(784)  # also initialize all spikes of img with that initialized spike
    ]).numpy().T.reshape((time, shape[1], shape[2]))  # transpose and reshape

    # fixed-frequency spike trains generating process (this code is not smart??)
    spikes[spikes % periods != 0] = 2  # In all spikes, puts dummy number '2' into non-spike times decided each periods.
    spikes[spikes % periods == 0] = 1  # next, the number '1' into the spike-exsisting times
    spikes[spikes == 2] = 0            # and finally, changes dummy numbers to 0

    return torch.tensor(spikes)  # to tensor


class FixedFrequencyEncoder(Encoder):
    def __init__(self, time: int, dt: float = 1.0, **kwargs):
        """
        Generates vey simple spike train by a fixed frequency, which means the all interval between two spikes are even.
        Then, each pixels values is used as a fixed frequency, so you have to normalize input images in advance.
        :param time:
        :param dt:
        :param frequency:
        :param kwargs:
        """
        super().__init__(time, dt=dt, **kwargs)
        self.enc = fixed_frequency
