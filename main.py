from snnlib import Spiking

if __name__ == '__main__':
    snn = Spiking(input_l=784, obs_time=100)
    snn.add_lif_layer(n=1000)

    # snn.run(epoch=10)
    snn.load_MNIST()
    snn.train_STDP()
