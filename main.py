from snnlib import Spiking


if __name__ == '__main__':

    # Build SNNs and decide the number of input neurons and the simulation time.
    snn = Spiking(input_l=784, obs_time=300)

    # Add a layer and give the num of neurons and the neuron model.
    snn.add_layer(n=100,
                  node=snn.LIF,          # or snn.DIEHL_COOK
                  w=snn.W_SIMPLE_RAND,   # initialize weights
                  scale=0.3,             # scale of random intensity
                  rule=snn.SIMPLE_STDP,  # learning rule
                  nu=(1e-4, 1e-3),       # learning rate
                  )

    # Add an inhibitory layer
    snn.add_inhibit_layer(inh_w=-100)

    # Load dataset
    snn.load_MNIST()

    # Check my network architecture
    snn.print_model()

    # Gpu is available?? If available, make it use.
    snn.to_gpu()

    # Plot weight maps before training
    snn.plot(plt_type='wmps', prefix='0')

    # Make my network run
    for i in range(10):
        snn.run(1000, unsupervised=True, alpha=0.7, debug=True)  # run
        snn.plot(plt_type='wmps', prefix='{}'.format(i+1))  # plot maps

    # Plot test accuracy transition
    snn.plot(plt_type='history', prefix='result')

    # Plot weight maps after training
    snn.plot(plt_type='wmps', prefix='result')

    # Plot output spike trains after training
    snn.plot(plt_type='sp', range=10)
