from wbn import Spiking


if __name__ == '__main__':

    # Build SNNs and decide the number of input neurons and the simulation time.
    snn = Spiking(input_l=784, obs_time=300)

    # Add a layer and give the num of neurons and the neuron model.
    snn.add_layer(n=100,
                  node=snn.DIEHL_COOK,          # or snn.DIEHL_COOK
                  w=snn.W_SIMPLE_RAND,   # initialize weights
                  scale=0.4,             # scale of random intensity
                  rule=snn.SIMPLE_STDP,  # learning rule
                  nu=(1e-4, 1e-2),       # learning rate
                  # norm=150,              # L1 weight normalization term
                  )

    # Add an inhibitory layer
    snn.add_inhibit_layer(inh_w=-128)

    # Load dataset
    snn.load_MNIST()

    # Check your network architecture
    snn.print_model()

    # If you use a small network, your network computation by GPU may be more slowly than CPU.
    # So you can change directly whether using GPU or not as below.
    snn.gpu = False

    # Gpu is available?? If available, make it use.
    snn.to_gpu()

    # Plot weight maps before training
    snn.plot(plt_type='wmps', prefix='0', f_shape=(10, 10))

    # Make my network run
    for i in range(5):
        snn.run(tr_size=60000,       # training data size
                unsupervised=True,   # do unsupervised learning?
                # alpha=0.8,           # assignment decay
                debug=True,          # Do you wanna watch neuron's assignments?
                interval=250,        # interval of assignment
                # ts_size=5000,        # If you have little time for experiments, be able to reduce test size
                )

        snn.plot(plt_type='wmps', prefix='{}'.format(i+1), f_shape=(10, 10))  # plot maps

    # Plot test accuracy transition
    snn.plot(plt_type='history', prefix='result')

    # Plot weight maps after training
    snn.plot(plt_type='wmps', prefix='result', f_shape=(10, 10))

    # Plot output spike trains after training
    snn.plot(plt_type='sp', range=10)
