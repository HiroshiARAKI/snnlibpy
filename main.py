from snnlib import Spiking


if __name__ == '__main__':

    # Build SNNs and decide the number of input neurons and the simulation time.
    snn = Spiking(input_l=784, obs_time=300)

    # Add a layer and give the num of neurons and the neuron model.
    snn.add_layer(n=100,
                  node=snn.LIF,
                  w=snn.W_SIMPLE_RAND,  # initialize weights
                  scale=0.3,
                  rule=snn.SIMPLE_STDP,  # learning rule
                  nu=(1e-4, 1e-3),  # learning rate
                  )

    # Add an inhibitory layer
    snn.add_inhibit_layer(inh_w=-100)

    # Load dataset
    snn.load_MNIST(batch=1)

    # Check my network architecture
    snn.print_model()

    # Gpu is available?? If available, make it use.
    snn.to_gpu()

    # Plot weight maps before training
    snn.plot(plt_type='wmp', range=5, prefix='pre')

    # Calculate test accuracy before training
    snn.test(1000)

    # Make my network run
    for _ in range(10):
        snn.run(1000)  # run
        snn.test(1000)  # and predict

    # Plot weight maps after training
    snn.plot(plt_type='wmp', range=5, prefix='result')

    # Plot output spike trains after training
    snn.plot(plt_type='sp', range=10)
