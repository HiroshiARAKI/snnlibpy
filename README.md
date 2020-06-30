# WrappedBindsNET
![update](https://img.shields.io/badge/last%20update-2020.06.30-lightgray.svg?style=flat)

これはBindsNETと呼ばれるPyTorchベースのSpiking Neural Networksフレームワークをさらに使いやすくしよう，
というコンセプトのもと作成中．  
この小さなライブラリは，大体[snnlib.py](wbn/snnlib.py)に詰められているので，各種定数などはかなり弄りやすいかと思います．  
もちろん，main.pyから直接クラス変数は変更できます．  
完全に個人利用ですが，使いたい人がいればご自由にどうぞ   
(結構頻繁に小さな(大したことない)アップデートをしています．)   
  
I am making a tiny and user friendly library of Spiking Neural Networks with BindsNET.  
All functions are packed to only [snnlib.py](wbn/snnlib.py), so you can use easily.  
This library is used by private myself, but if you want to use it, feel free to use.  
  
**未完成につきバグがまだある可能性があります．(Maybe, there are bugs because this is incompletely.)**   

## 実行保証環境 (Environment)
以下の環境において問題なく実行可能なことを確認しています．  

* OS.........MacOS 10.15 or Ubuntu 16.04 LTS
* Python.....3.6.* or 3.7.* (, or later)
* BindsNET...0.2.7 (not worked on < 0.2.7)
* PyTorch....1.10 
  (GPU: torch... 1.3.0+cu92, torchvision... 0.4.1+cu92)

## Example
* Sample code
```python
from wbn import Spiking


if __name__ == '__main__':

    # Build SNNs and decide the number of input neurons and the simulation time.
    snn = Spiking(input_l=784, obs_time=300, dt=0.5)
    snn.IMAGE_DIR += 'diehl/'

    # Add a layer and give the num of neurons and the neuron model.
    snn.add_layer(n=100,
                  node=snn.DIEHL_COOK,          # or snn.DIEHL_COOK
                  w=snn.W_SIMPLE_RAND,   # initialize weights
                  rule=snn.SIMPLE_STDP,  # learning rule
                  nu=(1e-4, 1e-2),       # learning rate
                  )

    # Add an inhibitory layer
    snn.add_inhibit_layer(inh_w=-128)

    # Load dataset
    snn.load_MNIST()

    # Check your network architecture
    snn.print_model()

    # If you use a small network, your network computation by GPU may be more slowly than CPU.
    # So you can change directly whether using GPU or not as below.
    # snn.gpu = False

    # Gpu is available?? If available, make it use.
    snn.to_gpu()

    # Plot weight maps before training
    snn.plot(plt_type='wmps', prefix='0', f_shape=(10, 10))

    # Make my network run
    for i in range(3):
        snn.run()

        snn.plot(plt_type='wmps', prefix='{}'.format(i+1), f_shape=(10, 10))  # plot maps

    # Plot test accuracy transition
    snn.plot(plt_type='history', prefix='result')

    # Plot weight maps after training
    snn.plot(plt_type='wmps', prefix='result', f_shape=(10, 10))

    # Plot output spike trains after training
    snn.plot(plt_type='sp', range=10)

    print(snn.history)
```

or very simply,
```python
from wbn import DiehlCook_unsupervised_model  # packed sample simulation code

DiehlCook_unsupervised_model()
```
is ok (actually this function is my backup data, so it's good for you to use this when you check whether it works properly).

* Generated image samples
    * A weight map before training 
      ![pre_training](sample_images/diehl0.png)  
        
    * A weight map after STDP training with 1,0000 MNIST data
      ![pre_training](sample_images/diehl10000.png)   


## BindsNET references
【docs】  
 [Welcome to BindsNET’s documentation! &mdash; bindsnet 0.2.5 documentation](https://bindsnet-docs.readthedocs.io)  
 
【Github】  
[Hananel-Hazan/bindsnet: Simulation of spiking neural networks (SNNs) using PyTorch.](https://github.com/Hananel-Hazan/bindsnet)  

【Paper】  
[BindsNET: A Machine Learning-Oriented Spiking Neural Networks Library in Python](https://www.frontiersin.org/articles/10.3389/fninf.2018.00089/full)
  
