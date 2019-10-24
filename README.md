# WrappedBindsNET
![version](https://img.shields.io/badge/version-0.1.4-lightgray.svg?style=flat)

(Last update: 2019.10.24)  
  
これはBindsNETと呼ばれるPyTorchベースのSpiking Neural Networksフレームワークをさらに使いやすくしよう，
というコンセプトのもと作成中．  
この小さなライブラリは，全て[snnlib.py](snnlib.py)に詰められていますので，各種定数などはかなり弄りやすいかと思います．  
もちろん，main.pyから直接クラス変数は変更できます．  


完全に個人利用ですが，使いたい人がいればご自由にどうぞ😎  
(結構頻繁に小さな(大したことない)アップデートをしています．) 
  
**未完成につきバグがまだある可能性があります．**   

## 実行保証環境 (Environment)
以下の環境において問題なく実行可能なことを確認しています．  

* OS.........MacOS 10.15 or Ubuntu 16.04 LTS
* Python.....3.6.* or 3.7.*
* BindsNET...0.2.5
* PyTorch....1.10 
  (GPU: torch... 1.3.0+cu92, torchvision... 0.4.1+cu92)

## Example
* Sample code
```python
from snnlib import Spiking

if __name__ == '__main__':
    # SNN構築　入力層ニューロンの数，シミュレーション時間などを決める
    snn = Spiking(input_l=784, obs_time=300)

    # レイヤーを追加　数とニューロンモデルを指定する
    # STDPの学習率は(pre, post)で指定
    snn.add_layer(n=100,
                  node=snn.LIF,
                  w=snn.W_SIMPLE_RAND,
                  rule=snn.SIMPLE_STDP,
                  scale=0.3,
                  mu=0.1, sigma=0.1,
                  nu=(1e-4, 1e-3),
                  )

    # 即抑制層を追加
    snn.add_inhibit_layer(inh_w=-100)

    # データセットの選択
    snn.load_MNIST(batch=1)

    # gpu is available??
    snn.to_gpu()

    # 訓練前のweight mapを描画
    snn.plot(plt_type='wmp', range=5, prefix='pre')

    snn.test(1000)

    # データを順伝播させる
    for _ in range(10):
        snn.run(1000)
        snn.test(1000)

    # 訓練後のweight mapを描画
    snn.plot(plt_type='wmp', range=5, prefix='result')

    # 学習後のスパイク列を訓練データから10個プロット
    snn.plot(plt_type='sp', range=10)

```

* Generated image samples
    * A weight map of pre-training 
      ![pre_training](sample_images/img1.png)  
        
    * A weight map after STDP training with 1,000 MNIST data
      ![pre_training](sample_images/img2.png)  


## BindsNET references
【docs】  
 [Welcome to BindsNET’s documentation! &mdash; bindsnet 0.2.5 documentation](https://bindsnet-docs.readthedocs.io)  
 
【Github】  
[Hananel-Hazan/bindsnet: Simulation of spiking neural networks (SNNs) using PyTorch.](https://github.com/Hananel-Hazan/bindsnet)  

【Paper】  
[BindsNET: A Machine Learning-Oriented Spiking Neural Networks Library in Python](https://www.frontiersin.org/articles/10.3389/fninf.2018.00089/full)

## 現状
* 訓練データとテストデータで精度を測定可能になった (このときSTDP学習はされない)
* 既存のSTDP学習は簡単にできる
* コメントやドキュメントが不完全
* コードの最適化が不完全
