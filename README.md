# WrappedBindsNET
(Last update: 2019.10.11)  
  
これはBindsNETと呼ばれるPyTorchベースのSpiking Neural Networksフレームワークをさらに使いやすくしよう，
というコンセプトのもと作成中．  

完全に個人利用ですが，使いたい人がいればご自由にどうぞ．

## 実行保証環境 (Environment)
以下の環境において問題なく実行可能なことを確認しています．  

* OS.........MacOS 10.15 or Ubuntu 16.04 LTS
* Python.....3.6.* or 3.7.*
* BindsNET...0.2.5
* PyTorch....1.10

## Example
* Code
```python
from snnlib import Spiking

if __name__ == '__main__':
    # SNN構築　入力層ニューロンの数，シミュレーション時間などを決める
    snn = Spiking(input_l=784, obs_time=300)

    # レイヤーを追加　数とニューロンモデルを指定する
    # 以下の場合は重みを平均0.2，分散0.2で初期化
    # STDPの学習率は(pre, post)=(1e-3, 1e-3)で指定
    snn.add_layer(n=500, node=snn.LIF,
                  w=snn.W_NORMAL_DIST,
                  mu=0.2, sigma=0.2,
                  nu=(1e-3, 1e-3))

    # 即抑制層を追加
    snn.add_inhibit_layer()

    # データセットの選択
    snn.load_MNIST(batch=10)

    # weight mapを描画
    for i in range(5):
        snn.plot_output_weights_map(index=i, save=True, file_name='pre_wmp_'+str(i)+'.png')

    # データを順伝播させる
    snn.run()

    for i in range(5):
        snn.plot_output_weights_map(index=i, save=True, file_name='result_wmp_'+str(i)+'.png')

    # 学習後のスパイク列をプロット
    snn.plot_spikes(save=True)

```

## BindsNET references
【docs】  
 [Welcome to BindsNET’s documentation! &mdash; bindsnet 0.2.5 documentation](https://bindsnet-docs.readthedocs.io)  
 
【Github】  
[Hananel-Hazan/bindsnet: Simulation of spiking neural networks (SNNs) using PyTorch.](https://github.com/Hananel-Hazan/bindsnet)  

【Paper】  
[BindsNET: A Machine Learning-Oriented Spiking Neural Networks Library in Python](https://www.frontiersin.org/articles/10.3389/fninf.2018.00089/full)

## 現状
* 既存のSTDP学習は簡単できる
* GPU (CUDA9以降は対応？)
* コメントやドキュメントが不完全
* コードの最適化が不完全
