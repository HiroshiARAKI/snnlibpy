# WrappedBindsNET
(Last update: 2019.10.11)  
  
これはBindsNETと呼ばれるPyTorchベースのSpiking Neural Networksフレームワークをさらに使いやすくしよう，
というコンセプトのもと作成中．  
この小さなライブラリは，全て[snnlib.py](snnlib.py)に詰められていますので，各種定数などはかなり弄りやすいかと思います．  
もちろん，main.pyから直接クラス変数は変更できます．  


完全に個人利用ですが，使いたい人がいればご自由にどうぞ😎  
(結構頻繁に小さな(大したことない)アップデートをしています．)  

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
    # STDPの学習率は(pre, post)で指定
    snn.add_layer(n=300, node=snn.LIF,
                  w=snn.W_SIMPLE_RAND,
                  rule=snn.SIMPLE_STDP,
                  scale=0.3,
                  nu=(1e-3, 1e-2),
                  )

    # 即抑制層を追加
    snn.add_inhibit_layer()

    # データセットの選択
    snn.load_MNIST(batch=10)

    # 学習前のスパイク列を訓練データから10個プロット
    for i in range(10):
        snn.plot_spikes(save=True, index=i)

    # 訓練前のweight mapを描画
    for i in range(5):
        snn.plot_output_weights_map(index=i, save=True, file_name='pre_wmp_'+str(i)+'.png')

    # データを順伝播させる
    snn.run(tr_size=10000)

    # 訓練後のweight mapを描画
    for i in range(5):
        snn.plot_output_weights_map(index=i, save=True, file_name='result_wmp_'+str(i)+'.png')

    # 学習後のスパイク列を訓練データから10個プロット
    for i in range(10):
        snn.plot_spikes(save=True, index=i)
```

## BindsNET references
【docs】  
 [Welcome to BindsNET’s documentation! &mdash; bindsnet 0.2.5 documentation](https://bindsnet-docs.readthedocs.io)  
 
【Github】  
[Hananel-Hazan/bindsnet: Simulation of spiking neural networks (SNNs) using PyTorch.](https://github.com/Hananel-Hazan/bindsnet)  

【Paper】  
[BindsNET: A Machine Learning-Oriented Spiking Neural Networks Library in Python](https://www.frontiersin.org/articles/10.3389/fninf.2018.00089/full)

## 現状
* 既存のSTDP学習は簡単にできる
* GPU (CUDA9以降は対応？)
* コメントやドキュメントが不完全
* コードの最適化が不完全
