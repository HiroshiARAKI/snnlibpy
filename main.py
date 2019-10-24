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
