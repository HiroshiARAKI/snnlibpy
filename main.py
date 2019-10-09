from snnlib import Spiking

if __name__ == '__main__':
    # SNN構築　入力層ニューロンの数，シミュレーション時間などを決める
    snn = Spiking(input_l=784, obs_time=100)

    # レイヤーを追加　数とニューロンモデルを指定する
    snn.add_layer(n=1000, node=snn.LIF)

    # データセットの選択
    snn.load_MNIST()

    # データを順伝播させる
    snn.run()

    # 各種結果のプロット
    snn.plot_out_voltage(1, save=True)
    snn.plot_spikes(save=True)
