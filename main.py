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
        snn.plot_output_weights_map(index=i, save=True, file_name='0_wmp_'+str(i)+'.png')

    # データを順伝播させる
    snn.run(tr_size=1000)

    for i in range(5):
        snn.plot_output_weights_map(index=i, save=True, file_name='1000_wmp_'+str(i)+'.png')

    # 学習後のスパイク列をプロット
    snn.plot_spikes(save=True)
