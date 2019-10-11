from snnlib import Spiking

if __name__ == '__main__':
    # SNN構築　入力層ニューロンの数，シミュレーション時間などを決める
    snn = Spiking(input_l=784, obs_time=300)

    # レイヤーを追加　数とニューロンモデルを指定する
    snn.add_layer(n=500, node=snn.LIF,
                  w=snn.W_RANDOM,
                  w_max=0.5, w_min=-0.1)

    # 即抑制層を追加
    snn.add_inhibit_layer()

    # データセットの選択
    snn.load_MNIST(batch=10)

    # weight mapを描画
    for i in range(3):
        snn.plot_output_weights_map(index=i, save=True, file_name='pre_wmp_'+str(i)+'.png')

    # データを順伝播させる
    snn.run(tr_size=1000)

    for i in range(3):
        snn.plot_output_weights_map(index=i, save=True, file_name='post_wmp_'+str(i)+'.png')

    # 各種結果のプロット
    snn.plot_out_voltage(1, save=True)
    snn.plot_spikes(save=True)
