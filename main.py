from snnlib import Spiking

if __name__ == '__main__':
    # SNN構築　入力層ニューロンの数，シミュレーション時間などを決める
    snn = Spiking(input_l=784, obs_time=300)

    # レイヤーを追加　数とニューロンモデルを指定する
    # STDPの学習率は(pre, post)で指定
    snn.add_layer(n=200,
                  node=snn.LIF,
                  w=snn.W_NORMAL_DIST,
                  rule=snn.SIMPLE_STDP,
                  mu=0.3, sigma=0.35,
                  nu=(1e-4, 1e-3),
                  )

    # 即抑制層を追加
    snn.add_inhibit_layer()

    # データセットの選択
    snn.load_MNIST(batch=200)

    snn.predict_test_accuracy()

    # 学習前のスパイク列を訓練データから10個プロット
    # for i in range(10):
    #     snn.plot_spikes(save=True, index=i)

    # 訓練前のweight mapを描画
    # for i in range(3):
    #     snn.plot_output_weights_map(index=i, save=True, file_name='0_wmp_'+str(i)+'.png')

    # データを順伝播させる
    snn.predict_train_accuracy()
    snn.run(tr_size=10000)
    snn.predict_train_accuracy()
    snn.predict_test_accuracy()

    # 訓練後のweight mapを描画
    for i in range(3):
        snn.plot_output_weights_map(index=i, save=True, file_name='result_wmp_'+str(i)+'.png')

    # 学習後のスパイク列を訓練データから10個プロット
    for i in range(10):
        snn.plot_spikes(save=True, index=i)
