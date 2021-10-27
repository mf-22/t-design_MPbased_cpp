from util import data_loader, label_generator

import os
import sys
import datetime
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import openpyxl


def set_seed(my_seed):
    """ seedをセットする関数
        Arg:
            my_seed := int型の疑似乱数のシードになる整数(0以上2**31以下)
    """
    #print("seed :", my_seed)
    np.random.seed(my_seed)
    tf.random.set_seed(my_seed)


def deep_learning(repeat=False):
    """ Deep Learningを行う関数。流れは以下：
          1.データ読み込み
          2.ハイパーパラメータ等の設定
          3.訓練、訓練課程の保存
          4.テストデータの評価、評価結果の保存
        Arg:
            repeat := グラフ等を表示するかのbool型のフラグ(default=False)
    """
    ## pathの区切り文字("/"か"\")をOSに応じて取得
    SEP = os.sep
    ## 現在の時刻を文字列で取得 (例)2021年7月28日18時40分39秒 => 20210728184039
    dt_now = datetime.datetime.now()
    dt_index = dt_now.strftime("%Y%m%d%H%M%S")
    ## 疑似乱数のシードを指定
    seed = np.random.randint(2**31)
    set_seed(seed)
    
    ## ファイル名の入力
    print("input dataset : ", end=(""))
    data_name = input()
    if not os.path.isdir("datasets" + SEP + data_name):
        print('ERROR: Cannnot find the dataset "{}"'.format(data_name))
        return -1
    ## deep learningの結果を保存するディレクトリのパスを作成
    ## (例: datasets/dataset1/NN/20210729123234/)
    dir_path = "datasets" + SEP + data_name + SEP + "NN" + SEP + dt_index + SEP
    ## 結果を保存するディレクトリを作成
    os.makedirs(dir_path)

    ## データセットを読み込む
    train_data, train_label, valid_data, valid_label, test_dataset, test_labelset \
        = data_loader.load_data("NN", data_name, dt_index)

    ## 教師データのそれぞれの特徴量について、平均0分散1になるようにスケールする
    scaler = StandardScaler()
    ## trainデータを用いてシフトと拡縮の割合を計算し変換
    train_data = scaler.fit_transform(train_data)
    ## validとtestは計算済みのシフト・拡縮幅で変換
    valid_data = scaler.transform(valid_data)
    test_dataset = [scaler.transform(test_data) for test_data in test_dataset]
    
    print("\n===== Start Deep Learning (Train and validation Step) =====")
    ## 特徴量ベクトルの次元を取得
    input_node_num = train_data.shape[1]
    ## 機械学習のハイパーパラメータを指定
    hidden_node_num = input_node_num #中間層のノード数
    hidden_layer_num = 1 #中間層の層数
    epoch = 100
    batch = 128

    ## 空のNNモデルを生成
    model = tf.keras.Sequential()
    ## 入力層を追加
    model.add(tf.keras.layers.Dense(input_node_num, input_dim=input_node_num))
    #model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Activation("relu"))
    ## 中間層を追加
    for _ in range(hidden_layer_num):
        #model.add(tf.keras.layers.Dense(hidden_node_num, input_dim=input_node_num))
        model.add(tf.keras.layers.Dense(hidden_node_num))
        #model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Activation("relu"))
        #model.add(tf.keras.layers.Dropout(0.5))
    ## 出力層を追加
    #model.add(tf.keras.layers.Dense(1, input_dim=input_node_num, activation="sigmoid"))
    model.add(tf.keras.layers.Dense(1, activation="sigmoid"))

    ##set learing rate(keras's adam default = 0.001)
    learning_rate = 1.0 * 10**(-5)
    ##set the optimizer, loss function and others
    opt = tf.keras.optimizers.Adam(lr=learning_rate)
    model.compile(optimizer=opt,
                  loss="binary_crossentropy",
                  metrics=["accuracy"])
    
    ## 作成したモデルの出力
    print(model.summary())
    ## 作成したモデルをテキストファイルに描き込んで保存
    sys.stdout = open(dir_path+"model.txt", mode="w")
    print(model.summary())
    sys.stdout = sys.__stdout__
    ## 作成したモデルをpngでも保存
    tf.keras.utils.plot_model(model, to_file=dir_path+"model.png")

    ## コールバックの用意
    ## エポックごとにモデルを保存するコールバック。ただしval_accuracyが良くなったとき
    mc_cb = tf.keras.callbacks.ModelCheckpoint(
        filepath=dir_path+"best_model",
        save_weights_only=False,
        monitor="val_accuracy",
        verbose=1,
        save_best_only=True,
    )
    ## モニターしている値が特定のepoch間で向上しないときに学習を打ち切るコールバック
    """
    es_cb = tf.keras.callbacks.EarlyStopping(
        monitor="val_accuracy",
        mode="max",
        min_delta=0.001,
        patience=epoch//4,
        verbose=1,
        restore_best_weights=True #モニターしている値が最も優れているときの重みを訓練終了時にロードする
    )"""

    ## 訓練データでモデルを訓練し、学習過程を取得する
    start = time.perf_counter()
    history = model.fit(
        train_data,
        train_label,
        epochs=epoch,
        batch_size=batch,
        validation_data=(valid_data, valid_label),
        callbacks=[mc_cb]
        #callbacks=[mc_cb, es_cb]
    )
    finish = time.perf_counter()
    ## 訓練にかかった時間を出力
    print("fit time : {}[s]".format(finish-start))

    ## 学習時の各エポックごとの過程をexcelファイルに一括で保存
    hist_df = pd.DataFrame(history.history)
    hist_df.to_excel(dir_path+"results.xlsx", sheet_name="history")

    ## 学習時の各エポックでのlossやaccuracyをグラフに描く
    ## 各エポックごとの誤差関数の値を取得
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]
    ## 1からepochまでの要素を持つオブジェクトを作成
    epochs = range(1, len(loss)+1)
    ## プロット
    plt.plot(epochs, loss, "bo", label="Training loss") #訓練データの誤差関数の値のプロット
    plt.plot(epochs, val_loss, "b", label="Validation loss") #検証データの誤差関数の値のプロット
    plt.title("Training and validation loss") #グラフのタイトル
    plt.xlabel("Epochs") #横軸のラベル
    plt.ylabel("Loss") #縦軸のラベル
    plt.legend() #レジェンドの出力
    plt.savefig(dir_path+"loss.png") #グラフをpngで保存
    if not repeat:
        ## deeplearningを10回とか繰り返しているときは表示しない
        plt.show() #ウィンドウでグラフの表示
    
    ## 各エポックごとの識別精度の値を取得
    acc = history.history["accuracy"]
    val_acc = history.history["val_accuracy"]
    plt.figure() #グラフのリセット
    plt.plot(epochs, acc, "bo", label="Training acc") #訓練データの精度のプロット
    plt.plot(epochs, val_acc, "b", label="Validation acc") #検証データの精度のプロット
    plt.title("Training and validation accuracy") #グラフのタイトル
    plt.xlabel("Epochs") #横軸のラベル
    plt.ylabel("Accuracy") #縦軸のラベル
    plt.legend() #レジェンドの出力
    plt.savefig(dir_path+"acc.png") #グラフをpngで保存
    if not repeat:
        plt.show() #ウィンドウでグラフの表示
    
    ## 機械学習時のパラメータの保存
    with open(dir_path+"train_parameters.txt", mode="w") as f:
        f.write("seed           : {}\n".format(seed))
        f.write("optimizer      : {}\n".format(opt))
        f.write("learning rate  : {}\n".format(learning_rate))
        f.write("#hidden layers : {}\n".format(hidden_layer_num))
        f.write("input_dim      : {}\n".format(input_node_num))
        f.write("#hidden node   : {}\n".format(hidden_node_num))
        f.write("epoch          : {}\n".format(epoch))
        f.write("batch size     : {}\n".format(batch))
        f.write("fit time       : {}\n".format(finish-start))

    ## train dataとvalid dataがどのように識別されたか保存するフォルダを作成
    os.makedirs(dir_path+"predicted_labels")
    ## 学習済みのNNにtrain dataを入力し、ラベルに変換した後保存
    train_predicted_label = label_generator.scaler_to_label(model.predict(train_data))
    np.save(dir_path+"predicted_labels/train_predicted_label.npy", train_predicted_label)
    ## 学習済みのNNにvalid dataを入力し、ラベルに変換した後保存
    valid_predicted_label = label_generator.scaler_to_label(model.predict(valid_data))
    np.save(dir_path+"predicted_labels/valid_predicted_label.npy", valid_predicted_label)
    
    ## テストデータの評価を行う
    print("\n\n===== Start test using NN (Test step) =====")
    ## テストは訓練の中で最も良かった(val_accが高かった)ときのモデルで行う、そのためにロードする
    ## EarlyStoppingのコールバックがありのときはEalryStoppingのコールバックが同じことをしてくれる
    model = tf.keras.models.load_model(dir_path+"best_model")

    ## 先程の訓練課程を保存したエクセルにテストの結果等を保存する
    wb = openpyxl.load_workbook(dir_path + "results.xlsx")
    ## 新たなシート"test_results"に書き込んでいく
    test_ws = wb.create_sheet(title="test_results")
    test_ws["B1"] = "accuracy"
    test_ws["C1"] = "count0"
    test_ws["D1"] = "count1"

    for i in range(len(test_dataset)):
        ## テストデータのリストから抜き出す
        test_data = test_dataset[i]
        test_label = test_labelset[i]
        print("\n** test{} **".format(i+1))

        ## テストデータの識別精度を取得
        data_loss, data_acc = model.evaluate(test_data, test_label)
        
        ## 学習済みのNNにtest dataを入力し、ラベルに変換し保存
        test_predicted_label = label_generator.scaler_to_label(model.predict(test_data))
        np.save(dir_path+"predicted_labels/test{}_predicted_label.npy".format(i+1), test_predicted_label)
        ## NNの予測結果が0と1の数をそれぞれ取得
        count_0 = np.count_nonzero(test_predicted_label==0)
        count_1 = np.count_nonzero(test_predicted_label==1)
        print("acc : {}, label0 : {}, label1 : {}".format(data_acc, count_0, count_1))

        ## 結果をexcelのシートに書き込む
        test_ws.cell(row=i+2, column=1, value="test{}".format(i+1))
        test_ws.cell(row=i+2, column=2, value=data_acc)
        test_ws.cell(row=i+2, column=3, value=count_0)
        test_ws.cell(row=i+2, column=4, value=count_1)
    
    ## excelのシートへの書き込みを保存して終了
    wb.save(dir_path+"results.xlsx")


if __name__ == "__main__":
    deep_learning()