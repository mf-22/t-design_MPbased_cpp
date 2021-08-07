from util import data_loader, ElementSearch

import datetime
import glob
import os
import time

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier


def sklearn_rf():
    """ sklearnに実装されているライブラリを用いてランダムフォレストを実行する関数
    """
    ## pathの区切り文字("/"か"\")をOSに応じて取得
    SEP = os.sep
    ## 現在の時刻を文字列で取得 (例)2021年7月28日18時40分39秒 => 20210728184039
    dt_now = datetime.datetime.now()
    dt_index = dt_now.strftime("%Y%m%d%H%M%S")
    ## 疑似乱数のシードを指定
    seed = np.random.randint(2**31)
    
    ## ファイル名の入力
    print("input dataset : ", end=(""))
    data_name = input()
    if not os.path.isdir("datasets" + SEP + data_name):
        print('ERROR: Cannnot find the dataset "{}"'.format(data_name))
        return -1
    ## ランダムフォレストの結果を保存するディレクトリのパスを作成
    ## (例: datasets/dataset1/RF/20210729123234/)
    dir_path = "datasets" + SEP + data_name + SEP + "RF" + SEP + dt_index + SEP
    ## 結果を保存するディレクトリを作成
    os.makedirs(dir_path)

    ## データセットを読み込む
    train_data, train_label, valid_data, valid_label, test_dataset, test_labelset \
        = data_loader.load_data("RF", data_name, dt_index)

    ## 教師データのそれぞれの特徴量について、平均0分散1になるようにスケールする
    scaler = StandardScaler()
    ## trainデータを用いてシフトと拡縮の割合を計算し変換
    train_data = scaler.fit_transform(train_data)
    ## validとtestは計算済みのシフト・拡縮幅で変換
    valid_data = scaler.transform(valid_data)
    test_dataset = [scaler.transform(test_data) for test_data in test_dataset]

    ## 特徴量ベクトルの寄与が大きい部分を出力したとき、その特徴量ベクトルのインデックスから
    ## その特徴量が何次のモーメントで何点ビット相関を計算したか求めるためのクラス
    Searcher = ElementSearch.Element_Searcher(data_name, dir_path)

    ## define a model
    forest = RandomForestClassifier(random_state=seed, n_jobs=-1)

    ## 訓練の開始
    print("\r" + "fitting...", end="")
    start = time.perf_counter()
    forest.fit(train_data, train_label)
    finish = time.perf_counter()

    ## 訓練にかかった時間を出力
    print("\r" + "fit time : {}[s]".format(finish-start))

    ## randomforestが出力する、予測のラベルを保存するディレクトリを作成
    label_path = dir_path + "predicted_labels" + SEP 
    os.makedirs(label_path)
    ## randomforestの訓練結果(識別精度)を保存するテキストファイルを用意
    result_file = open(dir_path+"result.txt", mode="w")

    ## 訓練データの結果
    train_acc = forest.score(train_data, train_label)
    print("\ntrain :", train_acc)
    result_file.write("train : {}\n".format(train_acc))
    np.save(label_path+"train_predicted_label", forest.predict(train_data))

    ## 検証データの結果
    valid_acc = forest.score(valid_data, valid_label)
    print("valid :", valid_acc)
    result_file.write("valid : {}\n".format(valid_acc))
    np.save(label_path+"valid_predicted_label", forest.predict(valid_data))

    ## テストデータの結果
    for i in range(len(test_dataset)):
        test_acc = forest.score(test_dataset[i], test_labelset[i])
        print('test{} : {}'.format(i+1, test_acc))
        result_file.write("test{} : {}\n".format(i+1, test_acc))
        np.save(label_path+"test{}_predicted_label".format(i+1), forest.predict(test_dataset[i]))
    
    ## svmの結果を保存するテキストファイルを閉じる
    result_file.close()

    ## パラメータなどの保存
    with open(dir_path+"paras.txt", mode="w") as f:
        f.write("seed : {}\n".format(seed))
        f.write("fit time : {}\n".format(finish-start))

    ## 特徴量ベクトルの重要度を取得
    importances = forest.feature_importances_
    ## 重要度が高い順(降順)に並び替え、そのときのindexのリストを取得
    indices = np.argsort(-importances)
    
    ## インデックスと係数(重要度)を送りcsvファイルに保存する
    ## インデックスの順で保存されるのでソートして置いたほうが見やすい
    Searcher.search_and_save_all(indices, importances)


if __name__ == "__main__":
    sklearn_rf()