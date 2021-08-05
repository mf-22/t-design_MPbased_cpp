from util import data_loader, label_generator

import datetime
import glob
import os
import time

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score


def sklearn_lr():
    """ scikit-learnのライブラリを用いて線形回帰を実行する関数
        2値分類を行いたいので、回帰の出力からクラス分けする
    """
    ## pathの区切り文字("/"か"\")をOSに応じて取得
    SEP = os.sep
    ## 現在の時刻を文字列で取得 (例)2021年7月28日18時40分39秒 => 20210728184039
    dt_now = datetime.datetime.now()
    dt_index = dt_now.strftime("%Y%m%d%H%M%S")
    
    ## ファイル名の入力
    print("input dataset : ", end=(""))
    data_name = input()
    if not os.path.isdir("datasets" + SEP + data_name):
        print('ERROR: Cannnot find the dataset "{}"'.format(data_name))
        return -1
    ## 線形回帰の結果を保存するディレクトリのパスを作成
    ## (例: datasets/dataset1/LR/20210729123234/)
    dir_path = "datasets" + SEP + data_name + SEP + "LR" + SEP + dt_index + SEP
    ## 結果を保存するディレクトリを作成
    os.makedirs(dir_path)

    ## データセットを読み込む
    train_data, train_label, valid_data, valid_label, test_dataset, test_labelset \
        = data_loader.load_data("LR", data_name, dt_index)
    ## define a model set to do normalization
    lr = LinearRegression(normalize=True, n_jobs=-1)

    ## 訓練の開始
    print("\r" + "fitting...", end="")
    start = time.perf_counter()
    lr.fit(train_data, train_label)
    finish = time.perf_counter()

    ## 訓練にかかった時間を出力
    print("\r" + "fit time : {}[s]".format(finish-start))

    ## 回帰モデルが出力する、予測のラベルを保存するディレクトリを作成
    label_path = dir_path + "predicted_labels" + SEP 
    os.makedirs(label_path)
    ## 線形回帰の訓練結果(識別精度)を保存するテキストファイルを用意
    result_file = open(dir_path+"result.txt", mode="w")

    ## trainデータの予測ラベルを取得
    predicted_train = label_generator.scaler_to_label(lr.predict(train_data))
    ## 識別精度(TP/(TP+FP))を計算し出力
    train_acc = accuracy_score(train_label, predicted_train)
    print("\ntrain :", train_acc)
    ## 識別精度と識別結果を保存
    result_file.write("train : {}\n".format(train_acc))
    np.save(label_path+"train_predicted_label", predicted_train)

    ## validデータの予測ラベルを取得
    predicted_valid = label_generator.scaler_to_label(lr.predict(valid_data))
    ## 識別精度(TP/(TP+FP))を計算し出力
    valid_acc = accuracy_score(valid_label, predicted_valid)
    print("valid :", valid_acc)
    ## 識別精度と識別結果を保存
    result_file.write("valid : {}\n".format(valid_acc))
    np.save(label_path+"valid_predicted_label", predicted_valid)

    ## テストデータの結果
    for i in range(len(test_dataset)):
        ## テストデータの予測ラベルを取得
        predicted_test = label_generator.scaler_to_label(lr.predict(test_dataset[i]))
        ## 識別精度(TP/(TP+FP))を計算し出力
        test_acc = accuracy_score(test_labelset[i], predicted_test)
        print("test{} : {}".format(i+1, test_acc))
        ## 識別精度と識別結果を保存
        result_file.write("test{} : {}\n".format(i+1, test_acc))
        np.save(label_path+"test{}_predicted_label".format(i+1), predicted_test)
    
    ## svmの結果を保存するテキストファイルを閉じる
    result_file.close()

    ## パラメータなどの保存
    with open(dir_path+"paras.txt", mode="w") as f:
        f.write("fit time : {}\n".format(finish-start))


if __name__ == "__main__":
    sklearn_lr()