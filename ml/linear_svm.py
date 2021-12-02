from util import data_loader

import datetime
import glob
import os
import time

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC


def sklearn_linear_svm(shuffle=True):
    """ scikit-learnのライブラリを用いてLinear support vector machineを実行する関数
        Arg:
            shuffle := trainとvalidのデータをランダムに分割するかのbool型のフラグ(default=True)
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
    ## 線形サポートベクターマシンの結果を保存するディレクトリのパスを作成
    ## (例: datasets/dataset1/LinearSVM/20210729123234/)
    dir_path = "datasets" + SEP + data_name + SEP + "LinearSVM" + SEP + dt_index + SEP
    ## 結果を保存するディレクトリを作成
    os.makedirs(dir_path)

    ## データセットを読み込む
    train_data, train_label, valid_data, valid_label, test_dataset, test_labelset \
        = data_loader.load_data("LinearSVM", data_name, dt_index)
    
    if shuffle:
        ## 疑似乱数のシードを指定
        seed = np.random.randint(2**31)
        ## trainとvalidのデータを結合してランダムに分割する
        num_train_data = train_data.shape[0]
        temp_data = np.concatenate([train_data, valid_data], axis=0)
        temp_label = np.concatenate([train_label, valid_label], axis=0)
        from sklearn.model_selection import train_test_split
        train_data, valid_data, train_label, valid_label = train_test_split(temp_data, temp_label, train_size=num_train_data, random_state=seed)

    ## 教師データのそれぞれの特徴量について、平均0分散1になるようにスケールする
    scaler = StandardScaler()
    ## trainデータを用いてシフトと拡縮の割合を計算し変換
    train_data = scaler.fit_transform(train_data)
    ## validとtestは計算済みのシフト・拡縮幅で変換
    valid_data = scaler.transform(valid_data)
    test_dataset = [scaler.transform(test_data) for test_data in test_dataset]

    ## define a model
    """ Ref: https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html#sklearn.svm.LinearSVC  
        dual: bool, default=True
            Select the algorithm to either solve the dual or primal optimization problem.
            Prefer dual=False when n_samples > n_features.
    """
    clf = LinearSVC(dual=False)

    ## 訓練の開始
    print("\r" + "fitting...", end="")
    start = time.perf_counter()
    clf.fit(train_data, train_label)
    finish = time.perf_counter()

    ## 訓練にかかった時間を出力
    print("\r" + "fit time : {}[s]".format(finish-start))
    
    ## svmが出力する、予測のラベルを保存するディレクトリを作成
    label_path = dir_path + "predicted_labels" + SEP 
    os.makedirs(label_path)
    ## svmの訓練結果(識別精度)を保存するテキストファイルを用意
    result_file = open(dir_path+"result.txt", mode="w")

    ## 結果の出力
    ## 訓練データの結果
    train_acc = clf.score(train_data, train_label)
    print("\ntrain :", train_acc)
    result_file.write("train : {}\n".format(train_acc))
    np.save(label_path+"train_predicted_label", clf.predict(train_data))

    ## 検証データの結果
    valid_acc = clf.score(valid_data, valid_label)
    print("valid :", valid_acc)
    result_file.write("valid : {}\n".format(valid_acc))
    np.save(label_path+"valid_predicted_label", clf.predict(valid_data))
    
    ## テストデータの結果
    for i in range(len(test_dataset)):
        test_acc = clf.score(test_dataset[i], test_labelset[i])
        print('test{} : {}'.format(i+1, test_acc))
        result_file.write("test{} : {}\n".format(i+1, test_acc))
        np.save(label_path+"test{}_predicted_label".format(i+1), clf.predict(test_dataset[i]))
    
    ## svmの結果を保存するテキストファイルを閉じる
    result_file.close()

    ## パラメータなどの保存
    with open(dir_path+"paras.txt", mode="w") as f:
        f.write("dual     : False\n")
        f.write("fit time : {}\n".format(finish-start))
        f.write("shuffle  : {}\n".format(shuffle))
        if shuffle:
            f.write("seed     : {}\n".format(seed))


if __name__ == "__main__":
    sklearn_linear_svm()