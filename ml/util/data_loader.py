from . import label_generator, read_parameter

import glob
import os

import numpy as np


def load_data(ml_alg, data_name, dt_index, k=0, kp_list=[]):
    """ datasetフォルダの中のフォルダを指定し、教師データを読み込んで返す関数。
        元データはcsvファイルかnpyファイルで、最大のビット相関と20次までのモーメント
        を計算している。ここから必要な部分を抜き出す。
        PCAのときと機械学習アルゴリズムのときで戻り値が異なる：
        ・PCA => 読み込んだデータのリスト
        ・機械学習系 => 機械学習用に成型されたデータ
        Arguments:
            ml_alg(string)    := 機械学習アルゴリズムの名前の文字列(例: NN, svm, ...)
            data_name(string) := データセットフォルダの文字列(例: dataset1)
            dt_index(string)  := ファイル名の文字列(例: 20210728161623)
            k(int)            := 計算するビット相関の数(default=0)。0なら後で標準入力
            kp_list(list)     := 計算するモーメントの次数のリスト(default=[])。空なら後で標準入力
        Returns:
            PCA:
                train_data    := 訓練用のデータセット(2次元のndarray, データセット数×特徴量の次元)
                train_info    := 結合してtrain_dataを作ったときの、結合する前のそれぞれのデータの、
                                 種類とデータサイズのリストを要素に持つリスト
                valid_data    := 検証用のデータセット(2次元のndarray, データセット数×特徴量の次元)
                valid_info    := 結合してvalid_dataを作ったときの、結合する前のそれぞれのデータの、
                                 種類とデータサイズのリストを要素に持つリスト
                test_dataset  := テスト用のデータセットを要素に持つリスト(2次元のndarrayを要素に持つリスト)
                test_infoset  := 結合する前のそれぞれのデータの種類とデータサイズのリストを要素に持つリストのリスト

            Machine Learning:
                train_data    := 訓練用のデータセット(2次元のndarray, データセット数×特徴量の次元)
                train_label   := 訓練用のデータのラベル(1次元のndarrayで要素が0か1, 大きさがデータセット数)
                valid_data    := 検証用のデータセット(2次元のndarray, データセット数×特徴量の次元)
                valid_label   := 検証用のデータのラベル(1次元のndarrayで要素が0か1, 大きさがデータセット数)
                test_dataset  := テスト用のデータセットを要素に持つリスト(2次元のndarrayを要素に持つリスト)
                test_labelset := テスト用のデータのラベルを要素に持つリスト(1次元のndarrayを要素に持つリスト)
    """
    ## pathの区切り文字("/"か"\")をOSに応じて取得
    SEP = os.sep

    ## 量子ビット数がいくつだったか取得
    Nq = read_parameter.get_num_qubits(data_name)
    
    ## ビット相関を計算したとき、考えられる最大の組み合わせの数を計算
    Nq_prime = 2 ** Nq - 1

    ## データセットの中のパスを探索する
    ## "train"フォルダーを検索する
    dataset_path = glob.glob('datasets/{}/train/*_*.npy'.format(data_name))
    dataset_path.extend(glob.glob('datasets/{}/train/*_*.csv'.format(data_name)))
    ##search in "valid" folder
    dataset_path.extend(glob.glob('datasets/{}/valid/*_*.npy'.format(data_name)))
    dataset_path.extend(glob.glob('datasets/{}/valid/*_*.csv'.format(data_name)))
    ##search in "test" folder
    dataset_path.extend(glob.glob('datasets/{}/test/**/*_*.npy'.format(data_name), recursive=True))
    dataset_path.extend(glob.glob('datasets/{}/test/**/*_*.csv'.format(data_name), recursive=True))

    ## globによって得られるリストの順序がOSに依存していので、ソートして同じになるようにしておく
    dataset_path.sort()
    ## テストデータセットの数を取得
    testset_num = len(glob.glob('datasets/{}/test/*'.format(data_name)))
    
    ## 探索してきたパスの出力
    print("Extracted path:")
    print(np.array(dataset_path))
    print("num_testset =", testset_num)

    ## 抽出するビット相関数kとモーメントの次数k'を入力
    if k == 0:
        ## ビット相関の数を選択して抽出するのは未実装、必要になったら
        #print('input k(Max={}) :'.format(Nq), end=(' '))
        #k = int(input())
        k = Nq
    if len(kp_list) == 0:
        print('\ninput k prime')
        print('  input min(min=1) :', end=(' '))
        kp_min = int(input())
        print('  input max(max=20) :', end=(' '))
        kp_max = int(input())
        print('  input step(default=1) :', end=(' '))
        kp_step = int(input())
        #kp_step = 1
        kp_list = [i for i in range(kp_min, kp_max+1, kp_step)]
    print("Dim of moment =", kp_list)
    
    """ 抽出してきたデータを保持しておくリスト。要素数4のリストを要素に持つ
          [0] : データの種類(haar, clif, lrc, ...)
          [1] : データの使用目的(train, valid, test1, ...)
          [2] : 特徴量ベクトル(データサイズ×特徴量の2次元ndarray)
          [3] : 特徴量ベクトルのデータサイズ(int)
    """
    data_list = []

    ## 順番に抽出していく
    for data_path in dataset_path:
        ## 今操作しているデータが何か出力
        print("\rNow => {}   ".format(data_path), end="")

        ## データの種類を特定(haar, clif, ...)
        if "haar" in data_path:
            data_type = "haar"
        elif "clif" in data_path:
            data_type = "clif"
        elif "lrc" in data_path:
            data_type = "lrc"
        elif "rdc" in data_path:
            data_type = "rdc"
        
        ## データの使用目的を特定(train, valid, ...)
        if "train" in data_path:
            purpose = "train"
        elif "valid" in data_path:
            purpose = "valid"
        elif "test" in data_path:
            count = 1
            while True:
                if "test{}".format(count)+SEP in data_path:
                    purpose = "test{}".format(count)
                    break
                else:
                    count += 1
        
        ## ファイルを形式に応じて読み込む
        if "npy" in data_path:
            ## npyファイルなら普通に読み込み、単精度で
            data = np.load(data_path).astype(np.float32)
        elif "csv" in data_path:
            ## csvファイルの場合は区切り文字をカンマで読み込む
            data = np.loadtxt(data_path, delimiter=",").astype(np.float32)
            ## csvファイルをnpy形式(バイナリ)に置き換えてしまう
            ## 機械学習し直すときにnpyの方が読み込みが早くファイルサイズも小さいので
            np.save(data_path.split(".csv")[0], data)
            os.remove(data_path)
            
        ## 必要な部分を抽出する
        data = np.concatenate(
            [data[:, Nq_prime*i:(Nq_prime*i)+Nq_prime] for i in kp_list],
            axis=1
        )

        ## リストに追加
        data_list.append([data_type, purpose, data, data.shape[0]])
    print("\r" + " "*(len(dataset_path[-1])+12) + "\n")
 
    ## 抽出したパラメータを保存
    with open("datasets/{}/{}/{}/extract_parameters.txt".format(data_name, ml_alg, dt_index), mode="w") as f:
        f.write('k : {}\nk` : {}\n'.format(k, kp_list))
    

    ## 訓練用のデータを作る。データのリストから目的がtrainであるデータを抜き出して結合する
    train_data = np.concatenate([temp_list[2] for temp_list in data_list if "train" == temp_list[1]], axis=0)
    ## 検証用のデータを作る。データのリストから目的がtrainであるデータを抜き出して結合する
    valid_data = np.concatenate([temp_list[2] for temp_list in data_list if "valid" == temp_list[1]], axis=0)
    ## テスト用のデータを作る。テストデータは複数与えたいことがあるので、データセットを要素に持つリストで作成する
    test_dataset = []
    for i in range(1, testset_num+1):
        ## 検証用のデータを作る、データのリストから目的がtrainであるデータを抜き出して結合する
        test_dataset.append(np.concatenate([temp_list[2] for temp_list in data_list if "test{}".format(i) == temp_list[1]], axis=0))
        
    if ml_alg == "PCA":
        """ PCAのときは、ラベルは"無し"か"機械学習アルゴリズムが予測した結果"を使うのでここではラベルは作らない
            そのかわり、それぞれのデータの種類とデータサイズを保持するリストを作成し返す
        """
        ## 訓練データを構成するそれぞれのデータについて、そのデータの種類とデータサイズを保持するリストを作る
        train_info = [[temp_list[0], temp_list[3]] for temp_list in data_list if "train" == temp_list[1]]
        ## 検証データを構成するそれぞれのデータについて、そのデータの種類とデータサイズを保持するリストを作る
        valid_info = [[temp_list[0], temp_list[3]] for temp_list in data_list if "valid" == temp_list[1]]
        ## テストデータを構成するそれぞれのデータについて、そのデータの種類とデータサイズを保持するリストを作る
        test_infoset = []
        for i in range(1, testset_num+1):
            temp_info = [[temp_list[0], temp_list[3]] for temp_list in data_list if "test{}".format(i) == temp_list[1]]
            test_infoset.append(temp_info)

        return train_data, train_info, valid_data, valid_info, test_dataset, test_infoset

    else:
        """ 機械学習を行う場合はラベルを作成し返す
        """
        ## 訓練用のデータのラベルを作る。データのリストから目的がtrainであるデータを抜き出し、ラベルを生成した後結合する
        train_label = np.concatenate(
            [ label_generator.generate_label(temp_list[0], temp_list[3]) for temp_list in data_list if "train" == temp_list[1] ],
            axis=0
        )
        ## 検証用のデータのラベルを作る。データのリストから目的がvalidであるデータを抜き出し、ラベルを生成した後結合する
        valid_label = np.concatenate(
            [ label_generator.generate_label(temp_list[0], temp_list[3]) for temp_list in data_list if "valid" == temp_list[1] ],
            axis=0
        )
        ## テスト用のデータのラベルを作る。テストデータは複数与えたいことがあるので、データセットを要素に持つリストで作成する
        test_labelset = []
        for i in range(1, testset_num+1):
            ## 検証用のデータを作る、データのリストから目的がtrainであるデータを抜き出して結合する
            temp_label = np.concatenate(
                [ label_generator.generate_label(temp_list[0], temp_list[3]) for temp_list in data_list if "test{}".format(i) == temp_list[1] ],
                axis=0
            )
            test_labelset.append(temp_label)

        return train_data, train_label, valid_data, valid_label, test_dataset, test_labelset


def load_pred_labels(path):
    """ PCAの出力で、識別器の出力(予測ラベル)の情報も合わせてプロットしたい。
        そのときに保存されている予測ラベルを読み込み、含まれるデータの情報と合わせて返す関数。
        Arg:
            path(string) := 予測ラベルが保存されているディレクトリのパス
        Returns:
            train_label(ndarray) := 訓練データの予測ラベル
            valid_label(ndarray) := 検証データの予測ラベル
            test_labelset(list of ndarray) := テストデータの予測ラベルを要素に持つリスト
    """
    ## 指定されたディレクトリにあるラベルが保存されているnpyファイルのパスを取得
    label_path = sorted(glob.glob(path + "/*.npy"))
    ## 順番にデータを取得し、データの使用目的に応じて保存する
    test_label_listset = []
    for each_path in label_path:
        label = np.load(each_path)
        if "train" in each_path:
            train_label = label
        elif "valid" in each_path:
            valid_label = label
        elif "test" in each_path:
            test_label_listset.append(label)
        else:
            print("CAUTION: Cannot detect the label, the name of .npy file will be invalid.")

    return train_label, valid_label, test_label_listset