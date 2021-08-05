from util import data_loader, read_parameter, ElementSearch

import datetime
import glob
import os
import time
import copy

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


def draw_train_base(label_exist, axes, train_info, train_data, train_label):
    """ 訓練データのプロットの上に検証データやテストデータのプロットを重ねたい。
        そのときの、訓練データのプロットを行う関数
        Args:
            label_exist(int)             := ラベルを用いてプロットするかどうか、0ならしない
            axes(matloptlibのaxesクラス) := 空のグラフ
            train_info(list)             := データの種類と、その教師データのデータ数を覚えておくためのリスト
            train_data(ndarray)          := 訓練データの主成分ベクトル
            train_label(ndarray)         := 訓練データの予測ラベル(default=0)
    """
    p = 0 #リストの位置を示すポインタ
    for i, info in enumerate(train_info):
        if info[0] == "haar":
            color = "blue" #Haar-trainを青
        elif info[0] == "clif":
            color = "red" #クリフォード系-trainを赤
        else:
            color = "black" #他はひとまず黒にしておく
        
        if label_exist == 0:
            axes.scatter(train_data[p:p+info[1],0], train_data[p:p+info[1],1], label=info[0]+"-train", marker='o', alpha=0.5, s=15, c=color)
        else:
            ## 結合して作成した教師データのうちの、1つのデータのラベルを抜き出す
            partial_label = train_label[p:p+info[1]]
            ## 特徴量ベクトルのうち、0(Haar)または1(clif)と予測されたベクトルをそれぞれ保存するリスト
            pred_0, pred_1 = [], []
            ## リスト内包表記でそれぞれのリストに分ける
            [pred_0.append(val) if partial_label[j] == 0 else pred_1.append(val) for j,val in enumerate(train_data[p:p+info[1]])]
            ## 散布図として描き込む
            if len(pred_0) != 0:
                pred_0 = np.array(pred_0)
                axes.scatter(pred_0[:,0], pred_0[:,1], label=info[0]+"-train:pred0", marker='o', alpha=0.5, s=15, c=color) #0と予測されたデータは○でプロット
            if len(pred_1) != 0:
                pred_1 = np.array(pred_1)
                axes.scatter(pred_1[:,0], pred_1[:,1], label=info[0]+"-train:pred1", marker='*', alpha=0.5, s=15, c=color) #1と予測されたデータは☆でプロット
        ## ヒストグラムを描く
        p += info[1] #ポインタを進める


def sklearn_pca(repeat=False):
    """ scikit-learnのライブラリを利用して主成分分析を行う関数
        Arg:
            repeat := グラフ等を表示するかのbool型のフラグ(default=False)
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
    ## deep learningの結果を保存するディレクトリのパスを作成
    ## (例: datasets/dataset1/PCA/20210729123234/)
    dir_path = "datasets" + SEP + data_name + SEP + "PCA" + SEP + dt_index + SEP
    ## 結果を保存するディレクトリを作成
    os.makedirs(dir_path)

    ## 機械学習アルゴリズムの実行により得られる予測ラベルが保存されているフォルダを探索
    label_path = sorted(glob.glob("datasets/"+data_name+"/**/predicted_labels", recursive=True))

    ## プロットの際に用いる予測済みのラベルとしてどれを使うか選択する
    if len(label_path) == 0:
        print("\nPredicted labels were not found.")
        label_choice = 0
    else:
        print("\nChoose the label to plot picture :")
        print("  0 := not use")
        for i,path in enumerate(label_path):
            print("  {} := {}".format(i+1, (path.split(data_name+SEP)[-1]).split(SEP+"predicted_labels")[0]))
        print("\nlabel_choice : ", end="")
        label_choice = int(input())

    ## 予測のラベルを使うとき
    if label_choice != 0:
        ## 機械学習を行い予測したときのビット相関の数とモーメントの次数を取得し、そのパラメータで教師データを再生成する
        k, kp_list = read_parameter.get_birCorr_moment(label_path[label_choice-1].split("predicted_labels")[0])
        train_data, train_info, valid_data, valid_info, test_dataset, test_infoset = data_loader.load_data("PCA", data_name, dt_index, k, kp_list)
        ## 予測済みのラベルの読み込み
        train_pred_label, valid_pred_label, test_pred_labelset = data_loader.load_pred_labels(label_path[label_choice-1])
    ## ラベルの情報は無しでPCAするとき
    else:
        ## ビット相関の数とモーメントの次数を入力し、データセットを読み込む
        train_data, train_info, valid_data, valid_info, test_dataset, test_infoset = data_loader.load_data("PCA", data_name, dt_index)
        ## draw_train_base関数を呼ぶときに引数で変数train_pred_labelを渡したり、ヒストグラムの
        ## 本数を決めるときにtrain_pred_labelの情報を使ったりするので、0で作っておく
        train_pred_label = np.zeros(train_data.shape[0], dtype=int)

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
    
    ## 第１,第２主成分を残し、射影後の分散が最大化するように次元を削減する
    pca = PCA(n_components=2)

    ## 訓練の開始
    print("\r" + "fitting...", end="")
    start = time.perf_counter()
    pca.fit(train_data)
    finish = time.perf_counter()

    ## 訓練にかかった時間を出力
    print("\r" + "fit time : {}[s]".format(finish-start))

    ## それぞれのデータに対してPCAを行う
    train_reduc = pca.transform(train_data)
    valid_reduc = pca.transform(valid_data)
    testset_reduc = [pca.transform(test_data) for test_data in test_dataset]

    """ 第1主成分(横軸)と第2主成分(縦軸)で訓練データの散布図を描く
        上の図は散布図で、下の図は第1主成分の密度をヒストグラムで表現する
    """
    fig = plt.figure() #図の作成
    ax1 = fig.add_subplot(2,1,1) #上側の主成分のプロット
    ax2 = fig.add_subplot(2,1,2) #下側の第1主成分の密度のヒストグラム
    hist_bins = train_pred_label.shape[0] // 20 #ヒストグラムの棒の数
    ## 結合して作成していた訓練データを、1つずつ順番にプロットしていく
    p = 0 #リストの位置を示すポインタ
    for i, info in enumerate(train_info):
        if info[0] == "haar":
            color = "blue" #Haar-trainを青
        elif info[0] == "clif":
            color = "red" #クリフォード系-trainを赤
        else:
            color = "black" #他はひとまず黒にしておく
        
        if label_choice == 0:
            ## 予測ラベルなしのときは色分けあり、記号○でプロット
            ax1.scatter(train_reduc[p:p+info[1], 0], train_reduc[p:p+info[1], 1], label=info[0]+"-train", marker='o', alpha=0.5, s=15, c=color)
        
        else:
            ## 結合して作成した教師データのうちの、1つのデータのラベルを抜き出す
            partial_label = train_pred_label[p:p+info[1]]
            ## 特徴量ベクトルのうち、0(Haar)または1(clif)と予測されたベクトルをそれぞれ保存するリスト
            pred_0, pred_1 = [], []
            ## リスト内包表記でそれぞれのリストに分ける
            [pred_0.append(val) if partial_label[j] == 0 else pred_1.append(val) for j,val in enumerate(train_reduc[p:p+info[1]])]
            ## 散布図として描き込む
            if len(pred_0) != 0:
                pred_0 = np.array(pred_0)
                ax1.scatter(pred_0[:,0], pred_0[:,1], label=info[0]+"-train:pred0", marker='o', alpha=0.5, s=15, c=color) #0と予測されたデータは○でプロット
            if len(pred_1) != 0:
                pred_1 = np.array(pred_1)
                ax1.scatter(pred_1[:,0], pred_1[:,1], label=info[0]+"-train:pred1", marker='*', alpha=0.5, s=15, c=color) #1と予測されたデータは☆でプロット
        ## ヒストグラムを描く
        ax2.hist(train_reduc[p:p+info[1], 0], label=info[0], bins=hist_bins, density=True, alpha=0.8, color=color)
        p += info[1] #ポインタを進める
    ## 2つのグラフの各軸の名前やタイトル、グリッド線や凡例を表示させる
    ax1.set_xlabel("PC1")
    ax1.set_ylabel("PC2")
    ax1.set_title("A plot of training data")    
    ax1.legend()
    ax1.grid()
    ax2.set_xlabel("PC1")
    ax2.set_ylabel("Density")
    ax2.set_title("Histgram of PC1 (sum=1)")
    ax2.legend()
    ax2.grid()
    fig.set_tight_layout(True)
    plt.savefig(dir_path+"train.png")
    if not repeat:
        plt.show()
    
    """ 訓練データと検証データをプロットする。
    """
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    draw_train_base(label_choice, ax, train_info, train_reduc, train_pred_label)
    ## 結合して作成していた検証データを、1つずつ順番にプロットしていく
    p = 0 #リストの位置を示すポインタ
    for i, info in enumerate(valid_info):
        if info[0] == "haar":
            color = "yellow" #Haar-validを黄
        elif info[0] == "clif":
            color = "green" #クリフォード系-validを緑
        else:
            color = "black" #他はひとまず黒にしておく
        
        if label_choice == 0:
            ## 予測ラベルなしのときは色分けあり、記号○でプロット
            ax.scatter(valid_reduc[p:p+info[1], 0], valid_reduc[p:p+info[1], 1], label=info[0]+"-valid", marker='o', alpha=0.5, s=15, c=color)
        
        else:
            ## 結合して作成した教師データのうちの、1つのデータのラベルを抜き出す
            partial_label = valid_pred_label[p:p+info[1]]
            ## 特徴量ベクトルのうち、0(Haar)または1(clif)と予測されたベクトルをそれぞれ保存するリスト
            pred_0, pred_1 = [], []
            ## リスト内包表記でそれぞれのリストに分ける
            [pred_0.append(val) if partial_label[j] == 0 else pred_1.append(val) for j,val in enumerate(valid_reduc[p:p+info[1]])]
            ## 散布図として描き込む
            if len(pred_0) != 0:
                pred_0 = np.array(pred_0)
                ax.scatter(pred_0[:,0], pred_0[:,1], label=info[0]+"-valid:pred0", marker='o', alpha=0.5, s=15, c=color) #0と予測されたデータは○でプロット
            if len(pred_1) != 0:
                pred_1 = np.array(pred_1)
                ax.scatter(pred_1[:,0], pred_1[:,1], label=info[0]+"-valid:pred1", marker='*', alpha=0.5, s=15, c=color) #1と予測されたデータは☆でプロット
        p += info[1] #ポインタを進める
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("A plot of training and validiation data")    
    ax.legend()
    ax.grid()
    fig.set_tight_layout(True)
    plt.savefig(dir_path+"valid.png")
    if not repeat:
        plt.show()

    """ 訓練データとテストデータでプロットする。
    """
    for num, each_test_info in enumerate(test_infoset):
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        draw_train_base(label_choice, ax, train_info, train_reduc, train_pred_label)
        ## 結合して作成していたテストデータを、1つずつ順番にプロットしていく
        p = 0 #リストの位置を示すポインタ
        for i, info in enumerate(each_test_info):
            ## グラフの色を決める。Haar以外は緑にしているので、もしテストデータセット内に含まれるなら変更する。
            if info[0] == "haar":
                color = "yellow" #Haar-testを黄
            elif info[0] == "clif":
                color = "green" #クリフォード系-testを緑
            elif info[0] == "lrc":
                color = "green" #lrc-testも緑
            elif info[0] == "nakata":
                color = "green" #nakata-testも緑

            if label_choice == 0:
                ## 予測ラベルなしのときは色分けあり、記号○でプロット
                ax.scatter(testset_reduc[i][p:p+info[1], 0], testset_reduc[i][p:p+info[1], 1], label=info[0]+"-test{}".format(num+1), marker='o', alpha=0.5, s=15, c=color)
            
            else:
                ## テストデータやラベルをリストにしていたものから対象のものを選択し、範囲を選んでデータを抜き出す
                partial_data = testset_reduc[i][p:p+info[1]]
                partial_label = test_pred_labelset[i][p:p+info[1]]
                ## 特徴量ベクトルのうち、0(Haar)または1(clif)と予測されたベクトルをそれぞれ保存するリスト
                pred_0, pred_1 = [], []
                ## リスト内包表記でそれぞれのリストに分ける
                [pred_0.append(val) if partial_label[j] == 0 else pred_1.append(val) for j, val in enumerate(partial_data)]
                ## 散布図として描き込む
                if len(pred_0) != 0:
                    pred_0 = np.array(pred_0)
                    ax.scatter(pred_0[:,0], pred_0[:,1], label=info[0]+"-test{}:pred0".format(num+1), marker='o', alpha=0.5, s=15, c=color) #0と予測されたデータは○でプロット
                if len(pred_1) != 0:
                    pred_1 = np.array(pred_1)
                    ax.scatter(pred_1[:,0], pred_1[:,1], label=info[0]+"-test{}:pred1".format(num+1), marker='*', alpha=0.5, s=15, c=color) #1と予測されたデータは☆でプロット
                p += info[1] #ポインタを進める

        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_title("A plot of training and test{} data".format(num+1))
        ax.legend()
        ax.grid()
        fig.set_tight_layout(True)
        plt.savefig(dir_path+"test{}.png".format(num+1))
        if not repeat:
            plt.show()

    ## 特徴量ベクトルの重要度を取得
    importances = pca.components_
    ## 重要度が高い順(降順)に並び替え、そのときのindexのリストを取得
    indices = np.argsort(-importances)
    
    ## インデックスと係数(重要度)を送りcsvファイルに保存する
    ## インデックスの順で保存されるのでソートして置いたほうが見やすい
    Searcher.search_and_save_all(indices[0], importances[0], filename="components_PC1")
    Searcher.search_and_save_all(indices[1], importances[1], filename="components_PC2")


if __name__ == "__main__":
    sklearn_pca()