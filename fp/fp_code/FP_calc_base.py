from .FP_base import FPBase
import numpy as np
import os

class FP_main_base(FPBase):
    """ FramePotentialを計算するにあたっての抽象(基底)クラス。
        FP_base.pyのクラスFPBaseを継承して定義する。
        計算したい回路がLRCとかRDCなど色々なパターンがあると思うので、
        この抽象クラスを継承してユニタリをサンプルするところだけ新たに
        作れば良いようにする。
    """
    def __init__(self, Nq=4, depth=5, t=2, epsilon=0.001, patience=5, monitor="mean") -> None:
        """ コンストラクタ
        """
        super(FP_main_base, self).__init__(Nq=Nq, depth=depth, t=t, epsilon=epsilon)
        self.monitor = monitor
        if monitor == "mean":
            self.patience = patience #ループを何回分監視するかの変数
            self.para_dict = {"patience" : patience} #パラメータをファイル書き込みで保存するための辞書
            self.result_arr = np.empty(2) #計算結果を保持する2次元のndarray
            ## 今は1次元だが計算していくにつれて(1,2)->(2,2)->...->(n,2)のように
            ## 1次元目を増やしていく。2次元目の3要素の説明は以下の通り：
            ##     [x,1] := Tr|UV^{\daggar}|^{2t}を計算したときの結果のリスト
            ##     [x,2] := #FramePotentailの平均値の値のリスト
        elif monitor == "std":
            self.result_arr = np.empty(3) #計算結果を保持する2次元のndarray
            ## 今は1次元だが計算していくにつれて(1,3)->(2,3)->...->(n,3)のように
            ## 1次元目を増やしていく。2次元目の3要素の説明は以下の通り：
            ##     [x,1] := Tr|UV^{\daggar}|^{2t}を計算したときの結果のリスト
            ##     [x,2] := #FramePotentailの平均値の値のリスト
            ##     [x,3] := #FramePotentailの標準偏差の値のリスト
            self.squared = 0.0  #標準偏差を計算する際に用いる、単発の計算結果の2乗の平均の値
            self.para_dict = {} #パラメータをファイル書き込みで保存するための辞書。stdの場合は空
        else:
            print('WARNING: monitor target "{}" is invalid. Default monitor "mean" is set.')
            self.monitor = "mean"
            self.patience = patience
            self.para_dict = {"patience" : patience}
            self.result_arr = np.empty(2)
    
    def set_parameter(self, key, val) -> None:
        if key == "monitor":
            if val == "mean":
                self.monitor == val
                self.result_arr = np.empty(2)
                self.para_dict = {"patience" : val}
            elif val == "std":
                self.monitor == val
                self.result_arr = np.empty(3)
                self.para_dict = {}
                self.squared = 0.0
            else:
                print('WARNING: The parameter "monitor" is specified, but value "{}" is invalid.'.format(val))
                print('         Valid paramters are "mean" or "std". Now monitor is "{}"'.format(self.monitor))
        
        elif key == "patience":
            self.patience = val

        else:
            super(FP_main_base, self).set_parameter(key, val)
        
    def calculate(self) -> None:
        """ 実際に収束まで計算するメソッド。
            「ランダムにユニタリ行列をサンプル=>計算」を繰り返し、標準偏差の値がパラメータとして
            指定したepsilonの値以下になるまで計算を続ける。
        """
        if self.monitor == "mean":
            ## 最初の1回と、patience回の間は次のwhileループから外して先に計算する
            ## 中に入れるとif文の条件判定が増えてしまい遅くなるので
            count = self.patience + 1
            for i in range(count):
                U = self.sample_U()
                Vdag = np.conjugate(self.sample_U().T)
                potential = np.abs(np.trace(U@Vdag))**(2*self.t)
                if i == 0:
                    ## 1回目の平均値は単発の結果そのまま
                    self.result_arr = np.array([potential, potential]).reshape(1,2)
                else:
                    temp = np.append(self.result_arr[:,0], potential)
                    self.result_arr = np.vstack( (self.result_arr, [potential, np.mean(temp)]) )

            while True:
                U = self.sample_U() #Uをランダムにサンプル
                Vdag = np.conjugate(self.sample_U().T) #Vをランダムにサンプルし複素転置共役
                potential = np.abs(np.trace(U@Vdag))**(2*self.t) #FramePotentialの値を計算
                count += 1
                #print("\r  {} times calculated...".format(count), end="") #計算回数の出力
                if self.check_mean_convergence(potential): #収束判定
                    print("")
                    break
                ## 逐次結果を出力(しないほうが早い)
                print("\r  [{:.3f}, {:.3f}] ({} times calculated)".format(self.result_arr[count-1][0], self.result_arr[count-1][1], count) + "   ", end="")
                if count % 10000 == 0: #計算回数が10000の倍数のとき
                    ## 途中の計算結果を保存しておく
                    self.save_result(foldername="LRC_Nq{}_depth{}_t{}".format(self.Nq, self.depth, self.t))
                """
                if count == 10000: #強制終了
                    print("")
                    print("avg:{}, std:{}".format(np.mean(self.result_arr[:,0]), np.std(self.result_arr[:,0])))
                    break
                """

        elif self.monitor == "std":
            ## 最初の2回だけ次のwhileループから外して計算する
            ## 中に入れるとif文の条件判定が増えてしまうので
            for i in range(2):
                U = self.sample_U()
                Vdag = np.conjugate(self.sample_U().T)
                potential = np.abs(np.trace(U@Vdag))**(2*self.t)
                if i == 0:
                    ## 1回目の平均値は単発の結果そのまま
                    ## 1回目の標準偏差はとりあえず0.0を(これは後で使わない)
                    self.result_arr = np.array([potential, potential, 0.0]).reshape(1,3)
                elif i == 1:
                    ## 2回目の平均値は普通に計算
                    ## 2回目の標準偏差も普通に
                    temp = [self.result_arr[0][0], potential]
                    self.result_arr = np.vstack( (self.result_arr, 
                                                [potential, np.mean(temp), np.std(temp)])
                                            )
                    self.squared = np.mean([j**2 for j in temp]) #要素の2乗の平均の値を計算

            ## 3回目から収束するまでは以下のwhileループで計算
            count = 2 #計算回数を示すカウンタ、既に2回計算しているので
            while True:
                U = self.sample_U() #Uをランダムにサンプル
                Vdag = np.conjugate(self.sample_U().T) #Vをランダムにサンプルし複素転置共役
                potential = np.abs(np.trace(U@Vdag))**(2*self.t) #FramePotentialの値を計算
                count += 1
                #print("\r  {} times calculated...".format(count), end="") #計算回数の出力
                if self.check_std_convergence(potential): #収束判定
                    print("")
                    break
                ## 逐次結果を出力(しないほうが早い)
                print("\r  [{:.3f}, {:.3f}, {:.3f}] ({} times calculated)".format(self.result_arr[count-1][0], self.result_arr[count-1][1], self.result_arr[count-1][2], count) + "   ", end="")
                if count % 10000 == 0: #計算回数が10000の倍数のとき
                    ## 途中の計算結果を保存しておく
                    self.save_result(foldername="LRC_Nq{}_depth{}_t{}".format(self.Nq, self.depth, self.t))
                """
                if count == 10000: #強制終了
                    print("")
                    print("avg:{}, std:{}".format(np.mean(self.result_arr[:,0]), np.std(self.result_arr[:,0])))
                    break
                """

    def check_mean_convergence(self, fp_shot) -> bool:
        """ FramePotentailの値が収束しているか判定をするメソッド。
            平均の値の計算を行い、最後に条件に応じてフラグを返す。
            Argument:
                fp_shot(float) := FramePotentialの単発の計算結果
            Return:
                flag(bool) := Trueなら収束した、Falseなら収束していない
        """
        ## 平均の値を計算
        num_calculated = self.result_arr.shape[0]
        new_avg = (self.result_arr[-1][1] * num_calculated + fp_shot) / (num_calculated+1)
        ## 単発の計算結果と平均の結果を保存
        self.result_arr = np.vstack( (self.result_arr, [fp_shot, new_avg]) )

        ## 最新の計算結果とpatience回までの全ての計算結果を比較し全部epsilon以下ならTrueを返す
        for i in range(self.patience):
            if np.abs(self.result_arr[-1][1] - self.result_arr[-1-(i+1)][1]) > self.eps:
                return False
        return True

    def check_std_convergence(self, fp_shot) -> bool:
        """ FramePotentailの値が収束しているか判定をするメソッド。
            平均の値や標準偏差の値の計算を行い、最後に条件に応じてフラグを返す。
            Argument:
                fp_shot(float) := FramePotentialの単発の計算結果
            Return:
                flag(bool) := Trueなら収束した、Falseなら収束していない
        """
        ## 平均の値を計算
        num_calculated = self.result_arr.shape[0]
        new_avg = (self.result_arr[-1][1] * num_calculated + fp_shot) / (num_calculated+1)
        ## 標準偏差の値を計算
        self.squared = (self.squared * num_calculated + fp_shot**2) / (num_calculated+1)
        new_std = np.sqrt(self.squared - new_avg**2)
        ## 単発の計算結果と平均・標準偏差の結果を保存
        self.result_arr = np.vstack( (self.result_arr, [fp_shot, new_avg, new_std]) )

        ## 標準偏差の値に応じて収束したかどうかのフラグを返す
        if new_std < self.eps:
            return True
        else:
            return False

    def save_result(self, foldername="", log=True) -> None:
        """ パラメータのpatienceを書き込むために、FPBaseクラスにあるsave_resultの
            引数を追加して呼び出すだけのメソッド
        """
        super(FP_main_base, self).save_result(foldername=foldername, log=log, paras=self.para_dict)