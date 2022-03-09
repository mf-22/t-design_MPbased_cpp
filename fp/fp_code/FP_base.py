import numpy as np
import os

class FPBase():
    """ FramePotentialを計算するにあたっての一番おおもとの抽象(基底)クラス。
        qubit数や回路の深さ等のパラメータのセットや保存などの基本的な動作を定義する。
    """
    def __init__(self, Nq=4, depth=5, t=2, epsilon=0.001) -> None:
        """ コンストラクタ
        """
        self.Nq = Nq        #qubit数
        self.depth = depth  #量子回路の深さ(深さがない回路のときはNoneとか)
        self.t = t          #t-designの次数  
        self.eps = epsilon  #収束判定の際に用いる変数
    
    def set_parameter(self, key, val) -> None:
        if key == "Nq":
            self.Nq = val
        elif key == "depth":
            self.depth = val
        elif key == "t":
            self.t = val
        elif key == "epsilon":
            self.eps = val
        else:
            print('Error: key "{}" is invalid, defalut parameter is set.')

    def calculate(self) -> None:
        raise NotImplementedError("This is abstract method")

    def sample_U(self) -> np.ndarray:
        raise NotImplementedError("This is abstract method")

    def check_mean_convergence(self, fp_shot) -> bool:
        raise NotImplementedError("This is abstract method")
    
    def check_std_convergence(self, fp_shot) -> bool:
        raise NotImplementedError("This is abstract method")

    def save_result(self, foldername="", log=True, paras={}) -> None:
        """ 結果の保存を行うメソッド。
            最終結果とパラメータがtxtファイルで保存され、計算過程は
            引数のフラグ log に応じて保存するか決定される。
            Arguments:
                foldername(str) := 計算結果が保存されるフォルダ名。空のときは現在時刻
                log(bool)       := Trueのときは過去の計算過程全てをcsvで保存し、
                                   Falseのときは保存されない
                paras(dict)     := 追加で保存したいパラメータとかを保持する辞書型
                                   保存する際は{key} : {value}の形で書き込まれる
        """
        if len(foldername) == 0:
            import datetime
            dt_now = datetime.datetime.now()
            foldername = dt_now.strftime("%Y%m%d%H%M%S")
            
        if not os.path.exists("./result/" + foldername):
            os.makedirs("./result/" + foldername)
        
        with open("./result/" + foldername + "/parameters.txt", mode="w") as f:
            f.write("#qubit  : {}\n".format(self.Nq))
            f.write("depth   : {}\n".format(self.depth))
            f.write("t       : {}\n".format(self.t))
            f.write("epsilon : {}\n".format(self.eps))
            for key in paras.keys():
                f.write("{} : {}\n".format(key, paras[key]))
        
        with open("./result/" + foldername + "/result.txt", mode="w") as f:
            #f.write("{} \pm {}".format(self.result_arr[-1][1], self.result_arr[-1][2]))
            f.write("{} \pm {}".format(self.result_arr[-1][1], np.std(self.result_arr[:][1])))
        
        if log:
            np.savetxt("./result/" + foldername + "/log.csv", self.result_arr, delimiter=",")
