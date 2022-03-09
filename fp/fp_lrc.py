from fp_code import FP_calc_base
import numpy as np
from scipy.stats import unitary_group


class FP_LRC(FP_calc_base.FP_main_base):
    def __init__(self, Nq=4, depth=5, t=2, epsilon=0.001, patience=5, monitor="mean") -> None:
        super(FP_LRC, self).__init__(Nq=Nq, depth=depth, t=t, epsilon=epsilon, patience=patience, monitor=monitor)
        self.circ_type = "LRC"
        ## LRCの各層における2-qubitゲートの数を計算
        self.num_gates_depthOdd = Nq // 2
        if self.Nq % 2 == 1:
            self.num_gates_depthEven = Nq // 2
        else:
            self.num_gates_depthEven = (Nq // 2) - 1

    def sample_U(self) -> np.ndarray:
        ## 2^Nq * 2^Nq の大きさのidentity
        ## これに各深さのlocal 2-qubit haar gateを結合したものの行列積を計算することで
        ## 最終的に欲しいユニタリを作る
        big_unitary = np.identity(2**self.Nq) 

        ## 回路の深さ分繰り返す
        for i in range(self.depth, 0, -1):
            ## 深さが奇数のとき
            if i % 2 == 1:
                ## 一番上のゲートは2-qubit haar random gate
                small_unitary = unitary_group.rvs(4)
                ## small_unitaryに必要な個数だけテンソル積をとり、local 2-qubit haarを結合する
                for _ in range(self.num_gates_depthOdd - 1):
                    small_unitary = np.kron(small_unitary, unitary_group.rvs(4))
                ## 量子ビット数、回路の深さともに奇数のときはIdentityを最後(一番下のqubit)に結合する
                if self.Nq % 2 == 1:
                    small_unitary = np.kron(small_unitary, np.identity(2))
            ## 深さが偶数のとき
            else:
                ## 回路の深さが偶数のときは最初は必ずIdentity
                small_unitary = np.identity(2)
                ## small_unitaryに必要な個数だけテンソル積をとり、local 2-qubit haarを結合する
                for _ in range(self.num_gates_depthEven):
                    small_unitary = np.kron(small_unitary, unitary_group.rvs(4))
                ## 量子ビット数が偶数、回路の深さが偶数のときはIdentityを最後につける
                if self.Nq % 2 == 0:
                    small_unitary = np.kron(small_unitary, np.identity(2))
            ## 行列をかけて各深さごとに作成した(num_qubits)サイズのユニタリをマージしていく
            big_unitary = big_unitary @ small_unitary
        
        return big_unitary
    
    def calculate(self) -> None:
        print("Now => circ:{}, Nq:{}, depth:{}, t:{}".format(self.circ_type, self.Nq, self.depth, self.t))
        super(FP_LRC, self).calculate()
        
def main():
    Nq = 5
    depth = 6
    t = 2
    epsilon = 0.001
    monitor = "mean"
    patience = 5
    FP_LRC_calclator = FP_LRC(Nq=Nq, depth=depth, t=t, epsilon=epsilon, monitor=monitor, patience=patience)
    FP_LRC_calclator.calculate()
    FP_LRC_calclator.save_result(foldername="LRC_Nq{}_depth{}_t{}".format(Nq, depth, t))

    """
    FP_LRC_calclator = FP_LRC()
    Nq_list = [i for i in range(3, 8)]
    depth_list = [i for i in range(4, 21)]
    t_list = [i for i in range(2, 6)]
    epsilon = 0.1
    FP_LRC_calclator.set_parameter("epsilon", 0.1)
    for t in t_list:
        FP_LRC_calclator.set_parameter("t", t)
        for Nq in Nq_list:
            FP_LRC_calclator.set_parameter("Nq", Nq)
            for depth in depth_list:
                FP_LRC_calclator.set_parameter("depth", depth)
                FP_LRC_calclator.calculate()
                FP_LRC_calclator.save_result(foldername="LRC_Nq{}_depth{}_t{}".format(Nq, depth, t))
    """

if __name__ == "__main__":
    main()