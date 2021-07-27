from group.clifford_group_circuit import CliffordCircuitGroup
from group.clifford_group import CliffordGroup
from qulacs import QuantumState
from qulacs.gate import CNOT, DenseMatrix
import itertools
import time
import numpy as np
import datetime
import sys
import multiprocessing

def input_parameters():
    """コマンドラインからパラメータの入力を受け取り辞書型に保存して返す関数
    """
    paras_dict = {}

    args = sys.argv
    args.remove("random_clif.py")
    if "seq" in args:
        args.remove("seq")
    if "sequential" in args:
        args.remove("sequential")
    
    ## auto_create.pyから呼ばれて実行されるときはコマンドライン引数からパラメータを指定する 
    if len(args) == 7:
        for arg in args:
            key, val = arg.split("=")
            paras_dict[key] = int(val)

    else:
        ## input number of data you want to create(|S|)
        print('input number of data (|S|) :', end=(' '))
        paras_dict["S"] = int(input())
        print('Nu :', end=(' '))
        paras_dict["Nu"] = int(input())
        print('Ns :', end=(' '))
        paras_dict["Ns"] = int(input())
        print('Nq :', end=(' '))
        paras_dict["Nq"] = int(input())
        print('Local random clifford?(1:=Yes, 0:=No) :', end=(' '))
        local_in = int(input())
        if local_in == 1:
            paras_dict["local"] = 1
            print('input circuit depth :', end=(' '))
            paras_dict["depth"] = int(input())
            print('CNOT & 1q Clifford?(1:=Yes, 0:=No) :', end=(' '))
            paras_dict["CNOT_1qC"] = int(input())
        else:
            paras_dict["local"] = 0

    return paras_dict

def gen_comb_list(num_qubits):
    """数の組み合わせ(nCr)のパターンを返す関数
       戻り値は2次元のリストで1次元目はパターンの数、2次元目は実際のパターン
       例) 3qubit なら
           [[0], [1], [2],       <= 1点ビット相関
           [0,1], [0,2], [1,2],  <= 2点ビット相関
           [0,1,2]]              <= 3点ビット相関
    """
    bit_list = [i for i in range(num_qubits)]
    result = []

    for i in range(1, num_qubits+1):
        for j in itertools.combinations(bit_list, i):
            result.append(list(j))
    
    return result

def gen_random_index(order):
    """ (0, order]の範囲でランダムに整数を生成し返す関数
        似たことを行うものにnp.random.randintがあるが、上限が2^32であり
        クリフォード群の位数よりも小さいため自作のものを用いる
    """
    d = str(order)
    dig_num = len(d)
    sig_dig = int(d[0])

    while True:
        clif_index = 0
        a = np.random.randint(sig_dig+1, size=1)
        f = np.append(a, np.random.randint(10, size=dig_num-1))
        clif_index = int("".join(map(str, f)))
        if clif_index < order:
            return clif_index

def sim_local_random_clifford(S, Nu, Ns, Nq, depth, RU_index_list, comb_list):
    ## 最終的に作成する教師データの配列
    teacher_data = np.empty((S, len(comb_list)*20), dtype=np.float32)
    ## 測定確率(ビット相関の期待値)の計算結果を保存する配列を用意
    MP_list = np.empty((Nu, len(comb_list)), dtype=np.float32)
    ## 量子状態の準備
    state = QuantumState(Nq)
    ## 2qubitのクリフォード群を宣言、位数は11520
    ## 4×4の行列ではなくゲート列が返ってくる
    #ccg = CliffordCircuitGroup(2)
    ##order = ccg.order <= 11520
    cg = CliffordGroup(2)

    ## 2進数のリストを作成
    binary_num_list = np.empty((2**Nq, Nq), dtype=np.int8)
    for i in range(2**Nq):
        binary_num = np.array(list(bin(i)[2:].zfill(Nq))).astype(np.int8)
        binary_num[binary_num == 0] = -1 #0を-1に変換
        binary_num_list[i] = binary_num

    for i in range(S):
        ## 教師データ1つにつき初期状態を固定する、そのための乱数のシード
        haar_seed = np.random.randint(2147483648) #2^31
        for j in range(Nu):
            ## 初期状態をHaar random stateにする
            state.set_Haar_random_state(haar_seed)
            ## 2qubitのランダムクリフォードを互い違いにかけていく
            for k in range(1, depth+1):
                for qubit_index in RU_index_list[k%2]:
                    #circuit = ccg.get_element(np.random.randint(11520))
                    #ccg.simulate_circuit_specific_qubit(2, circuit, state, qubit_index)
                    DenseMatrix([qubit_index, qubit_index+1], cg.get_element(np.random.randint(11520))).update_quantum_state(state)
            ## 測定を行う
            result_bin = np.array([binary_num_list[i] for i in state.sampling(Ns)])
            ## ビット相関を計算し、測定確率を計算
            for k, combination in enumerate(comb_list):
                bit_corr = result_bin[:, combination[0]].copy()
                for index in range(1, len(combination)):
                    bit_corr *= result_bin[:, combination[index]]
                MP_list[j][k] = np.mean(bit_corr)
        ## モーメントの計算
        teacher_data[i] = np.array([np.power(MP_list, m).mean(axis=0) for m in range(1, 21)]).flatten()
        print('\r{} / {} finished...'.format(i+1, S), end=(''))
    print('')
    
    return teacher_data

def sim_local_random_clif_CNOTand1qubitClif(S, Nu, Ns, Nq, depth, RU_index_list, comb_list):
    ## 最終的に作成する教師データの配列
    teacher_data = np.empty((S, len(comb_list)*20), dtype=np.float32)
    ## 測定確率(ビット相関の期待値)の計算結果を保存する配列を用意
    MP_list = np.empty((Nu, len(comb_list)), dtype=np.float32)
    ## 量子状態の準備
    state = QuantumState(Nq)
    ## 2qubitのクリフォード群を宣言、位数は11520
    ## 4×4の行列ではなくゲート列が返ってくる
    cg = CliffordGroup(1)
    ##order = ccg.order <= 24

    ## 2進数のリストを作成
    binary_num_list = np.empty((2**Nq, Nq), dtype=np.int8)
    for i in range(2**Nq):
        binary_num = np.array(list(bin(i)[2:].zfill(Nq))).astype(np.int8)
        binary_num[binary_num == 0] = -1 #0を-1に変換
        binary_num_list[i] = binary_num

    for i in range(S):
        ## 教師データ1つにつき初期状態を固定する、そのための乱数のシード
        haar_seed = np.random.randint(2147483648) #2^31
        for j in range(Nu):
            ## 初期状態をHaar random stateにする
            state.set_Haar_random_state(haar_seed)
            ## 2qubitのランダムクリフォードを互い違いにかけていく
            for k in range(1, depth+1):
                for qubit_index in RU_index_list[k%2]:
                    CNOT(qubit_index, qubit_index+1).update_quantum_state(state)
                    clif_matrix = cg.sampling(2)
                    DenseMatrix(qubit_index, clif_matrix[0]).update_quantum_state(state)
                    DenseMatrix(qubit_index+1, clif_matrix[1]).update_quantum_state(state)
            ## 測定を行う
            result_bin = np.array([binary_num_list[i] for i in state.sampling(Ns)])
            ## ビット相関を計算し、測定確率を計算
            for k, combination in enumerate(comb_list):
                bit_corr = result_bin[:, combination[0]].copy()
                for index in range(1, len(combination)):
                    bit_corr *= result_bin[:, combination[index]]
                MP_list[j][k] = np.mean(bit_corr)
        ## モーメントの計算
        teacher_data[i] = np.array([np.power(MP_list, m).mean(axis=0) for m in range(1, 21)]).flatten()
        print('\r{} / {} finished...'.format(i+1, S), end=(''))
    print('')

    return teacher_data

def sim_random_clifford(S, Nu, Ns, Nq, comb_list):
    ## 最終的に作成する教師データの配列
    teacher_data = np.empty((S, len(comb_list)*20), dtype=np.float32)
    ## 測定確率(ビット相関の期待値)の計算結果を保存する配列を用意
    MP_list = np.empty((Nu, len(comb_list)), dtype=np.float32)
    ## 量子状態の準備
    state = QuantumState(Nq)
    ## 2qubitのクリフォード群を宣言
    ## (2^n)×(2^n)の行列ではなくゲート列が返ってくる
    ccg = CliffordCircuitGroup(Nq)
    ## クリフォード群の要素数を取得
    order = ccg.order

    ## 2進数のリストを作成
    binary_num_list = np.empty((2**Nq, Nq), dtype=np.int8)
    for i in range(2**Nq):
        binary_num = np.array(list(bin(i)[2:].zfill(Nq))).astype(np.int8)
        binary_num[binary_num == 0] = -1 #0を-1に変換
        binary_num_list[i] = binary_num

    for i in range(S):
        ## 教師データ1つにつき初期状態を固定する、そのための乱数のシード
        haar_seed = np.random.randint(2147483648) #2^31
        for j in range(Nu):
            ## 初期状態をHaar random stateにする
            state.set_Haar_random_state(haar_seed)
            ## ランダムクリフォードの実行
            circuit = ccg.get_element(gen_random_index(order))
            ccg.simulate_circuit(Nq, circuit, state)
            ## 測定を行う
            result_bin = np.array([binary_num_list[i] for i in state.sampling(Ns)])
            ## ビット相関を計算し、測定確率を計算
            for k, combination in enumerate(comb_list):
                bit_corr = result_bin[:, combination[0]].copy()
                for index in range(1, len(combination)):
                    bit_corr *= result_bin[:, combination[index]]
                MP_list[j][k] = np.mean(bit_corr)
        ## モーメントの計算
        teacher_data[i] = np.array([np.power(MP_list, m).mean(axis=0) for m in range(1, 21)]).flatten()
        print('\r{} / {} finished...'.format(i+1, S), end=(''))
    print('')
    
    return teacher_data


def main(parallel = True):
    ## パラメータ読み込み
    paras_dict = input_parameters()
    ## ビット相関計算時のqubitのインデックスのリスト
    comb_list = gen_comb_list(paras_dict["Nq"])

    ## 処理開始時の時間の取得
    start = time.perf_counter()

    ## Local Random Cliffordの計算
    if paras_dict["local"] == 1:
        RU_index_list = []
        ## 深さが"奇数"のときのlocal random cliffordがかかるqubitのインデックス
        RU_index_list.append([i for i in range(1, paras_dict["Nq"]-1, 2)])
        ## 深さが"偶数"のときのlocal random cliffordがかかるqubitのインデックス
        RU_index_list.append([i for i in range(0, paras_dict["Nq"]-1, 2)])

        ## 並列実行
        if parallel:
            core_num = multiprocessing.cpu_count()
            multi_S = int(paras_dict["S"] / core_num)
            print('S = {}, core_num={}, multi_S={}, create_num={}'
                  .format(paras_dict["S"], core_num, multi_S, core_num*multi_S))
            ## S個をスレッド数できれいに割り切れたとき
            if multi_S * core_num == paras_dict["S"]:
                args = [(multi_S, paras_dict["Nu"], paras_dict["Ns"],
                         paras_dict["Nq"], paras_dict["depth"],
                         RU_index_list, comb_list) for i in range(core_num)]
            ## 割り切れなかったとき
            else:
                args = [(multi_S, paras_dict["Nu"], paras_dict["Ns"],
                         paras_dict["Nq"], paras_dict["depth"],
                         RU_index_list, comb_list) for i in range(core_num-1)]
                ## 最後の1つのスレッドに余り分をやらせる
                remain_S = paras_dict["S"] - multi_S * core_num
                args.append((multi_S+remain_S, paras_dict["Nu"], paras_dict["Ns"],
                             paras_dict["Nq"], paras_dict["depth"],
                             RU_index_list, comb_list))
            ## 並列実行の開始
            if paras_dict["CNOT_1qC"] == 0:
                p = multiprocessing.Pool(core_num)
                returns = p.starmap(sim_local_random_clifford, args)
                p.close()
            elif paras_dict["CNOT_1qC"] == 1:
                p = multiprocessing.Pool(core_num)
                returns = p.starmap(sim_local_random_clif_CNOTand1qubitClif, args)
                p.close()
            ##　それぞれのスレッドの実行結果をひとまとめにする
            result = np.concatenate(returns, axis=0)

        ## 逐次実行
        else:
            ## 量子回路シミュレーションと測定確率の計算
            if paras_dict["CNOT_1qC"] == 0:
                result = sim_local_random_clifford(paras_dict["S"],  paras_dict["Nu"],
                                                paras_dict["Ns"], paras_dict["Nq"],
                                                paras_dict["depth"], RU_index_list, comb_list)
            elif paras_dict["CNOT_1qC"] == 1:
                result = sim_local_random_clif_CNOTand1qubitClif(paras_dict["S"],  paras_dict["Nu"],
                                                paras_dict["Ns"], paras_dict["Nq"],
                                                paras_dict["depth"], RU_index_list, comb_list)

    ## Random Cliffordの計算
    else:
        if parallel:
            core_num = multiprocessing.cpu_count()
            multi_S = int(paras_dict["S"] / core_num)
            print('S = {}, core_num={}, multi_S={}, create_num={}'
                  .format(paras_dict["S"], core_num, multi_S, core_num*multi_S))
            ## S個をスレッド数できれいに割り切れたとき
            if multi_S * core_num == paras_dict["S"]:
                args = [(multi_S, paras_dict["Nu"], paras_dict["Ns"],
                         paras_dict["Nq"], comb_list) for i in range(core_num)]
            ## 割り切れなかったとき
            else:
                args = [(multi_S, paras_dict["Nu"], paras_dict["Ns"],
                         paras_dict["Nq"], comb_list) for i in range(core_num-1)]
                ## 最後の1つのスレッドに余り分をやらせる
                remain_S = paras_dict["S"] - multi_S * core_num
                args.append((multi_S+remain_S, paras_dict["Nu"], paras_dict["Ns"],
                             paras_dict["Nq"], comb_list))
            ## 並列実行の開始
            p = multiprocessing.Pool(core_num)
            returns = p.starmap(sim_random_clifford, args)
            p.close()
            ##　それぞれのスレッドの実行結果をひとまとめにする
            result = np.concatenate(returns, axis=0)            

        else:
            ## 量子回路シミュレーションと測定確率の計算
            result = sim_random_clifford(paras_dict["S"],  paras_dict["Nu"],
                                         paras_dict["Ns"], paras_dict["Nq"], comb_list)
    
    ## 処理終了時の時間の取得
    finish = time.perf_counter()
    ## 時間の取得
    dt_now = datetime.datetime.now()
    dt_index = dt_now.strftime("%Y%m%d%H%M%S")
    ## 結果の保存
    np.save("../result/clif_{}.npy".format(dt_index), result)
    ## パラメータの保存
    with open("../result/info_clif_{}.txt".format(dt_index), mode='w') as f:
        f.write("|S| : {}\n".format(paras_dict["S"]))
        f.write(" Nu : {}\n".format(paras_dict["Nu"]))
        f.write(" Ns : {}\n".format(paras_dict["Ns"]))
        f.write(" Nq : {}\n".format(paras_dict["Nq"]))
        if paras_dict["local"] == 1:
            f.write("depth : {}\n".format(paras_dict["depth"]))
            f.write("CNOT&1q clif : {}\n".format(paras_dict["CNOT_1qC"]))
        f.write("bit corrlation : {}\n".format(paras_dict["Nq"]))
        f.write("dim of moments : 1~20\n")
    ## 色々出力
    print('\nData is saved as "clif_{}.npy".'.format(dt_index))
    print('Information(parameters) of this data is in "info_clif_{}.npy".'.format(dt_index))
    print('Creating Time : {}[s].'.format(finish-start))


if __name__ == '__main__':
    ## コマンドライン引数で"seq"または"sequential"とあった場合の逐次実行
    parallel = True
    args = sys.argv
    if len(args) > 1:
        if "seq" in args or "sequential" in args:
            parallel = False
    main(parallel)    