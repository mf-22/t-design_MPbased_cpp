from group.clifford_group_circuit import CliffordCircuitGroup
from qulacs import QuantumState
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
    ## input number of data you want to create(|S|)
    print('input number of data (|S|) :', end=(' '))
    paras_dict["S"] = int(input())
    print('Nu :', end=(' '))
    paras_dict["Nu"] = int(input())
    print('Ns :', end=(' '))
    paras_dict["Ns"] = int(input())
    print('Nq :', end=(' '))
    paras_dict["Nq"] = int(input())
    print('Local random clifford(1:=Yes, 0:=No) :', end=(' '))
    local_in = int(input())
    if local_in == 1:
        paras_dict["local"] = True
        print('input circuit depth :', end=(' '))
        paras_dict["depth"] = int(input())
    else:
        paras_dict["local"] = False

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
    ## 測定確率(ビット相関の期待値)の計算結果を保存する配列を用意
    MP_list = np.empty(((S, Nu, len(comb_list))), dtype=float)
    ## 量子状態の準備
    state = QuantumState(Nq)
    ## 2qubitのクリフォード群を宣言、位数は11520
    ## 4×4の行列ではなくゲート列が返ってくる
    ccg = CliffordCircuitGroup(2)
    ##order = ccg.order <= 11520

    for i in range(S):
        ## 教師データ1つにつき初期状態を固定する、そのための乱数のシード
        haar_seed = np.random.randint(2147483648) #2^31
        for j in range(Nu):
            ## 初期状態をHaar random stateにする
            state.set_Haar_random_state(haar_seed)
            ## 2qubitのランダムクリフォードを互い違いにかけていく
            for k in range(1, depth+1):
                for qubit_index in RU_index_list[k%2]:
                    circuit = ccg.get_element(np.random.randint(11520))
                    ccg.simulate_circuit_specific_qubit(circuit, state, qubit_index)
            ## 測定を行う
            while True:
                sample_dec = state.sampling(Ns)
                if (2**Nq) not in sample_dec:
                    break
            ## 10進数表記の測定結果を2進数のビット列に変換
            result_bin = np.array([list(bin(n)[2:].zfill(Nq)) for n in sample_dec]).astype(np.int8)
            ## 0 => -1　に変換
            result_bin[result_bin == 0] = -1
            ## ビット相関を計算し、測定確率を計算
            for k, combination in enumerate(comb_list):
                bit_corr = result_bin[:, combination[0]]
                for index in range(1, len(combination)):
                    bit_corr *= result_bin[:, combination[index]]
                MP_list[i][j][k] = np.mean(bit_corr)
        print('\r{} / {} finished...'.format(i+1, S), end=(''))
    print('')
    
    return MP_list

def sim_random_clifford(S, Nu, Ns, Nq, comb_list):
    ## 測定確率(ビット相関の期待値)の計算結果を保存する配列を用意
    MP_list = np.empty(((S, Nu, len(comb_list))), dtype=float)
    ## 量子状態の準備
    state = QuantumState(Nq)
    ## 2qubitのクリフォード群を宣言
    ## (2^n)×(2^n)の行列ではなくゲート列が返ってくる
    ccg = CliffordCircuitGroup(Nq)
    ## クリフォード群の要素数を取得
    order = ccg.order

    for i in range(S):
        ## 教師データ1つにつき初期状態を固定する、そのための乱数のシード
        haar_seed = np.random.randint(2147483648) #2^31
        for j in range(Nu):
            ## 初期状態をHaar random stateにする
            state.set_Haar_random_state(haar_seed)
            ## ランダムクリフォードの実行
            circuit = ccg.get_element(gen_random_index(order))
            ccg.simulate_circuit(circuit, state)
            ## 測定を行う
            while True:
                sample_dec = state.sampling(Ns)
                if (2**Nq) not in sample_dec:
                    break
            ## 10進数表記の測定結果を2進数のビット列に変換
            result_bin = np.array([list(bin(n)[2:].zfill(Nq)) for n in sample_dec]).astype(np.int8)
            ## 0 => -1　に変換
            result_bin[result_bin == 0] = -1
            ## ビット相関を計算し、測定確率を計算
            for k, combination in enumerate(comb_list):
                bit_corr = result_bin[:, combination[0]]
                for index in range(1, len(combination)):
                    bit_corr *= result_bin[:, combination[index]]
                MP_list[i][j][k] = np.mean(bit_corr)
        print('\r{} / {} finished...'.format(i+1, S), end=(''))
    print('')
    
    return MP_list


def main(parallel = True):
    ## パラメータ読み込み
    paras_dict = input_parameters()
    ## ビット相関計算時のqubitのインデックスのリスト
    comb_list = gen_comb_list(paras_dict["Nq"])

    ## 処理開始時の時間の取得
    start = time.perf_counter()

    ## Local Random Cliffordの計算
    if paras_dict["local"]:
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
            p = multiprocessing.Pool(core_num)
            returns = p.starmap(sim_local_random_clifford, args)
            p.close()
            ##　それぞれのスレッドの実行結果をひとまとめにする
            result = np.concatenate(returns, axis=0)

        ## 逐次実行
        else:
            ## 量子回路シミュレーションと測定確率の計算
            result = sim_local_random_clifford(paras_dict["S"],  paras_dict["Nu"],
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
        if paras_dict["local"]:
            f.write("depth : {}\n".format(paras_dict["depth"]))
    ## 色々出力
    print('\nData is saved as "clif_{}.npy".'.format(dt_index))
    print('Information(parameters) of this data is in "info_clif_{}.npy".'.format(dt_index))
    print('Creating Time : {}[s].'.format(finish-start))
    print('\n\n  ***    All finished!!!    ***\n')


if __name__ == '__main__':
    ## コマンドライン引数で"seq"または"sequential"とあった場合の逐次実行
    args = sys.argv
    if len(args) > 1:
        if args[1] == "seq" or args[1] == "sequential":
            main(parallel=False)
        else:
            print('Argument Error')
    else:
        main(parallel=True)