def get_num_qubits(data_name):
    """ 教師データ作成時のパラメータを保存してあるファイルから量子ビット数の値を読んで返す
        Arg:
            data_name(string) := データセットのフォルダ名の文字列
        Return:
            Nq(int) := データ作成時の量子ビット数
    """
    with open("./datasets/{}/info.txt".format(data_name), mode="r") as f:
        l_strip = [s.strip() for s in f.readlines()]
        for line in l_strip:
            if "Nq :" in line:
                chars = line.split(":")
                Nq = int(chars[-1])
                break
    
    return Nq

def get_birCorr_moment(dir_path):
    """ 計算したビット相関の個数と、モーメントの次数を取得する
        Arg:
            dir_path(string)  := 実験結果が保存されるフォルダ名の文字列
        Returns:
            k(int)             := 計算した相関のビット数
            k_prime_list(list) := 計算したモーメントの次数の数値(int)を要素に持つリスト
    """

    with open(dir_path+"extract_parameters.txt", mode="r") as f:
        l_strip = [s.strip() for s in f.readlines()]
        for line in l_strip:
            if "k :" in line:
                chars = line.split(":")
                k = int(chars[-1])
            
            elif "k` :" in line:
                chars = line.split(":")[-1]
                moment_list = chars[2:-1].split(",")
                k_prime_list = [int(i) for i in moment_list]

    return k, k_prime_list