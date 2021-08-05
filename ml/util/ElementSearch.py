from . import read_parameter
import itertools
from scipy.special import comb

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

class Element_Searcher():
    """ PCAやRFで教師データのうち重要度の高い特徴量ベクトルの次元のインデックスが得られる。
        このインデックスのデータは、何点ビット相関で何次のモーメントなのかを求める。
        ただし、モーメントを連続で取っていないと正しい値が返らない
        OK => [1,2,3,4]や[4,5,6,7,8], NG =>[1,3,5,7,9]や[4,6,8,10]
    """
    def __init__(self, data_name, dir_path):
        """ コンストラクタ
        """
        ## 指定された次元から求めたいビット相関の位置とモーメントの次数
        self.bitcorr = []
        self.moment = 0
        self.dir_path = dir_path
        ## パラメータ取得
        Nq = read_parameter.get_num_qubits(data_name)
        k, self.k_prime_list = read_parameter.get_birCorr_moment(dir_path)
        ## ビット相関リストの取得
        self.bitcorr_list = gen_comb_list(Nq)
        ## Nqとkから、何点までのビット相関を求めているかとそのときのビットの位置はどこか計算
        self.corr_num = 0
        for i in range(k):
            self.corr_num += comb(Nq, i, exact=True)
        if Nq != k:
            self.bitcorr_list = self.bitcorr_list[:self.corr_num]
    
    def search(self, dim):
        """ 指定された次元から実際に計算
        """
        self.bitcorr = self.bitcorr_list[dim % self.corr_num]
        self.moment = self.k_prime_list[(dim // self.corr_num)]
    
    def output(self):
        """ 求まった求めたいビット相関の位置とモーメントの次数を出力
        """
        print("bitcorr:{}, k':{}".format(self.bitcorr, self.moment))
    
    def search_and_save_all(self, dims, coef, filename=""):
        """ 指定された次元から実際に計算
            結果はtsvファイル(水平タブ区切り)に保存する
        """
        ## 結果を保存するファイルを開く
        if len(filename) == 0:
            ## ファイル名の指定が無いときはfv_components.tsvという名前で保存
            result_file = open(self.dir_path+"fv_components.tsv", mode="w")
        else:
            ## ファイル名の指定が無いときは[filename].tsvという名前で保存
            result_file = open(self.dir_path+filename+".tsv", mode="w")

        for dim in dims:
            self.bitcorr = self.bitcorr_list[dim % self.corr_num]
            self.moment = self.k_prime_list[(dim // self.corr_num)]
            result_file.write("{}\t{}\t{}\n".format(self.bitcorr, self.moment, coef[dim]))
        
        result_file.close()