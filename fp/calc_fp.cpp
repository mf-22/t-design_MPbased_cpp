#define _USE_MATH_DEFINES
#include <cppsim/state.hpp>
#include <Eigen/QR>
#include <iostream>
#include <fstream>
#include <numeric>
#include <vector>
#include <deque>
#include <chrono>
#include <iomanip>
#include <string>
#include <complex>
//#include <Eigen/KroneckerProduct>
#include <filesystem>
#include <cmath>

ComplexMatrix my_tensor_product(ComplexMatrix A, ComplexMatrix B) {
    /** テンソル積(クロネッカー積、np.kron(A,B))を計算する関数
     *   Args:
     *       A,B(ComplexMatrix) := 複素行列
     *   Return:
     *       A tensor B (np.kron(A,B)) := AとBのテンソル積の計算結果の行列
     */
    // AとBの行列の次元を取得
    unsigned long long int mat_A_dim = A.rows();
    unsigned long long int mat_B_dim = B.rows();
    // 計算結果を保存する行列を宣言
    ComplexMatrix C(mat_A_dim*mat_B_dim, mat_A_dim*mat_B_dim);
    // 行列の一部のブロックを表現する行列を宣言
    ComplexMatrix small_mat;

    //テンソル積の計算
    for (int i = 0; i < mat_A_dim; i++) {
        for (int j = 0; j < mat_A_dim; j++) {
            // 行列Aの要素で行列Bのスカラー倍を計算してブロックの行列を作る
            small_mat = A(i, j) * B;
            // テンソル積後の行列の適切な部分にブロック行列の値を代入する
            for (int k = 0; k < mat_B_dim; k++) {
                for (int l = 0; l < mat_B_dim; l++) {
                    C(k + i * mat_B_dim, l + j * mat_B_dim) = small_mat(k, l);
                }
            }
        }
    }
    return C;
}

/**
 * Ref: https://github.com/qulacs/qulacs/blob/master/src/cppsim/gate_factory.cpp#L180
 */
ComplexMatrix gen_haar_RU(unsigned int num_qubits) {
    /** n-qubitのHaar Random Unitaryを生成する関数。
     *  Qulacsに実装されているもの参考に実装した。
     *  Arg:
     *      num_qubit(int) := 量子ビット数
     *  Return:
     *      Q(ComplexMatrix) := Haar Random Unitaryの行列
     */
    Random random;
    unsigned long long int dim = pow(2, num_qubits);
    ComplexMatrix matrix(dim, dim);
    for (unsigned long long int i = 0; i < dim; ++i) {
        for (unsigned long long int j = 0; j < dim; ++j) {
            matrix(i, j) = (random.normal() + 1.i * random.normal()) / sqrt(2.);
        }
    }
    Eigen::HouseholderQR<ComplexMatrix> qr_solver(matrix);
    ComplexMatrix Q = qr_solver.householderQ();
    // actual R matrix is upper-right triangle of matrixQR
    auto R = qr_solver.matrixQR();
    for (unsigned long long int i = 0; i < dim; ++i) {
        CPPCTYPE phase = R(i, i) / abs(R(i, i));
        for (unsigned long long int j = 0; j < dim; ++j) {
            Q(j, i) *= phase;
        }
    }
    return Q;
}

/** 現在の日時をstring型で返す関数
 *  シードの初期値やファイル名などの指定に用いる
 *  例) 2021年6月29日22時間58分45秒 => "20210629225845"
 */
inline std::string getDatetimeStr() {
#if defined(__GNUC__) || defined(__INTEL_COMPILER)
    //Linux(GNU, intel Compiler)
    time_t t = time(nullptr);
    const tm* localTime = localtime(&t);
    std::stringstream s;
    s << "20" << localTime->tm_year - 100;
    // setw(),setfill()で0詰め
    s << std::setw(2) << std::setfill('0') << localTime->tm_mon + 1;
    s << std::setw(2) << std::setfill('0') << localTime->tm_mday;
    s << std::setw(2) << std::setfill('0') << localTime->tm_hour;
    s << std::setw(2) << std::setfill('0') << localTime->tm_min;
    s << std::setw(2) << std::setfill('0') << localTime->tm_sec;
#elif _MSC_VER
    //Windows(Visual C++ Compiler)
    time_t t;
    struct tm localTime;
    time(&t);
    localtime_s(&localTime, &t);
    std::stringstream s;
    s << "20" << localTime.tm_year - 100;
    // setw(),setfill()で0詰め
    s << std::setw(2) << std::setfill('0') << localTime.tm_mon + 1;
    s << std::setw(2) << std::setfill('0') << localTime.tm_mday;
    s << std::setw(2) << std::setfill('0') << localTime.tm_hour;
    s << std::setw(2) << std::setfill('0') << localTime.tm_min;
    s << std::setw(2) << std::setfill('0') << localTime.tm_sec;
#endif
    // std::stringにして値を返す
    return s.str();
}

class FramePotential {
private:
    /* メンバ変数 */
    unsigned int num_qubits;    //量子ビット数
    unsigned int depth;         //回路の深さ
    unsigned long long int dim; //行列の次元
    unsigned int t;             //t-designの次数t
    double epsilon;             //収束誤差
    unsigned int patience;      //収束判定に用いる数

    std::string circuit;        //"LRC"か"RDC"を指定
    unsigned int num_gates_depthOdd;  //奇数次の深さのときの2qubitのHaarランダムユニタリの数
    unsigned int num_gates_depthEven; //偶数次の深さのときの2qubitのHaarランダムユニタリの数

    std::vector<double> result_oneshot;
    std::vector<double> result_mean;

    /* ユニタリをランダムにサンプルする */
    ComplexMatrix sample_unitary();

    /* 収束を判定する */
    bool check_convergence();

public:
    /* コンストラクタ */
    FramePotential(std::string circ) {
        this->circuit = circ;
        this->num_qubits = 3;
        this->depth = 3;
        this->dim = 8;
        this->t = 3;
        this->num_gates_depthOdd = 1;
        this->num_gates_depthEven = 1;

        this->result_oneshot.clear();
        this->result_mean.clear();
    }

    /* パラメータのセッタ */
    void set_paras(unsigned int Nq, unsigned int D, unsigned int dim_t, double eps, unsigned int pat);

    /* FramePotentialの値を収束まで計算 */
    void calculate();

    /* 計算結果を出力 */
    void output();

    /* 計算結果をファイル出力 */
    void save_result(std::string file_name);

    /* 計算結果の取得 */
    double get_result();
};

void FramePotential::set_paras(unsigned int Nq, unsigned int D, unsigned int dim_t, double eps = 0.0001, unsigned int pat = 5) {
    //引数からパラメータ設定
    this->num_qubits = Nq;
    this->dim = pow(2, Nq);
    this->depth = D;
    this->t = dim_t;
    this->epsilon = eps;
    this->patience = pat;

    //LRCのときの準備
    if (this->circuit == "LRC") {
        this->num_gates_depthOdd = Nq / 2;
        if (Nq % 2 == 1) {
            this->num_gates_depthEven = Nq / 2;
        }
        else {
            this->num_gates_depthEven = (Nq / 2) - 1;
        }
    //LRC以外のときは0を代入しておく
    } else {
        this->num_gates_depthOdd = 0;
        this->num_gates_depthEven = 0;
    }
}

void FramePotential::calculate() {
    ComplexMatrix U, Vdag;
    unsigned long long int count = 1;

    this->result_oneshot.clear();
    this->result_mean.clear();

    while (true) {
        U = sample_unitary();
        Vdag = sample_unitary().adjoint();
        this->result_oneshot.emplace_back(pow(abs((U*Vdag).trace()), 2. * this->t));
        std::cout << "\r" << "  calculated " << count << " times..." << std::string(10, ' ');
        count++;
        if (check_convergence()) {
            break;
        }
    }
    std::cout << std::endl;

}

ComplexMatrix FramePotential::sample_unitary() {
    //最終的に返すユニタリ行列
    ComplexMatrix big_unitary = Eigen::MatrixXd::Identity(this->dim, this->dim);

    if (this->circuit == "LRC") {
        //LRCの１層のユニタリ行列
        ComplexMatrix small_unitary;
        for (int i = this->depth; i > 0; i--) {
            //2qubitのHaarランダムユニタリを生成しテンソル積を取っていく
            if (i % 2 == 1) {
                //回路の深さが奇数のとき
                small_unitary = gen_haar_RU(2);
                for (int j = 0; j < this->num_gates_depthOdd - 1; j++) {
                    small_unitary = my_tensor_product(small_unitary, gen_haar_RU(2));
                }
                //量子ビット数、回路の深さともに奇数のときはIdentityを最後につける
                if (this->num_qubits % 2 == 1) {
                    small_unitary = my_tensor_product(small_unitary, Eigen::MatrixXd::Identity(2, 2));
                }
            }
            else {
                //回路の深さが偶数のときは最初は必ずIdentity
                small_unitary = Eigen::MatrixXd::Identity(2, 2);
                for (int j = 0; j < this->num_gates_depthEven; j++) {
                    small_unitary = my_tensor_product(small_unitary, gen_haar_RU(2));
                }
                //量子ビット数が偶数、回路の深さが偶数のときはIdentityを最後につける
                if (this->num_qubits % 2 == 0) {
                    small_unitary = my_tensor_product(small_unitary, Eigen::MatrixXd::Identity(2, 2));
                }
            }
            //行列をかけて各深さごとに作成した(num_qubits)サイズのユニタリをマージしていく
            big_unitary *= small_unitary;
        }
    }
    else if(this->circuit == "RDC") {
        //2*2のHゲート(1qubit Hadmard)の行列を作る
        ComplexMatrix hadmard_matrix_2d = Eigen::MatrixXd(2, 2);
        hadmard_matrix_2d(0, 0) = (1/sqrt(2)) * 1;
        hadmard_matrix_2d(0, 1) = (1/sqrt(2)) * 1;
        hadmard_matrix_2d(1, 0) = (1/sqrt(2)) * 1;
        hadmard_matrix_2d(1, 1) = (1/sqrt(2)) * -1;
        //num_qubitにかかるHゲート(n-qubit Hadmard)の行列を作る
        ComplexMatrix hadmard_matrix = hadmard_matrix_2d;
        for(int i=0;i<this->num_qubits-1;i++) {
            hadmard_matrix = my_tensor_product(hadmard_matrix, hadmard_matrix_2d);
        }
        //ランダム対角ユニタリの対角成分を要素に持つベクトルを作成
        ComplexMatrix RDU = Eigen::MatrixXd::Identity(this->dim, this->dim);
        //RDCの作成
        for(int i=0;i<this->depth;i++) {
            //RDUの対角成分に値を入れる
            for(int j=0;j<this->dim;j++) {
                Random random;
                RDU(j,j) = std::exp(std::complex<float>(0.0, random.uniform() * 2 * M_PI));
            }
            //RDUとn-qubit Hadmardを交互にかけることをdepth回繰り返す
            big_unitary *= hadmard_matrix * RDU;
        }
    }
    else {
        std::cerr << "CAUTION: The circuit '" << this->circuit << "' is not implemented. Identity will be returned." << std::endl;
    }

    return big_unitary;
}

bool FramePotential::check_convergence() {
    bool flag = true;
    unsigned long long int num_calclated = this->result_oneshot.size();

    if (num_calclated == 1) {
        this->result_mean.emplace_back(result_oneshot[0]);
        flag = false;
    }
    else if (num_calclated < this->patience) {
        this->result_mean.emplace_back(((num_calclated - 1)*this->result_mean.back() + result_oneshot.back()) / num_calclated);
        flag = false;
    }
    else {
        this->result_mean.emplace_back(((num_calclated - 1)*this->result_mean.back() + result_oneshot.back()) / num_calclated);
        for (int i = 0; i < this->patience; i++) {
            if (abs(this->result_mean.back() - this->result_mean[num_calclated - 1 - this->patience + i]) >= this->epsilon) {
                flag = false;
                break;
            }
        }
    }
    return flag;
}

void FramePotential::output() {
    std::cout << std::endl << "*** Result ***" << std::endl;
    std::cout << "num_qubits : " << this->num_qubits << std::endl;
    std::cout << "depth : " << this->depth << std::endl;
    std::cout << "t : " << this->t << std::endl;
    std::cout << "epsilon : " << this->epsilon << std::endl;
    std::cout << "patience : " << this->patience << std::endl;
    std::cout << "FramePotential : " << this->result_mean.back() << std::endl << std::endl;
}

void FramePotential::save_result(std::string file_name = getDatetimeStr() + ".csv") {
    std::string path = "./result/" + file_name;

    std::ofstream csv_file(path);
    csv_file << "circ:" << this->circuit << ",t:" << this->t << std::endl;
    csv_file << "Nq:" << this->num_qubits << ",depth:" << this->depth << std::endl;
    csv_file << "epsilon:" << this->epsilon << ",patience:" << this->patience << std::endl;
    csv_file << "shot,average" << std::endl;

    for (unsigned long long int i = 0; i < this->result_oneshot.size(); i++) {
        csv_file << this->result_oneshot[i] << "," << this->result_mean[i] << std::endl;
    }
    csv_file.close();
}

double FramePotential::get_result() {
    return this->result_mean.back();
}


int main() {
    //実行例
    /* パラメータの指定 */
    int ntimes = 10;
    std::vector<int> Nq_list = { 10 };
    std::vector<int> depth_list = { 8, 9,10,11,12 };
    std::vector<int> t_list = { 3 };
    /* 回路の指定 */
    std::string circ_type = "LRC";
    //std::string circ_type = "RDC";
    
    /* クラス呼び出し */
    FramePotential FP = { circ_type };
    
    /* 実行 */
    for (int i = 0; i < t_list.size(); i++) {
        for (int j = 0; j < Nq_list.size(); j++) {
            for (int k = 0; k < depth_list.size(); k++) {
                //パラメータのセット
                FP.set_paras(Nq_list[j], depth_list[k], t_list[i]);
                std::cout << std::endl << "Now => Nq:" << Nq_list[j] << ", depth:" << depth_list[k] << ", t:" << t_list[i] << std::endl;
                //ディレクトリ名
                std::string dir_name = circ_type
                                        + "_Nq" + std::to_string(Nq_list[j])
                                        + "_depth" + std::to_string(depth_list[k])
                                        + "_t" + std::to_string(t_list[i]);
                //ディレクトリ作成
                std::filesystem::create_directory("result/" + dir_name);
                //平均と標準偏差の値を計算するための変数
                double ave = 0.0, std = 0.0;
                //ntimes回実行
                for(int n=0; n < ntimes; n++) {
                    //計算開始
                    FP.calculate();
                    //FP.output();
                    //ログを保存
                    FP.save_result(dir_name + "/n=" + std::to_string(n) + ".csv");
                    //計算結果を取得し足し込む
                    ave += FP.get_result();
                    std += (FP.get_result() * FP.get_result());
                }
                //平均の計算
                ave /= ntimes;
                //分散の計算
                std = sqrt(std / ntimes - ave * ave);
                //計算結果の出力
                std::cout << "  result : " << ave << "±" << std << std::endl;
                //計算結果の保存
                std::ofstream result_file("./result/" + dir_name + "/ave_std.txt");
                result_file << ave << "±" << std << std::endl;
                result_file.close();
            }
        }
    }

    return 0;
}