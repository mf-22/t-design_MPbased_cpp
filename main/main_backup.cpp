#define _USE_MATH_DEFINES
#include <cmath>
#include <cppsim/state.hpp>
#include <cppsim/state_dm.hpp>
#include <cppsim/circuit.hpp>
#include <cppsim/gate_factory.hpp>
#include <cppsim/gate_merge.hpp>
#include <cppsim/gate_matrix.hpp>
#include <cppsim/gate_general.hpp>
#include <iostream>
#include <fstream>
#include <numeric>
#include <chrono>
#include <iomanip>
#include <algorithm>


std::complex<float> gen_RandomDiagonal_element(ITYPE dummy) {
    //ランダム対角ユニタリ行列の対角成分を出力する関数を定義
    //calculate exp( (0,2pi] * i )
    Random random;
    return std::exp(std::complex<float>(0.0, random.uniform() * 2 * M_PI));
}

void recursive_set(int n, int r, std::vector<std::vector<int>>& v, std::vector<int>& pattern) {
    /* 引数の説明
    int n := qubit数
    int r := ビット相関数(k点ビット相関)
    std::vector v := create_comb_list関数から参証渡しされたリスト
                このリストにビットの組み合わせを保存する
    std::vector pattern := 基準になるパターンのリスト
                      この最後の要素をインクリメントして組み合わせを作る
                      関数から戻すときは更新してから返す
    */
    
    /* 変数の説明
    std::vector temp := patternを書き換えてvにプッシュしていくが、もとのpattern
                   の配列は今後も使いたい.そこでtempというpatternのコピー
                   を作ってtempを弄ってvにpushしていく
    int poped := patternの末尾をpopしたときに得られた値
    int poped_next := patternを連続してpopするとき、次にpopする予定の位置の値
    int poped_thre := patternを連続してpopするか決定するための閾値
    */

    //patternの長さがrのとき => 組み合わせ列挙開始
    if (pattern.size() == r) {
        std::vector<int> temp = pattern;
        int poped;
        int pop_next;
        int pop_thre = n-1;

        //patternの最後尾をインクリメントしていき組み合わせのリストvにpushしていく
        for (int i=pattern[r-1];i<n;i++) {
            v.push_back(temp);
            temp[r-1] = i + 1;
        }

        //patternを更新
        while(true) {
            poped = pattern.back();
            pattern.pop_back();

            //popしてpatternが空になったとき => 組み合わせの列挙が終わっているはずなので終了
            if (pattern.size() == 0) {
                break;
            }
            //popedがpop_threのとき => 次を見る
            else if (poped == pop_thre) {
                pop_next = pattern.back();
                //pop_nextがpop_thre-1じゃないとき => patternを更新してbreak
                if (pop_next != pop_thre-1){
                    poped = pattern.back() + 1;
                    pattern.pop_back();
                    pattern.push_back(poped);
                    break;
                }
                //pop_nextがpop_thre-1のとき => pop_threをデクリメントし,もう一度このwhileループ
                else {
                    pop_thre = pop_thre - 1;
                }
            }
            //popedがpop_threでないとき => 末尾の値+1してbreak
            else {
                poped = pattern.back() + 1;
                pattern.pop_back();
                pattern.push_back(poped);
                break;
            }
        }
    }
    //patternの長さがrより小さいとき => 適切な初期patternになるよう追加し再帰
    else {
        int last_index = pattern.size()-1;
        pattern.push_back(pattern[last_index]+1);
        recursive_set(n, r, v, pattern);
    }
}


void create_comb_list(int n, int r, std::vector<std::vector<int>>& comb_list) {
    /* 引数の説明
    int n := qubit数
    int r := ビット相関数(k点ビット相関)
             組み合わせはよく"nCr"と書くので揃えてみた
    std::vector comb_list := main関数などから参照渡しされたリスト
                        このリストにビットの組み合わせを保存する
    */
    
    /* 変数の説明
    p_list := 基準になるパターンのリスト
              これをrecursive_setに渡して
              最後をインクリメントして組み合わせを作る
              r=1,nのときは簡単に返せる
    */

    //1点相関のとき => 単に各インデックスを2次元配列で返す
    if (r == 1) {
        std::vector<int> p_list = {0};
        for(int i=0;i<n;i++) {
            comb_list.push_back(p_list);
            p_list[0] = p_list[0] + 1;
        }
    }
    //n点相関(例.5qubit5点相関)のとき => 単に0からnまでの1次元のリストを返す
    else if (n == r) {
        std::vector<int> p_list;
        for(int i=0;i<n;i++) {
            p_list.push_back(i);
        }
        comb_list.push_back(p_list);
    }
    //上記以外のk点相関のとき
    else {
        std::vector<int> p_list;
        for(int i=0;i<r;i++) {
            p_list.push_back(i);
        }
        while (p_list.size() > 0) {
            recursive_set(n, r, comb_list, p_list);
        }
    }
}

std::vector<std::vector<int>> get_possibliyMax_bitCorr(unsigned int num_qubits) {
    std::vector<std::vector<int>> bitCorr_list;
    
    for(int i=1;i<num_qubits+1;i++) {
        create_comb_list(num_qubits, i, bitCorr_list);
    }
    
    return bitCorr_list;
}

std::string getDatetimeStr() {
    //Linux(GNU Compiler)
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
    
    //Windows(Visual C++ Compiler)
    /*
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
    */
    
    // std::stringにして値を返す
    return s.str();
}

class DataCreator {
private:
    /* メンバ変数 */
    //各パラメータを保存する辞書型のデータ
    std::map<std::string, std::string> parameters;
    //ユニタリの種類(0:Haar, 1:Clifford, 2:LRC, 3:RDC)
    unsigned int unitary_type;
    //教師データのサイズ
    unsigned int S;
    //教師データ1つに含まれる教師データのサイズ
    unsigned int Nu;
    //1つのユニタリに対して行われる測定回数
    unsigned int Ns;
    //量子ビット数
    unsigned int Nq;
    //回路の深さ
    unsigned int depth;
    //乱数シード
    unsigned int seed;
    //ノイズの種類(0:なし, 1:Depolarizing, 2:Measurement)
    unsigned int noise_operator;
    //ノイズの割合(1-pでI、pでノイズ)
    float noise_prob;
    //ビット相関を計算する際のインデックスのリスト
    std::vector<std::vector<int>> comb_list;
    //最終的に作成するデータ, 測定確率(ビット相関込み)が保存されるベクトル
    std::vector<std::vector<std::vector<float>>> sim_result;
    
    /* メンバ関数 */
    //設定ファイルを読みに行く
    std::map<std::string, std::string> _read_inputfile();
    //シミュレーションの前に行っておく処理をする
    void _run_preprocess();
    //実際にシミュレーションを行う関数
    void _haar_sim();
    void _lrc_sim(std::vector<std::vector<std::vector<unsigned int>>>& RU_index_list);
    void _lrc_depolarizing_sim(std::vector<std::vector<std::vector<unsigned int>>>& RU_index_list);
    void _lrc_MeasurementInduced_sim(std::vector<std::vector<std::vector<unsigned int>>>& RU_index_list);
    void _rdc_sim();
    //サンプリング結果のビット列から測定確率を計算
    std::vector<float> _calc_BitCorr_and_MP(std::vector<ITYPE>& sampling_result);
    //int型の10進数の値を2進数のvector型に変換
    void _decimal_to_binarylist(std::vector<ITYPE>& result_dec, std::vector<std::vector<int>>& result_bin);

public:
    /* コンストラクタ */
	DataCreator() {
        //パラメータの初期値を設定
        this->parameters["unitary_type"] = "0";
        this->parameters["S"] = "1000";
        this->parameters["Nu"] = "100";
        this->parameters["Ns"] = "100";
        this->parameters["Nq"] = "4";
        this->parameters["depth"] = "5";
        this->parameters["seed"] = getDatetimeStr();
        this->parameters["noise_operator"] = "0";
        this->parameters["noise_prob"] = "0.01";
	}

    //辞書型で一括でパラメータをセット
    void set_all_parameters();
    //パラメータをキーと値でセット
    void set_parameter(std::string key, std::string val);
    //データ生成の開始
    void run_simulation();
    //シミュレーション結果の保存
    void save_result();

};

std::map<std::string, std::string> DataCreator::_read_inputfile() {
    //設定ファイルのパラメータの辞書型変数
    std::map<std::string, std::string> input_paras;

    //ファイルオープン
    std::ifstream ifs("config_simulation.txt");
    std::string str;

    if (ifs.fail()) {
        //ファイルが存在しない
        std::cerr << "Failed to open file. Default setting will be loaded." << std::endl;
    } else {
        //ファイル読み込み
        std::string parameter_name;
        while (getline(ifs, str)) {
            //先頭が"#"の場合はコメント行としてスキップ
            if(str[0] != '#') {
                //":"で分割しパラメータ名を取得
                parameter_name = split(str, ":").front();
                auto itr = this->parameters.find(parameter_name);
                //キーが存在するとき
                if( itr != input_paras.end() ) {
                    input_paras[parameter_name] = split(str, ":").back();
                }
            }
        }
    }
    return input_paras;
}

void DataCreator::set_all_parameters() {
    //入力ファイルの内容を順番に反映していく
    auto input_paras = _read_inputfile();
    for(auto itr = input_paras.begin(); itr != input_paras.end(); ++itr) {
        if(this->parameters.find(itr->first) != this->parameters.end()) {
            this->parameters[itr->first] = itr->second;
        } else{
            std::cerr << "Key error: " << itr->first << std::endl;
        }
    }
}

void DataCreator::set_parameter(std::string key, std::string val) {
    if(this->parameters.find(key) != this->parameters.end()) {
        this->parameters[key] = val;
        /*
        if(key == "seed" && val == "time") {
            this->parameters["seed"] = getDatetimeStr();
        } else {
            this->parameters[key] = val;
        }*/
    } else{
        std::cerr << "Key error: " << key << std::endl;
    }
}

void DataCreator::_haar_sim() {
    //測定結果を保存するベクトル
    std::vector<ITYPE> sampling_result;

    //ループカウンタ
    int i,j;
    
    //教師データ作成
    #pragma omp parallel for private(j, sampling_result)
    for(i=0;i<this->S;i++) {
        for(j=0;j<this->Nu;j++) {
            //量子状態の生成と初期化
            QuantumState state(this->Nq);
            state.set_Haar_random_state();
            //測定と測定確率(ビット相関も)の計算
            sampling_result = state.sampling(this->Ns);
            this->sim_result[i][j] = _calc_BitCorr_and_MP(sampling_result);
        }
        std::cout << "\r" << i+1 << "/" << this->S << " finished..." << std::string(20, ' ');
    }
    std::cout << std::endl;
}

void DataCreator::_lrc_sim(
    std::vector<std::vector<std::vector<unsigned int>>>& RU_index_list) {

    //測定結果を保存するベクトル
    std::vector<ITYPE> sampling_result;

    //ループカウンタ
    int i,j,l;
    //教師データ作成
    #pragma omp parallel for private(j, l, sampling_result)
    for(i=0;i<this->S;i++) {
        for(j=0;j<this->Nu;j++) {
            //量子状態の作成と初期化
            QuantumState state(this->Nq);
            //state.set_zero_state();
            //LRCの実行
            for(l=1;l<this->depth+1;l++) {
                for(const auto& qubit_indecies : RU_index_list[l%2]) {
                    gate::RandomUnitary(qubit_indecies)->update_quantum_state(&state);
                }    
            }
            //測定と測定確率(ビット相関も)の計算
            sampling_result = state.sampling(this->Ns);
            this->sim_result[i][j] = _calc_BitCorr_and_MP(sampling_result);
        }
        std::cout << "\r" << i+1 << "/" << this->S << " finished..." << std::string(20, ' ');
    }
    std::cout << std::endl;
}

void DataCreator::_lrc_depolarizing_sim(
    std::vector<std::vector<std::vector<unsigned int>>>& RU_index_list) {
    
    //測定結果を保存するベクトル
    std::vector<ITYPE> sampling_result;

    //ノイズモデルの定義
    //各qubitにかかるCPTPを事前に回路として作っておき各層で適用する
    QuantumCircuit depolarizing_circuit(this->Nq);
    //回路の作成
    for(int i=0;i<this->Nq;i++) {
        depolarizing_circuit.add_gate(gate::DepolarizingNoise(i, this->noise_prob));
    }

    //ループカウンタ
    int i,j,l;
    //教師データ作成
    #pragma omp parallel for private(j, l, sampling_result)
    for(i=0;i<this->S;i++) {
        for(j=0;j<this->Nu;j++) {
            //量子状態(密度行列)の生成と初期化
            DensityMatrix state(this->Nq);
            //state.set_zero_state();
            //ノイズありのLRCの実行
            for(l=1;l<this->depth+1;l++) {
                for(const auto& qubit_indecies : RU_index_list[l%2]) {
                    gate::RandomUnitary(qubit_indecies)->update_quantum_state(&state);
                }
                //ノイズの適用
                depolarizing_circuit.update_quantum_state(&state);
            }
            //測定と測定確率(ビット相関も)の計算
            sampling_result = state.sampling(this->Ns);
            this->sim_result[i][j] = _calc_BitCorr_and_MP(sampling_result);
        }
        std::cout << "\r" << i+1 << "/" << this->S << " finished..." << std::string(20, ' ');
    }
    std::cout << std::endl;
}

void DataCreator::_lrc_MeasurementInduced_sim(
    std::vector<std::vector<std::vector<unsigned int>>>& RU_index_list) {

    //測定結果を保存するベクトル
    std::vector<ITYPE> sampling_result;

    //ノイズモデルの定義
    //各qubitにかかるCPTPを事前に回路として作っておき各層で適用する
    QuantumCircuit p_measure_circuit(this->Nq);
    //Identity
    ComplexMatrix dim2_matrix = Eigen::MatrixXd::Zero(2, 2);
    dim2_matrix(0, 0) = 1;
    dim2_matrix(1, 1) = 1;
    ComplexMatrix kraus_identity = sqrt(1-this->noise_prob) * dim2_matrix;
    //0測定
    dim2_matrix(1, 1) = 0;
    ComplexMatrix kraus_measure_0 = sqrt(this->noise_prob) * dim2_matrix;
    //1測定
    dim2_matrix(0, 0) = 0;
    dim2_matrix(1, 1) = 1;
    ComplexMatrix kraus_measure_1 = sqrt(this->noise_prob) * dim2_matrix;
    //回路の作成
    for(int i=0;i<this->Nq;i++) {
        p_measure_circuit.add_gate(
            gate::CPTP({
                gate::DenseMatrix(i, kraus_identity),
                gate::DenseMatrix(i, kraus_measure_0),
                gate::DenseMatrix(i, kraus_measure_1)
            })
        );
    }
    
    //ループカウンタ
    int i,j,l;
    //教師データ作成-sequential
    #pragma omp parallel for private(j, l, sampling_result)
    for(i=0;i<this->S;i++) {
        for(j=0;j<this->Nu;j++) {
            //量子状態の生成と初期化
            QuantumState state(this->Nq);
            //state.set_zero_state();
            //LRCの実行
            for(l=1;l<this->depth+1;l++) {
                for(const auto& qubit_indecies : RU_index_list[l%2]) {
                    gate::RandomUnitary(qubit_indecies)->update_quantum_state(&state);
                }
                //測定の適用
                p_measure_circuit.update_quantum_state(&state);
            }
            //2層のLRCの追加
            for(l=1;l>-1;l--) {
                //l=1(奇数層)とl=0(偶数層)で実行
                for(const auto& qubit_indecies : RU_index_list[l]) {
                    gate::RandomUnitary(qubit_indecies)->update_quantum_state(&state);
                }
            }
            //量子状態の正規化
            state.normalize(state.get_squared_norm());
            //測定と測定確率(ビット相関も)の計算
            sampling_result = state.sampling(this->Ns);
            this->sim_result[i][j] = _calc_BitCorr_and_MP(sampling_result);
        }
        std::cout << "\r" << i+1 << "/" << this->S << " finished..." << std::string(20, ' ');
    }
    std::cout << std::endl;
}

void DataCreator::_rdc_sim() {
    //測定結果を保存するベクトル
    std::vector<ITYPE> sampling_result;
    
    //基底を変更する、全てのqubitにHadmardを適用する回路
    QuantumCircuit basis_change_circuit(this->Nq);
    for(int i=0;i<this->Nq;i++) {
        basis_change_circuit.add_H_gate(i);
    }

    //ループカウンタ
    int i,j,d;
    //教師データ作成
    #pragma omp parallel for private(j, d, sampling_result)
    for(i=0;i<this->S;i++) {
        for(j=0;j<this->Nu;j++) {
            //量子状態の生成と初期化
            QuantumState state(this->Nq);
            //state.set_zero_state();
            //RDCの実行
            for(d=0;d<this->depth;d++) {
                state.multiply_elementwise_function(gen_RandomDiagonal_element);
                basis_change_circuit.update_quantum_state(&state);
            }
            //測定と測定確率(ビット相関も)の計算
            sampling_result = state.sampling(this->Ns);
            sim_result[i][j] = _calc_BitCorr_and_MP(sampling_result);
        }
        std::cout << "\r" << i+1 << "/" << this->S << " finished..." << std::string(20, ' ');
    }
    std::cout << std::endl;
}

void DataCreator::_run_preprocess() {
    //値を取り出して各変数にセット
    this->unitary_type = std::stoi(this->parameters["unitary_type"]);
    this->S = std::stoi(this->parameters["S"]);
    this->Nu = std::stoi(this->parameters["Nu"]);
    this->Ns = std::stoi(this->parameters["Ns"]);
    this->Nq = std::stoi(this->parameters["Nq"]);
    this->depth = std::stoi(this->parameters["depth"]);
    this->seed = std::stoi(this->parameters["seed"]);    
    this->noise_operator = std::stoi(this->parameters["noise_operator"]);
    this->noise_prob = std::stof(this->parameters["noise_prob"]);
    
    //設定を出力
    std::cout << "** Parameters **" << std::endl;
    for(auto itr = this->parameters.begin(); itr != this->parameters.end(); ++itr) {
        std::cout << itr->first << " : " << itr->second << std::endl;
    }
    std::cout << std::endl;

    //ビット相関を計算する量子ビットの位置のリストを取得
    //1点からNq点までの、取り得るすべてのビット相関を事前に計算
    this->comb_list.clear();
    this->comb_list = get_possibliyMax_bitCorr(this->Nq);

    //測定確率が保存されるvectorのサイズを事前に決定
    this->sim_result.clear();
    this->sim_result.resize(this->S, std::vector<std::vector<float>>(this->Nu, std::vector<float>(this->comb_list.size())));
}

void DataCreator::run_simulation() {
    // 計測開始時間
    auto start = std::chrono::system_clock::now();

    //前処理の実行
    _run_preprocess();

    if(this->unitary_type == 0) {
        _haar_sim();
    } else if(this->unitary_type == 2) {
        /**
         * 2qubitのHaarランダムユニタリがかかるインデックスのリストを用意
         * circuit.add_Random_unitary_gate()で渡す
         * 構造は3次元のvectorとし、1次元目は深さが偶数or奇数、2次元目は
         * 2qubit Haarゲートのインデックス、3次元目は量子ビットのインデックスとする
         * (例) 6qubitのLRCのとき 
         *      RU_index_list = [[[1,2],   [3,4]          ],  ←深さが偶数のとき([0])
         *                        [0,1],   [2,3],   [4,5] ]]  ←深さが奇数のとき([1])
         *                        ↑1つ目の2qubit Haarがかかるqubitのインデックス
         */
        //深さが奇数と偶数のときの2qubit Haarゲートがかかる量子ビットの
        //インデックスを保持する3次元のベクトル
        std::vector<std::vector<std::vector<unsigned int>>> RU_index_list;
        //深さが奇数、または偶数のときどちらかのインデックスを保持するベクトル
        std::vector<std::vector<unsigned int>> target_index;
        //長さが2の量子ビットのインデックスを保持する1次元のベクトル
        std::vector<unsigned int> index_twoQubit;
        //実際に作成
        int count = 0;
        //深さが奇数のときの量子ビットのインデックスのリストを作る
        for(int i=1;i<Nq-1;i++) {
            index_twoQubit.emplace_back(i);
            count++;
            if(count == 2){
                target_index.emplace_back(index_twoQubit);
                index_twoQubit.clear();
                count = 0;
            }
        }
        RU_index_list.emplace_back(target_index);
        target_index.clear();
        //深さが偶数のときの量子ビットのインデックスのリストを作る
        unsigned int maxIndex_depthOdd;
        if (Nq%2 == 1) {
            maxIndex_depthOdd = Nq - 1;
        } else {
            maxIndex_depthOdd = Nq;
        }
        for(int i=0;i<maxIndex_depthOdd;i++) {
            index_twoQubit.emplace_back(i);
            count++;
            if(count == 2){
                target_index.emplace_back(index_twoQubit);
                index_twoQubit.clear();
                count = 0;
            }
        }
        RU_index_list.emplace_back(target_index);

        if(this->noise_operator == 0) {
            _lrc_sim(RU_index_list);
        }
        else if(this->noise_operator == 1) {
            _lrc_depolarizing_sim(RU_index_list);
        }
        else if(this->noise_operator == 2) {
            _lrc_MeasurementInduced_sim(RU_index_list);
        }
    } else if(this->unitary_type == 2) {
        _rdc_sim();
    }

    // 計測終了時間
    auto finish = std::chrono::system_clock::now();
    //処理に要した時間をミリ秒に変換して1000で割って[s]に
    double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(finish-start).count() / 1000.0;
    std::cout << "Creating time : " << elapsed << "[s]" << std::endl;
}

std::vector<float> DataCreator::_calc_BitCorr_and_MP(std::vector<ITYPE>& sampling_result) {
    //測定確率が入ったリスト
    std::vector<float> MP_list;
    //測定結果のビット列(0または1)を保持するリスト、-1で初期化
    std::vector<std::vector<int>> sampling_result_bin(this->Ns, std::vector<int>(this->Nq, -1));
    //測定結果を10進数から2進数に変換、-1で初期化してあるので、2進数にしたときの1の部分を書き換えれば良い
    _decimal_to_binarylist(sampling_result, sampling_result_bin); 

    //各測定結果に対してビット相関を計算するので、shots回分のビット相関の値が出てくる。
    //これを足しこんでいき、最後に測定回数で割れば測定確率(期待値=:ビット相関の値)となる。
    float sum_bitcorr;
    int bitcorr_oneshot;
    
    //測定確率の計算
    for(const auto& bit_index_list : this->comb_list){
        //ビット相関を計算する位置を取得
        sum_bitcorr = 0.0;
        for(const auto& result_oneshot : sampling_result_bin) {
            bitcorr_oneshot = 1;
            for(const auto& qubit_index : bit_index_list) {
                bitcorr_oneshot *= result_oneshot[qubit_index];
            }
            sum_bitcorr += bitcorr_oneshot;
        }
        MP_list.emplace_back(sum_bitcorr/this->Ns);
    }

    return MP_list;
}

void DataCreator::_decimal_to_binarylist(std::vector<ITYPE>& result_dec, std::vector<std::vector<int>>& result_bin) {
    int meas_result;

    for(int i=0;i<result_dec.size();i++) {
        meas_result = result_dec[i];
        if(meas_result != 0) {
            for(int j=0;j<log2(result_dec[i])+1;j++) {
                if(meas_result%2 == 1) {
                    result_bin[i][this->Nq-1-j] = 1;
                }
                meas_result /= 2;
            }
        }
    }
}

void DataCreator::save_result() {
    //ID(時刻)を取得
    std::string date_ID = getDatetimeStr();
    std::string unitary_str;
    //ユニタリが何か設定
    if (this->unitary_type == 0) {
        unitary_str = "haar";
    } else if (this->unitary_type == 1) {
        unitary_str = "clif";
    } else if (this->unitary_type == 2) {
        unitary_str = "lrc";
    } else if (this->unitary_type == 3) {
        unitary_str = "nakata";
    }

    //測定確率をCSVファイルで保存
    std::string file_name = "../result/" + unitary_str + "_" + date_ID + ".csv";
    std::ofstream csv_file(file_name);
    for(const auto& each_S : this->sim_result) {
        for(const auto& each_U : each_S) {
            for(auto itr=each_U.begin(); itr!=each_U.end()-1; ++itr) {
                csv_file << *itr << ",";
            }
            csv_file << each_U.back() << std::endl;
        }
        csv_file << std::endl;
    }
    csv_file.close();

    //シミュレーション時のパラメータなどをテキストファイルに保存
    file_name = "../result/info_" + unitary_str + '_' + date_ID + ".txt";
    std::ofstream config_file(file_name);
    config_file << "|S| : " << this->S << std::endl;
    config_file << " Nu : " << this->Nu << std::endl;
    config_file << " Ns : " << this->Ns << std::endl;
    config_file << " Nq : " << this->Nq << std::endl;
    if(this->unitary_type == 2 || this->unitary_type == 3) {
        config_file << " depth : " << this->depth << std::endl;
    }
    config_file << " seed : " << this->seed << std::endl;
    if(this->unitary_type == 2) {
        if(this->noise_operator == 0) {
            config_file << " noise_operator : none" << std::endl;
        } else if(this->noise_operator == 1) {
            config_file << " noise_operator : DepolarizingNoise" << std::endl;
            config_file << " noise_prob : " << this->noise_prob << std::endl;
        } else if(this->noise_operator == 2) {
            config_file << " noise_operator : Measurement-Induced" << std::endl;
            config_file << " noise_prob : " << this->noise_prob << std::endl;
        }        
        
    }
    config_file.close();
}



int main(int argc, char *argv[]) {
    //教師データの作成器を作成
    DataCreator Creator;
    //デフォルトのパラメータをセット
    Creator.set_all_parameters();

    //コマンドライン引数によるパラメータの指定
    if(argc > 1) {
        for(int i=1;i<argc;i++) {
            auto parameter_name = split(argv[i], "=").front();
            auto parameter_value = split(argv[i], "=").back();
            Creator.set_parameter(parameter_name, parameter_value);
        }
    }

    Creator.run_simulation();
    Creator.save_result();
    
    return 0;
}