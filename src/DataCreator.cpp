#include "DataCreator.hpp"

#include <map>
#include <vector>
#include <fstream>
#include <iostream>
#include <cppsim/state.hpp>
#include <cppsim/state_dm.hpp>
#include <cppsim/circuit.hpp>
#include <cppsim/gate_factory.hpp>
#include <cppsim/gate_general.hpp>
#include <cppsim/gate_merge.hpp>
#include <cppsim/gate_matrix.hpp>


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

void DataCreator::read_configFile() {
    //入力ファイルの内容を順番に反映していく
    auto input_paras = _read_inputfile();
    for(auto itr = input_paras.begin(); itr != input_paras.end(); ++itr) {
        set_parameter(itr->first, itr->second);
    }
}

void DataCreator::set_parameter(std::string key, std::string val) {
    //引数のkeyとvalueをもとにパラメータ(辞書型)の値を更新する
    //keyが存在するときのみ更新し、存在しないときはエラーとする(新しく追加しない)
    if(this->parameters.find(key) != this->parameters.end()) {
        this->parameters[key] = val;
    } else{
        std::cerr << "Key error: " << key << std::endl;
    }
}

void DataCreator::_haar_sim() {
    //各ユニタリにおけるqubitの測定確率を保持するベクトル
    //std::vector<std::vector<float>> MP_list(this->Nu, std::vector<float>(this->comb_list.size()));
    //測定結果を保存するベクトル
    std::vector<ITYPE> sampling_result;

    //ループカウンタ
    int i,j;
    
    //教師データ作成
    #pragma omp parallel for private(j, sampling_result)
    for(i=0;i<this->S;++i) {
        std::vector<std::vector<float>> MP_list(this->Nu, std::vector<float>(this->comb_list.size()));
        for(j=0;j<this->Nu;++j) {
            //量子状態の生成と初期化
            QuantumState state(this->Nq);
            state.set_Haar_random_state();
            //測定と測定確率(ビット相関も)の計算
            sampling_result = state.sampling(this->Ns);
            MP_list[j] = _calc_BitCorr_and_MP(sampling_result);
        }
        //測定確率のモーメントを計算
        this->teacher_data[i] = _calc_moment_of_MP(MP_list);
        std::cout << "\r" << i+1 << "/" << this->S << " finished..." << std::string(20, ' ');
    }
    std::cout << std::endl;
}

void DataCreator::_lrc_sim(
    std::vector<std::vector<std::vector<unsigned int>>>& RU_index_list) {

    //各ユニタリにおけるqubitの測定確率を保持するベクトル
    //std::vector<std::vector<float>> MP_list(this->Nu, std::vector<float>(this->comb_list.size()));
    //測定結果を保存するベクトル
    std::vector<ITYPE> sampling_result;

    //ループカウンタ
    int i,j,l;
    //教師データ作成
    #pragma omp parallel for private(j, l, sampling_result)
    for(i=0;i<this->S;++i) {
        std::vector<std::vector<float>> MP_list(this->Nu, std::vector<float>(this->comb_list.size()));
        for(j=0;j<this->Nu;++j) {
            //量子状態の作成と初期化
            QuantumState state(this->Nq);
            //state.set_zero_state();
            //LRCの実行
            for(l=1;l<this->depth+1;++l) {
                for(const auto& qubit_indecies : RU_index_list[l%2]) {
                    gate::RandomUnitary(qubit_indecies)->update_quantum_state(&state);
                }    
            }
            //測定と測定確率(ビット相関も)の計算
            sampling_result = state.sampling(this->Ns);
            MP_list[j] = _calc_BitCorr_and_MP(sampling_result);
        }
        //測定確率のモーメントを計算
        this->teacher_data[i] = _calc_moment_of_MP(MP_list);
        std::cout << "\r" << i+1 << "/" << this->S << " finished..." << std::string(20, ' ');
    }
    std::cout << std::endl;
}

void DataCreator::_lrc_depolarizing_sim(
    std::vector<std::vector<std::vector<unsigned int>>>& RU_index_list) {
    
    //各ユニタリにおけるqubitの測定確率を保持するベクトル
    //std::vector<std::vector<float>> MP_list(this->Nu, std::vector<float>(this->comb_list.size()));
    //測定結果を保存するベクトル
    std::vector<ITYPE> sampling_result;

    //ノイズモデルの定義
    //各qubitにかかるCPTPを事前に回路として作っておき各層で適用する
    QuantumCircuit depolarizing_circuit(this->Nq);
    //回路の作成
    for(int i=0;i<this->Nq;++i) {
        depolarizing_circuit.add_gate(gate::DepolarizingNoise(i, this->noise_prob));
    }

    //ループカウンタ
    int i,j,l;
    //教師データ作成
    #pragma omp parallel for private(j, l, sampling_result)
    for(i=0;i<this->S;++i) {
        std::vector<std::vector<float>> MP_list(this->Nu, std::vector<float>(this->comb_list.size()));
        for(j=0;j<this->Nu;++j) {
            //量子状態(密度行列)の生成と初期化
            DensityMatrix state(this->Nq);
            //state.set_zero_state();
            //ノイズありのLRCの実行
            for(l=1;l<this->depth+1;++l) {
                for(const auto& qubit_indecies : RU_index_list[l%2]) {
                    gate::RandomUnitary(qubit_indecies)->update_quantum_state(&state);
                }
                //ノイズの適用
                depolarizing_circuit.update_quantum_state(&state);
            }
            //測定と測定確率(ビット相関も)の計算
            sampling_result = state.sampling(this->Ns);
            MP_list[j] = _calc_BitCorr_and_MP(sampling_result);
        }
        //測定確率のモーメントを計算
        this->teacher_data[i] = _calc_moment_of_MP(MP_list);
        std::cout << "\r" << i+1 << "/" << this->S << " finished..." << std::string(20, ' ');
    }
    std::cout << std::endl;
}

void DataCreator::_lrc_MeasurementInduced_sim(
    std::vector<std::vector<std::vector<unsigned int>>>& RU_index_list) {
    
    //各ユニタリにおけるqubitの測定確率を保持するベクトル
    //std::vector<std::vector<float>> MP_list(this->Nu, std::vector<float>(this->comb_list.size()));
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
    for(int i=0;i<this->Nq;++i) {
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
    for(i=0;i<this->S;++i) {
        std::vector<std::vector<float>> MP_list(this->Nu, std::vector<float>(this->comb_list.size()));
        for(j=0;j<this->Nu;++j) {
            //量子状態の生成と初期化
            QuantumState state(this->Nq);
            //state.set_zero_state();
            //LRCの実行
            for(l=1;l<this->depth+1;++l) {
                for(const auto& qubit_indecies : RU_index_list[l%2]) {
                    gate::RandomUnitary(qubit_indecies)->update_quantum_state(&state);
                }
                //測定の適用
                p_measure_circuit.update_quantum_state(&state);
            }
            //2層のLRCの追加
            for(l=1;l>-1;--l) {
                //l=1(奇数層)とl=0(偶数層)で実行
                for(const auto& qubit_indecies : RU_index_list[l]) {
                    gate::RandomUnitary(qubit_indecies)->update_quantum_state(&state);
                }
            }
            //量子状態の正規化
            state.normalize(state.get_squared_norm());
            //測定と測定確率(ビット相関も)の計算
            sampling_result = state.sampling(this->Ns);
            MP_list[j] = _calc_BitCorr_and_MP(sampling_result);
        }
        //測定確率のモーメントを計算
        this->teacher_data[i] = _calc_moment_of_MP(MP_list);
        std::cout << "\r" << i+1 << "/" << this->S << " finished..." << std::string(20, ' ');
    }
    std::cout << std::endl;
}

void DataCreator::_rdc_sim() {
    //各ユニタリにおけるqubitの測定確率を保持するベクトル
    //std::vector<std::vector<float>> MP_list(this->Nu, std::vector<float>(this->comb_list.size()));
    //測定結果を保存するベクトル
    std::vector<ITYPE> sampling_result;
    
    //基底を変更する、全てのqubitにHadmardを適用する回路
    QuantumCircuit basis_change_circuit(this->Nq);
    for(int i=0;i<this->Nq;++i) {
        basis_change_circuit.add_H_gate(i);
    }

    //ループカウンタ
    int i,j,d;
    //教師データ作成
    #pragma omp parallel for private(j, d, sampling_result)
    for(i=0;i<this->S;++i) {
        std::vector<std::vector<float>> MP_list(this->Nu, std::vector<float>(this->comb_list.size()));
        for(j=0;j<this->Nu;++j) {
            //量子状態の生成と初期化
            QuantumState state(this->Nq);
            //state.set_zero_state();
            //RDCの実行
            for(d=0;d<this->depth;++d) {
                state.multiply_elementwise_function(gen_RandomDiagonal_element);
                basis_change_circuit.update_quantum_state(&state);
            }
            //測定と測定確率(ビット相関も)の計算
            sampling_result = state.sampling(this->Ns);
            MP_list[j] = _calc_BitCorr_and_MP(sampling_result);
        }
        //測定確率のモーメントを計算
        this->teacher_data[i] = _calc_moment_of_MP(MP_list);
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
    //this->seed = std::stoi(this->parameters["seed"]);    
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
    this->teacher_data.clear();
    this->teacher_data.resize(this->S, std::vector<float>(this->comb_list.size()));

    //2進数のリストを作成
    this->binary_num_list.clear();
    int num_decimal;
    for (int i = 0; i < pow(2, this->Nq); ++i) {
        //10進数の値
        num_decimal = i;
        //2進数に変換された値。vectorに保存される。
        //-1で初期化して置いて、ビットの1が立つところだけを書き換えれば0=>-1の変換になる
        std::vector<int> bit_Nq_size(this->Nq, -1);
        if (num_decimal != 0) {
            for (int j = 0; j < log2(i) + 1; ++j) {
                if (num_decimal % 2 == 1) {
                    bit_Nq_size[this->Nq - 1 - j] = 1;
                }
                num_decimal /= 2;
            }
        }
        this->binary_num_list.emplace_back(bit_Nq_size);
    }
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
        for(int i=1;i<Nq-1;++i) {
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
        for(int i=0;i<maxIndex_depthOdd;++i) {
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

    //各測定結果に対してビット相関を計算するので、shots回分のビット相関の値が出てくる。
    //これを足しこんでいき、最後に測定回数で割れば測定確率(期待値=:ビット相関の値)となる。
    float sum_bitcorr;
    int bitcorr_oneshot;

    for (const auto& bit_index_list : this->comb_list) {
        //ビット相関を計算する位置を取得
        sum_bitcorr = 0.0;
        for (const auto& result : sampling_result) {
            bitcorr_oneshot = 1;
            for (const auto& qubit_index : bit_index_list) {
                bitcorr_oneshot *= this->binary_num_list[result][qubit_index];
            }
            sum_bitcorr += bitcorr_oneshot;
        }
        MP_list.emplace_back(sum_bitcorr / this->Ns);
    }

    return MP_list;
}

std::vector<float> DataCreator::_calc_moment_of_MP(std::vector<std::vector<float>>& MP_data) {
    int Nq_prime = MP_data[0].size();

    std::vector<float> moments_of_MP;
    moments_of_MP.reserve(Nq_prime*20);

    std::vector<float> MP_mom_eachDim(Nq_prime, 0.0);

    for(int dim=1;dim<21;++dim) {
        for(const auto& MP_eachU : MP_data) {
            for(int j=0;j<Nq_prime;++j) {
                MP_mom_eachDim[j] += pow(MP_eachU[j], dim);
            }
        }
        for(auto&& each_MP_mom : MP_mom_eachDim) {
            moments_of_MP.emplace_back(each_MP_mom / this->Nu);
            each_MP_mom = 0.0;
        }
    }
    
    return moments_of_MP;
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
    for(const auto& each_S : this->teacher_data) {
        for(auto itr=each_S.begin(); itr!=each_S.end()-1; ++itr) {
            csv_file << *itr << ",";
        }
        csv_file << each_S.back() << "\n";
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
    //config_file << " seed : " << this->seed << std::endl;
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
    config_file << " bit corrlation : " << this->Nq << std::endl;
    config_file << " dim of moments : 1~20 " << std::endl;
    config_file.close();
}