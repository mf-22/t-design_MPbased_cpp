#pragma once

#include <vector>
#include <map>
#include "t-design_util.hpp"

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
        this->parameters["seed"] = "8010";
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