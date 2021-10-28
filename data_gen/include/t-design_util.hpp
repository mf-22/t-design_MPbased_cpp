#pragma once

#define _USE_MATH_DEFINES
#include <cmath>
#include <cppsim/type.hpp>
#include <cppsim/utility.hpp>
#include <vector>
#include <numeric>
#include <iomanip>
#include <math.h>

/** ランダム対角ユニタリ行列の対角成分を生成する関数
 * qulacsのmultiply_elementwise_functionに渡す
 */
inline std::complex<double> gen_RandomDiagonal_element(ITYPE dummy) {
    //calculate exp( (0,2pi] * i )
    Random random;
    return std::exp(std::complex<double>(0.0, random.uniform() * 2 * M_PI));
}

/** 組み合わせ(nCr)を作るために利用する関数
 */
inline void recursive_set(int n, int r, std::vector<std::vector<int>>& v, std::vector<int>& pattern) {
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
        int pop_thre = n - 1;

        //patternの最後尾をインクリメントしていき組み合わせのリストvにpushしていく
        for (int i=pattern[r-1];i<n;++i) {
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

/** 組み合わせ(nCr)を作成する関数
 * nとrを指定し2次元のリストを渡すと、そのリストに全通りの組み合わせを入れて返す
 */
inline void create_comb_list(int n, int r, std::vector<std::vector<int>>& comb_list) {
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
        for(int i=0;i<n;++i) {
            comb_list.push_back(p_list);
            p_list[0] = p_list[0] + 1;
        }
    }
    //n点相関(例.5qubit5点相関)のとき => 単に0からnまでの1次元のリストを返す
    else if (n == r) {
        std::vector<int> p_list;
        for(int i=0;i<n;++i) {
            p_list.push_back(i);
        }
        comb_list.push_back(p_list);
    }
    //上記以外のk点相関のとき
    else {
        std::vector<int> p_list;
        for(int i=0;i<r;++i) {
            p_list.push_back(i);
        }
        while (p_list.size() > 0) {
            recursive_set(n, r, comb_list, p_list);
        }
    }
}

/** 組み合わせ(nCr)について、nを指定したときにr=1から
 *  r=nまでの全ての組み合わせを生成する関数
 */
inline std::vector<std::vector<int>> get_possibliyMax_bitCorr(unsigned int num_qubits) {
    std::vector<std::vector<int>> bitCorr_list;
    
    for(int i=1;i<num_qubits+1;++i) {
        create_comb_list(num_qubits, i, bitCorr_list);
    }
    
    return bitCorr_list;
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