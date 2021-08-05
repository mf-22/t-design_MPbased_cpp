#include "DataCreator.hpp"

int main(int argc, char *argv[]) {
    //教師データの作成器を作成
    DataCreator Creator;
    //設定ファイルからパラメータを読み込み
    Creator.read_configFile();

    //コマンドライン引数によるパラメータの指定
    if(argc > 1) {
        for(int i=1;i<argc;i++) {
            auto parameter_name = split(argv[i], "=").front();
            auto parameter_value = split(argv[i], "=").back();
            Creator.set_parameter(parameter_name, parameter_value);
        }
    }
    //データの作成開始
    Creator.run_simulation();
    //結果の保存
    Creator.save_result();
    
    return 0;
}