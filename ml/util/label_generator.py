import numpy as np

def generate_label(data_type, size):
    """ データの種類に応じてラベルを作成する
        Haarは0、それ以外は1を作る
        Args:
            data_type := データの種類を指定する文字列(haar, clif, ...)
            size      := 作成するラベルのサイズ(int)
        Return:
            label := 0か1を要素に持つsize個の1次元ndarray
    """
    if data_type == "haar":
        label = np.zeros(size, dtype=np.int8)
    elif data_type == "clif" or data_type == "lrc" or data_type == "rdc":
        label = np.ones(size, dtype=np.int8)
    else:
        print("**CAUTION** Detect undefined data_type :", data_type)
        print("label 1 is created.")
        label = np.ones(size, dtype=np.int8)
    
    return label

def scaler_to_label(output):
    """ 線形回帰やNNの識別器でmodel.predict()をしたときの出力がラベルではなく
        スカラーのときに、そのスカラーからラベル{0,1}に変換する関数
        Arg:
            output := model.predict()の出力
        Return:
            ndarray := ラベルの0か1を要素に持つ1次元のリスト
    """
    predicted_label = [0 if i < 0.5 else 1 for i in output.flatten()]
    return np.array(predicted_label)