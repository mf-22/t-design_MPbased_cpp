# t-design_MPbased_cpp
## 概要
t-designの識別のための教師データ生成と機械学習のプログラム、そして特定の量子ビット数・回路の深さのときのLocal Random Circuit, Random Diagonal Circuitが
どれぐらいのdesignになっているか推定するためのFramePotentialの計算プログラムです。  
C++とPythonで書かれており、C++のコンパイラはWindowsならMSVC++、Linux系ならGNUを想定しています。
PythonのほうはWindowsならAnaconda Python、Linux系ではvanilaなpythonを想定しています。

WindowsとLinuxで動くように作ったつもりですが、保証できません（データ生成は動くことを確認しましたが、機械学習・FPの方はまだ確認できていません）。  

データ生成、FPの計算のプログラムは[Qulacs](https://github.com/qulacs/qulacs)にかなり依存しています。

## Requirement
pythonのpip3とanacondaのrequirements.txtは後で追加します。  
基本的に[Qulacsのrequirement](https://github.com/qulacs/qulacs#requirement)を満たしていれば大丈夫なはずです。  
細かい部分は以下：  
- ディープラーニングのプログラム（ml/deep.py）
  - 作成したNeural Networkのモデルの図をpdfで保存するときに[tensorflow.keras.utils.plot_model](https://www.tensorflow.org/api_docs/python/tf/keras/utils/plot_model)を使っています
  （[ココ](https://github.com/mf-22/t-design_MPbased_cpp/blob/master/ml/deep.py#L106)）。
  このとき環境に応じてgraphvizが必要になるかもしれません。ダウンロードして設定するか、面倒だったらコメントアウトしてください。一応テキストファイルでモデルを保存するようにしてあります。
- FramePotentialの計算のプログラム（fp/calc_fp.cpp）
  - cloneしたままのコードだとC++17が必要です。ただ、計算結果を保存するためのディレクトリ作成でfilesystemを使っているだけなので、標準出力でOKなら問題ないです。

## Setup
Windowsの場合はトップにある`setup_win.bat`を、Linuxの場合は`setup_linux.sh`を実行すると必要なディレクトリの作成とQulacsのクローン・ビルドをします。  
ただ、データ生成のプログラムをコンパイルするときに最適化オプション(-O3等)を使うようMakefileに書いてあり、Qulacsでも同等の最適化オプションをつける必要がある場合があるかもしれません。
そのときは[QualcsのCMakeLists.txt](https://github.com/qulacs/qulacs/blob/master/CMakeLists.txt#L19)を編集しもう一度Qulacsをビルドしてください。Qulacsのビルド時にOMPをつけない
ようなときも同様に編集してください。  
その後はOSに応じて次の操作を行ってください：  
### Windows, Linux共通の変更
色々データ生成するときに、保存するフォルダ名が被らないようにデータ出力するときの現在時刻を取得しファイル名にするようにしています（例: 2021年8月5日21時50分20秒=>20210805215020）。  
この文字列を生成するコードをコンパイラに応じて変更する必要があります。`data_gen/include/t-design.util.hpp`の[getDatetimeStr関数](https://github.com/mf-22/t-design_MPbased_cpp/blob/master/data_gen/include/t-design_util.hpp#L161)と
`fp/calc_fp.cpp`の[getDatetimeStr関数](https://github.com/mf-22/t-design_MPbased_cpp/blob/master/fp/calc_fp.cpp#L84)内のコードを使用するコンパイラに応じてコメントアウトしてください。

### Windows環境の場合  
- データ生成のプログラム（data_gen/auto_crate.py）の設定
  - Visual Studio等を使って`main.cpp`をコンパイルし実行ファイル(\*\*\*.exe)を作成してください。その後、できた実行ファイルを`data_gen/main/`にコピーし、
  ファイル名を`msvc_project.exe`に変更するか`data_gen/main/auto_create.py`の[12行目](https://github.com/mf-22/t-design_MPbased_cpp/blob/master/data_gen/main/auto_create.py#L12)を
  ```
  exe_cpp = "***.exe" (実行ファイル名)
  ```
  に変更してください。また、Pythonの環境に応じてその次の行のpythonの実行コマンドも修正してください。

### Linux環境の場合
基本的にMakeFileのmakeコマンドを使ってビルドすると実行ファイル`main`が生成されます。そのmainファイルを実行すればOKです。もし`data_gen/main/main.cpp`の実行ファイル名が
`main`以外の場合は`data_gen/main/auto_create.py`の[16行目](https://github.com/mf-22/t-design_MPbased_cpp/blob/master/data_gen/main/auto_create.py#L16)を
```
exe_cpp = "./(実行ファイル名)"
```
に変更してください。

## プログラムの実行方法
### 教師データの生成（data_gen/）
データ生成プログラムは`auto_create.py`を実行する方法と、`random_clif.py`/`main.cpp`をそれぞれで実行するの２通りがあります。これらの違いは生成されたデータを特定のディレクトリ
に格納するかの違いです。データ生成をした後は機械学習を行いますが、機械学習のプログラムでは教師データを読み込むときに以下のディレクトリ構造を想定しています：
```
datasets --- dataset1 --- info.txt 
                       |
                       |- train --- haar_***.npy
                       |          ∟ clif_***.npy
                       |
                       |- valid --- haar_***.npy
                       |          ∟ clif_***.npy
                       |
         　             ∟ test  --- test1 --- haar_***.npy
                                 |          ∟ clif_***.npy
                                 |
                                 |- test2 --- haar_***.npy          
                                 |          ∟ clif_***.npy
                                 |
                   　             ∟ test3 --- lrc_***.npy
```
上記の例はHaar測度とClifford系のデータを使ってtraining/validiationを行い、テストデータはtest1,2がHaar測度とClifford系のミックス、test3がLocal Random Circuitのみでテストを行う
データセットです。`info.txt`はデータ生成時のパラメータが保存されたテキストファイルで機械学習時にパラメータを読みに行く必要がある場合はこのファイルを読みに行きます。（なお、
本来はそれぞれの一番下のディレクトリに各教師データ作成時のパラメータが保存されたテキストファイル`info_+++_***.npy`も作成されますが省いています）  
この`dataset1`以下のディレクトリ構造をauto_create.pyを実行すると自動で作成してくれます。作成場所は`data_gen/main/`の中なので`ml/datasets/`に移動してください。  
`random_clif.py`/`main.cpp`をそれぞれで実行した場合は`data_gen/result/`の中に教師データ`haar_***.npy`とその時のパラメータが保存されたテキストファイル`info_haar_***.txt`が
作成されるので、上記のデータ構造に沿って格納してもらえれば問題ありません。

### 機械学習の実行（ml/）
機械学習の実行は`ml/`の中のプログラムで実行したいアルゴリズムが書かれたファイルを実行してください。実行すると最初にデータセット名（上記の例では`dataset1`）の入力が求められ、
その次に計算するビット相関の数`k`、モーメントの次数`k'`を入力するように求められます。その後は自動的に学習・テストが行われ、その結果は
`ml/datasets/{データセット名}/{選択した機械学習アルゴリズム名}/{現在時刻}/`の中に保存されます。

### FramePotentailの計算（fp/）
`fp/calc_fp.cpp`をコンパイルし実行すればOKです。cloneしたままのコードであれば、`fp/result`の中にそれぞれのパラメータ(LRC/RDC, qubit数, 回路の深さ, t-designのtの値)のときの
計算結果が保存されます。
