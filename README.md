# Kiva／クラウドファンディングの資金調達額予測 3位解法


解法の詳細は[トピック](https://comp.probspace.com/competitions/kiva2021/discussions/bananabanana-Poste3e093d73691d7602ed9)に記載しています．  


### 必要ライブラリ
* python
* torch
* torchvision
* numpy
* pandas
* scikit-learn
* lightGBM
* seaborn
* pytorch-lightning
* gensim
* tensorflow
* tensorflow-text
* tensorflow_hub
* transformers
* [OpenAI CLIP](https://github.com/openai/CLIP)


## 準備

`data/raw/`にコンペのデータを置いてください．  
`train_images.zip`と`test_images.zip`は解凍してください．

```
.
├── data
│     ├── raw
│     │    ├── train_images
│     │    ├── test_images
│     │    ├── sample_submission.csv
│     │    ├── test.csv
│     │    └── train.csv
│     │

```

## 実行手順
以下の手順はすべて`easy_gold/`ディレクトリで行ってください．


### step.1 
* 前処理ファイルの作成

```bash
$ python preprocess.py -f  
```

* 希望金額関連の特徴作成
```
$ python kiva_utils.py -m loan
```
* CLIPによる画像特徴作成
```
$ python kiva_utils.py -m clip
```
* 類似文章のLOAN_AMOUNT特徴作成
```
$ python kiva_utils.py -m simAug
```
`data/proc/`に`df_proc_train.pkl`と`df_proc_test.pkl`とその他の特徴量ファイルが生成されます．


### step.2
* LightGBMによる学習と推論

```bash
$ python train.py -m lgb -ep 10000  -es 200 -lr 0.02 -e simAug -clip
```
`data/submission/`に`[実行時年月日-時分秒]_LGBWrapper_regr--[score]--_submission.csv`と`[実行時年月日-時分秒]_LGBWrapper_regr--[score]--_oof.csv`が生成されます．


### step.3
* テーブルデータ＋画像特徴のNeural Netによる学習と推論

```bash
$ python train.py -m nn -ep 10 -es 10 -lr 0.01 -e table -clip
```
`data/submission/`に`[実行時年月日-時分秒]_BERT_Wrapper--[score]--_submission.csv`と`[実行時年月日-時分秒]_BERT_Wrapper--[score]--_oof.csv`が生成されます．


### step.4
* BARTによる学習と推論

```bash
$ python train.py -m nn -model bart -ep 60 -es 10 -lr 0.001 
```
`data/submission/`に`[実行時年月日-時分秒]_BERT_Wrapper--[score]--_submission.csv`と`[実行時年月日-時分秒]_BERT_Wrapper--[score]--_oof.csv`が生成されます．


### step.5
* BERT(classification)による学習と推論

```bash
$ python train.py -m nn -model bert -t classification -ep 30 -es 10 -lr 0.01 -e simAug
```
`data/submission/`に`[実行時年月日-時分秒]_BERT_Wrapper--[score]--_submission.csv`と`[実行時年月日-時分秒]_BERT_Wrapper--[score]--_oof.csv`が生成されます．


### step.6
* Distlibertによる学習と推論

```bash
$ python train.py -m nn -model distilbert -ep 10 -es 10 -lr 0.01 -e simAug -clip
```
`data/submission/`に`[実行時年月日-時分秒]_BERT_Wrapper--[score]--_submission.csv`と`[実行時年月日-時分秒]_BERT_Wrapper--[score]--_oof.csv`が生成されます．


### step.7
* Averaging

```bash
$ mkdir ../data/submission/ave_dir
$ cp ../data/submission/*.csv ../data/submission/ave_dir
```
`data/submission/`の中に`ave_dir`という名前のディレクトリを作成し，step2~6で生成されたcsvをすべてコピーします．


```bash
$ python train.py -m ave -stack_dir ave_dir 
```

`data/submission/`に提出ファイルである`[実行時年月日-時分秒]_Averaging_Wrapper--[score]--_submission.csv`が生成されます．