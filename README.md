# GBDT Classification with LightGBM

## Overview

LightGBMを用いた勾配ブースティング決定木（Gradient Boosting Decision Tree: GBDT）による多クラス分類の実装です。

Dry Bean Datasetを使用し、豆の形状・寸法情報から品種を分類します。

基本的なLightGBMモデルの構築・学習に加え、Optunaによるハイパーパラメータ探索、およびStratified K-Foldによる交差検証を実装しています。

## Objectives

- LightGBMによる勾配ブースティング決定木の構築・学習
- GBDTの主要なハイパーパラメータの理解
- Early Stoppingによる学習の制御
- Optunaによるハイパーパラメータ最適化
- Stratified K-Foldによる交差検証を用いたモデル評価

## Dataset

### Dry Bean Dataset

乾燥豆の形状・寸法情報から、豆の品種を分類するデータセットです。

| Item | Description |
|---|---|
| Number of samples | 13,611 |
| Number of features | 16 |
| Number of classes | 7 |
| Feature type | Numerical |
| Target | Bean variety |

### Classification Classes

以下の7種類の豆を分類します。

- Seker
- Barbunya
- Bombay
- Cali
- Horoz
- Sira
- Dermason

### Features

豆の形状・寸法に関する16個の特徴量を使用します。

主な特徴量：

- Area
- Perimeter
- MajorAxisLength
- MinorAxisLength
- AspectRation
- Eccentricity
- ConvexArea
- EquivDiameter
- Extent
- Solidity
- roundness
- Compactness
- ShapeFactor1～4

## Implementations

### 1. Basic GBDT Classification

File: `GBDT_Classification.py`

LightGBMのClassifier APIを使用した多クラス分類モデルです。

主な実装内容：

- LightGBMモデルの構築
- 多クラス分類（multiclass）
- 学習率・木の本数・葉の数などのパラメータ設定
- 行・特徴量のサンプリング
- L1 / L2正則化
- Early Stopping
- 学習曲線の可視化
- 学習済みモデルの保存

### 2. Hyperparameter Optimization with Optuna and K-Fold

File: `GBDT_Classification_optuna_Kfold.py`

Optunaによるハイパーパラメータ探索にStratified K-Foldによる交差検証を組み合わせた実装です。

主な実装内容：

- Stratified K-Fold（5分割）
- 各FoldにおけるLightGBMモデルの学習・評価
- Early Stoppingによる最適なBoosting Roundの探索
- Foldごとの評価Loss・Accuracyの集計
- 交差検証結果に基づくハイパーパラメータ探索

### Hyperparameters

Optunaによる主な探索対象：

| Parameter | Description |
|---|---|
| learning_rate | 学習率 |
| num_leaves | 1つの木における葉の最大数 |
| max_depth | 木の最大深さ |
| min_child_samples | 葉に必要な最小データ数 |
| subsample | 学習データのサンプリング率 |
| colsample_bytree | 各木で使用する特徴量のサンプリング率 |
| reg_alpha | L1正則化係数 |
| reg_lambda | L2正則化係数 |

## Model Architecture

LightGBMのGradient Boosting Decision Treeを使用します。

複数の決定木を逐次的に構築し、前段のモデルで残った誤差を次の木で補正することで、予測性能を向上させるアンサンブル学習手法です。

本プロジェクトでは、LightGBMの多クラス分類機能を利用しています。

## Preprocessing

- 重複データの確認・削除
- Label Encodingによる目的変数の数値化
- 訓練データ・検証データの分割
- Stratified K-Foldによる交差検証

決定木ベースのモデルであるため、基本実装ではStandardScalerによる特徴量の標準化を行っていません。

## Training

- Framework：LightGBM
- Objective：Multiclass Classification
- Evaluation Metric：Multi Log Loss
- Early Stopping：Validation Lossに基づく学習停止
- Hyperparameter Optimization：Optuna
- Cross Validation：Stratified K-Fold

## Environment

- Python
- LightGBM
- Optuna
- scikit-learn
- pandas
- NumPy
- Matplotlib
- joblib

## How to Run

リポジトリをクローンします。

```bash
git clone https://github.com/Temp-Muraoka/GBDT_Classification.git
cd GBDT_Classification
```

必要なライブラリをインストールします。

```bash
pip install lightgbm optuna scikit-learn pandas numpy matplotlib joblib openpyxl
```

実行例：

```bash
python GBDT_Classification.py
python GBDT_Classification_optuna_Kfold.py
```

データセットは以下のディレクトリに配置してください。

```text
data/
└── Dry_Bean_Dataset/
    └── Dry_Bean_Dataset.xlsx
```

## Evaluation

モデルの評価には、主に以下の指標を使用します。

- Accuracy
- Multi Log Loss
- Training / Validation Loss

交差検証では、各Foldの評価結果を集計し、モデル性能を確認します。

## Results

学習結果については、以下を出力

- 学習曲線（Training / Validation Loss）
- Accuracy
- Optunaによる探索履歴


## Notes

本リポジトリは個人学習・自主開発を目的としたものです。

LightGBMの既存ライブラリを利用してモデルを構築し、データ前処理、学習条件設定、ハイパーパラメータ探索、交差検証、評価処理を実装しています。

なお、本リポジトリでは独立した最終テストデータによる汎化性能評価は実施していません。記載する評価結果は、検証データまたは交差検証による結果です。