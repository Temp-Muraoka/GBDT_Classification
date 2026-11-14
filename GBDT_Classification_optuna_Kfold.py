"""
Optunaを用いたハイパーパラメータチューニング
NN_Classification_pytorch.pyのモデルを対象

交差検証法

"""

import pandas as pd
import numpy as np
import time
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import StratifiedKFold
import lightgbm as lgb

import optuna
import os


# ============================================================
# Setting
# ============================================================
TRIAL_N = 20 #探索回数
N_SPLITS = 5    # 交差検証の分割数

# HIDDEN_LAYERS = [32, 16]

# LEARNING_RATE = 0.001

# EPOCHS = 100 #Earlystoppingするので、十分大きい数

# DROPOUT_RATE = 0.0

# Early Stopping
PATIENCE = 15       # 検証Lossが改善しないことを許容するエポック数




# # =====================================================================
# # デバイス設定と Google Drive 保存先の設定
# # =====================================================================
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"使用中の実行デバイス: {device}")

# # Google Drive 上の保存用ディレクトリを作成
# save_dir = "/content/drive//MyDrive/Colab Notebooks/・・・"
# os.makedirs(save_dir, exist_ok=True)

# # データベース（.db）と CSV の保存パスを設定
# db_path = os.path.join(save_dir, "optuna_study.db")

csv_path = "./data/Dry_Bean_Dataset/trials_history.csv"



# ==========================================
# データ読み込み・前処理
# ==========================================
# Excel形式で保存されている場合
df = pd.read_excel("./data/Dry_Bean_Dataset/Dry_Bean_Dataset.xlsx")

# 削除対象となる重複行を抽出（デフォルトでは最初の1件を残し、2件目以降がTrueになる）
duplicates = df[df.duplicated()]

# 重複データが存在する場合のみ中身と件数をprint表示
if len(duplicates) > 0:
    print(f"削除対象の重複データ {len(duplicates)} 件を削除します")
    # print(duplicates)
else:
    print("重複データはありませんでした。")

# 重複を削除して更新
df = df.drop_duplicates()


X = df.drop(columns=["Class"])
Y = df["Class"]

# 目的変数を数値化（0～6）
encoder = LabelEncoder()
Y = encoder.fit_transform(Y)

# クラス数
OUTPUT_DIM = len(encoder.classes_)



# NumPy配列に変換
X = X.values.astype(np.float32)
Y = Y.astype(np.int64)


print(f"Number of samples: {len(X)}")
print(f"Number of features: {X.shape[1]}")
print("Classes:", encoder.classes_)
print("Number of classes:", OUTPUT_DIM)


# ==========================================
# optunaによるハイパーパラメータ最適化（交差検証）
# ==========================================


start_time = time.perf_counter()

print("--- ハイパーパラメータ探索開始 (Cross-Validation) ---")

skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

def objective(trial):

    # --------------------------------------
    # 探索したいハイパーパラメータの設定 (LightGBM用)
    # --------------------------------------
    params = {
        'objective': 'multiclass',
        'num_class': OUTPUT_DIM,
        'metric': 'multi_logloss',
        'boosting_type': 'gbdt',
        'random_state': 42,
        'verbose': -1,
        
        # チューニング対象パラメータ
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 15, 127),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'min_child_samples': trial.suggest_int('min_child_samples', 10, 100),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True),
    }

    # 各フォールドのスコアを格納するリスト
    fold_valid_losses = []
    fold_train_losses = []
    fold_valid_accs = []
    fold_train_accs = []
    fold_best_rounds = []

    # 交差検証ループ
    for fold, (train_idx, valid_idx) in enumerate(skf.split(X, Y)):

        X_tr, X_va = X[train_idx], X[valid_idx]
        Y_tr, Y_va = Y[train_idx], Y[valid_idx]

        # 決定木系モデルのため、StandardScaler等のスケーリングは不要（そのまま渡す）

        # LightGBM Datasetの作成
        lgb_train = lgb.Dataset(X_tr, label=Y_tr)
        lgb_valid = lgb.Dataset(X_va, label=Y_va, reference=lgb_train)

        # ==========================================
        # 学習 (LightGBM train API + Early Stopping)
        # ==========================================
        evals_result = {}
        
        callbacks = [
            lgb.early_stopping(stopping_rounds=PATIENCE, verbose=False),
            lgb.log_evaluation(period=0), # 学習中の標準出力を抑制
            lgb.record_evaluation(evals_result) # 記録用コールバックを追加
        ]

        model = lgb.train(
            params,
            lgb_train,
            num_boost_round=1000, # 十分大きい最大木の本数
            valid_sets=[lgb_train, lgb_valid],
            valid_names=['train', 'valid'],
            callbacks=callbacks,
        )

        # ベストイテレーション（何本目で止まったか）
        best_iter = model.best_iteration
        if best_iter == 0:
            best_iter = model.current_iteration()

        # 各データセットに対する予測とメトリクス計算
        train_preds_proba = model.predict(X_tr, num_iteration=best_iter)
        valid_preds_proba = model.predict(X_va, num_iteration=best_iter)

        train_preds_cls = train_preds_proba.argmax(axis=1)
        valid_preds_cls = valid_preds_proba.argmax(axis=1)

        # 損失と正解率
        train_loss = log_loss(Y_tr, train_preds_proba, labels=np.arange(OUTPUT_DIM))
        valid_loss = log_loss(Y_va, valid_preds_proba, labels=np.arange(OUTPUT_DIM))
        
        train_acc = accuracy_score(Y_tr, train_preds_cls)
        valid_acc = accuracy_score(Y_va, valid_preds_cls)

        # 進捗の動的上書き表示
        print(f" Trial {trial.number+1} | {fold + 1}/{N_SPLITS} fold finished (Best Iter: {best_iter})", end="\r", flush=True)

        # 各フォールドの結果を保存
        fold_valid_losses.append(valid_loss)
        fold_train_losses.append(train_loss)
        fold_valid_accs.append(valid_acc)
        fold_train_accs.append(train_acc)
        fold_best_rounds.append(best_iter)

    # 全フォールドの平均を計算
    mean_valid_loss = np.mean(fold_valid_losses)
    mean_train_loss = np.mean(fold_train_losses)
    mean_valid_acc = np.mean(fold_valid_accs)
    mean_train_acc = np.mean(fold_train_accs)
    mean_best_round = np.mean(fold_best_rounds)

    # 各トライアルの最後に、全フォールド平均でのLossとAccを表示
    print(
        f"Trial {trial.number+1} CV Mean [Avg Best Iter: {mean_best_round:.1f}] "
        f"Train Loss: {mean_train_loss:.4f} | Valid Loss: {mean_valid_loss:.4f} | "
        f"Train Acc: {mean_train_acc:.4f} | Valid Acc: {mean_valid_acc:.4f}"
    )

    # UserAttrs へ保存
    trial.set_user_attr("best_epoch", mean_best_round) # 互換性のためキー名を合わせる
    trial.set_user_attr("best_train_loss", mean_train_loss)
    trial.set_user_attr("best_train_acc", mean_train_acc)
    trial.set_user_attr("best_valid_acc", mean_valid_acc)

    return mean_valid_loss

study = optuna.create_study(
    study_name="subrutin_lgbm_tuning",
    # storage=f"sqlite:///{db_path}",  # Google Drive 上の SQLite DB を使う場合はコメント解除
    # load_if_exists=True,             # ★重要: DBが存在すれば自動で続きから再開
    direction="minimize",            # 目的関数（Loss）を最小化
    pruner=optuna.pruners.MedianPruner()  # 枝刈り用
)

print(f"現在完了している Trial 数: {len(study.trials)}")

# 探索の実行
study.optimize(objective, n_trials=TRIAL_N)

# 完了後に結果を CSV としても出力
df_trials = study.trials_dataframe()
df_trials.to_csv(csv_path, index=False, encoding="utf-8-sig")

print("\nハイパーパラメータチューニング完了！")
print(f"ベスト評価値: {study.best_value:.4f}")
print("ベストパラメータ:")
for key, val in study.best_params.items():
    print(f"  - {key}: {val}")
print(f"履歴保存先: {csv_path}")

# --- 最も良かったTrialから情報を取得 ---
best_trial = study.best_trial

best_epoch = best_trial.user_attrs.get("best_epoch")
best_train_loss = best_trial.user_attrs.get("best_train_loss")
best_valid_acc = best_trial.user_attrs.get("best_valid_acc")

print("\nハイパーパラメータチューニング完了！")
print(f"ベストTrial番号: {best_trial.number}")
print(f"ベストイテレーション（平均）: {best_epoch:.1f} ラウンド")
print(f"ベスト検証 Loss (Valid Loss): {best_trial.value:.4f}")
print(f"ベスト訓練 Loss: {best_train_loss:.4f}")
print(f"ベスト検証 正解率 (Valid Acc): {best_valid_acc:.4f}")
print("ベストパラメータ:")
for key, val in best_trial.params.items():
    print(f"  - {key}: {val}")

elapsed_time = time.perf_counter() - start_time
print(f"処理時間: {elapsed_time:.2f}秒")


# # ==========================================
# # 最も良いパラメータでのモデル構築・再学習と保存
# # ==========================================
# print("\n--- ベストパラメータによるモデルの再学習と保存 ---")

# best_params = study.best_params
# # 再学習用の設定パラメータを構築
# final_params = {
#     'objective': 'multiclass',
#     'num_class': OUTPUT_DIM,
#     'metric': 'multi_logloss',
#     'boosting_type': 'gbdt',
#     'random_state': 42,
#     'verbose': -1,
#     **best_params
# }

# # 全データを用いて、クロスバリデーションの平均最適ラウンド数（または少し余裕を持たせた回数）で再学習
# # あるいは全データに対するtrain/val分割で厳密にEarly Stoppingを行うことも可能です
# best_best_iter = int(best_epoch) if not np.isnan(best_epoch) else 100

# train_data_full = lgb.Dataset(X, label=Y)
# best_model = lgb.train(
#     final_params,
#     train_data_full,
#     num_boost_round=best_best_iter
# )

# # モデルとクラス情報・エンコーダをまとめて joblib で保存
# model_save_path = "./data/Dry_Bean_Dataset/dry_bean_lgbm_model.pkl"
# joblib.dump(
#     {
#         "model": best_model,
#         "classes": encoder.classes_,
#         "params": final_params
#     },
#     model_save_path
# )

# print(f"ベストモデルを保存しました: {model_save_path}")

# elapsed_time = time.perf_counter() - start_time
# print(f"総処理時間: {elapsed_time:.2f}秒")