"""
勾配ブースティング決定木を用いた分類モデル
LightGBM使用
"""

import pandas as pd
import numpy as np
import time
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score

import lightgbm as lgb


# ============================================================
# Setting
# ============================================================
RANDOM_STATE = 42

# Early Stopping
PATIENCE = 5       
PERIOD = 10 #表示頻度

LEARNING_RATE = 0.05
ESTIMATORS = 1000


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


# 訓練データと検証データに分割（8:2）
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y,
    test_size=0.2,
    random_state=42,
    stratify=Y
)


# # 標準化(決定木モデルは無意味)
# scaler = StandardScaler()
# X_train = scaler.fit_transform(X_train)
# X_test = scaler.transform(X_test)



# ============================================================
# Initialize Model
# ============================================================

model = lgb.LGBMClassifier(
    random_state=RANDOM_STATE,
    n_estimators=ESTIMATORS,
    learning_rate=LEARNING_RATE
)

print("\nModel Structure:")
print(model)


# ==========================================
# 4. 学習ループ (Early Stopping含む)
# ==========================================
start_time = time.perf_counter()

print("--- 学習開始 ---")

# Early Stopping とログ出力を設定
# eval_setの最後のデータセットに対して評価される。
callbacks = [
            lgb.early_stopping(stopping_rounds=PATIENCE, verbose=False),
            # lgb.log_evaluation(period=PERIOD)
            ]

model.fit(
          X_train, Y_train,
          eval_set=[(X_train, Y_train), (X_test, Y_test)],  # result出力させたいデータセット。0番目に訓練、1番目に検証
          eval_metric="multi_logloss",
          callbacks=callbacks
          )

print("--- 学習完了 ---")

elapsed_time = time.perf_counter() - start_time
print(f"処理時間: {elapsed_time:.2f}秒")


# ==========================================
# 5. 学習済みモデルの保存
# ==========================================
model_save_path = "./data/Dry_Bean_Dataset/dry_bean_lgbm_model.pkl"

joblib.dump(
    {
        "model": model,
        "input_dim": X.shape[1],
        "output_dim": OUTPUT_DIM,
        "classes": encoder.classes_
    },
    model_save_path
)

print(f"モデルを保存しました: {model_save_path}")


# ==========================================
# 6. 学習曲線の描画
# ==========================================
evals_result = model.evals_result_
print(evals_result.keys()) #キー項目名確認

fig, axes = plt.subplots(1, 1, figsize=(6, 4))

# Loss (Multi-Logloss)
axes.plot(
    evals_result['training']['multi_logloss'],
    label="Train Logloss"
)

axes.plot(
    evals_result['valid_1']['multi_logloss'],
    label="Validation Logloss"
)

axes.set_title("LightGBM Training & Validation Loss")
axes.set_xlabel("Boosting Iteration")
axes.set_ylabel("Loss")
axes.legend()
axes.grid()

plt.tight_layout()
plt.show()


# ==========================================
# 7. 検証データの予測・精度評価
# ==========================================
pred_np = model.predict(X_test)
accuracy = accuracy_score(Y_test, pred_np)

print(f"Validation Accuracy: {accuracy * 100:.2f}%")


# ==========================================
# 8. 予測結果の保存
# ==========================================
predicted_labels = encoder.inverse_transform(pred_np)

result_df = pd.DataFrame({
    "Actual": encoder.inverse_transform(Y_test),
    "Predicted": predicted_labels
})

result_df.to_csv(
    "./data/Dry_Bean_Dataset/predictions_lgbm.csv",
    index=False,
    encoding="utf-8-sig"
)

print(result_df.head())
print("予測結果を保存しました。")