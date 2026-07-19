import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    precision_recall_curve,
    auc
)
from catboost import CatBoostClassifier

# =========================
# 1. LOAD DATA
# =========================
df = pd.read_csv("dataset.csv")
print("Original dataset shape:", df.shape)

# =========================
# 2. DATETIME FEATURES
# =========================
df["timestamp"] = pd.to_datetime(
    df["Date"] + " " + df["Time"],
    format="%Y-%m-%d %H:%M:%S"
)

df["hour"] = df["timestamp"].dt.hour
df["day"] = df["timestamp"].dt.day
df["month"] = df["timestamp"].dt.month
df["weekday"] = df["timestamp"].dt.weekday

df.drop(columns=["Time", "Date", "timestamp", "Laundering_type"], inplace=True)

# =========================
# 3. STRATIFIED SAMPLING (SAFE WAY)
# =========================
fraud_df = df[df["Is_laundering"] == 1]
normal_df = df[df["Is_laundering"] == 0]

fraud_sample = fraud_df.sample(frac=0.2, random_state=42)
normal_sample = normal_df.sample(frac=0.2, random_state=42)

df = pd.concat([fraud_sample, normal_sample]).sample(frac=1, random_state=42)

print("Sampled dataset shape:", df.shape)

# =========================
# 4. FEATURES & TARGET
# =========================
target = "Is_laundering"
X = df.drop(columns=[target])
y = df[target]

categorical_features = [
    "Sender_account",
    "Receiver_account",
    "Payment_currency",
    "Received_currency",
    "Sender_bank_location",
    "Receiver_bank_location",
    "Payment_type"
]

cat_features_index = [X.columns.get_loc(col) for col in categorical_features]

# =========================
# 5. TRAIN–TEST SPLIT
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

print("Train fraud ratio:", y_train.mean())
print("Test fraud ratio:", y_test.mean())

# =========================
# 6. CATBOOST MODEL
# =========================
fraud_ratio = y_train.mean()
normal_ratio = 1 - fraud_ratio

class_weights = [1, int(normal_ratio / fraud_ratio * 2)]

model = CatBoostClassifier(
    iterations=500,
    learning_rate=0.03,
    depth=8,
    loss_function="Logloss",
    eval_metric="AUC",
    class_weights=class_weights,
    cat_features=cat_features_index,
    random_seed=42,
    verbose=50
)

model.fit(
    X_train,
    y_train,
    eval_set=(X_test, y_test),
    use_best_model=True
)

# =========================
# 7. PROBABILITY PREDICTION
# =========================
y_prob = model.predict_proba(X_test)[:, 1]

# =========================
# 8. THRESHOLD TUNING
# =========================
thresholds = np.linspace(0.001, 0.2, 200)

best_f1 = 0
best_threshold = 0.01

for t in thresholds:
    y_pred_temp = (y_prob >= t).astype(int)
    report = classification_report(
        y_test,
        y_pred_temp,
        output_dict=True,
        zero_division=0
    )
    f1 = report["1"]["f1-score"]
    if f1 > best_f1:
        best_f1 = f1
        best_threshold = t

print(f"\nBest threshold selected: {best_threshold:.4f}")
print(f"Best fraud F1-score: {best_f1:.4f}")

y_pred = (y_prob >= best_threshold).astype(int)

# =========================
# 9. FINAL EVALUATION
# =========================
print("\nFinal Classification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

roc_auc = roc_auc_score(y_test, y_prob)
print("ROC-AUC Score:", roc_auc)

precision, recall, _ = precision_recall_curve(y_test, y_prob)
pr_auc = auc(recall, precision)
print("Precision–Recall AUC:", pr_auc)

# =========================
# 10. SAVE MODEL
# =========================
model.save_model("aml_fraud_catboost_practical.cbm")
print("Model saved as aml_fraud_catboost_practical.cbm")
