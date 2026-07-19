import pandas as pd
from catboost import CatBoostClassifier, Pool

MODEL_PATH = "aml_fraud_catboost_practical.cbm"
THRESHOLD = 0.20

model = CatBoostClassifier()
model.load_model(MODEL_PATH)

FEATURE_COLUMNS = [
    "Sender_account",
    "Receiver_account",
    "Payment_currency",
    "Received_currency",
    "Sender_bank_location",
    "Receiver_bank_location",
    "Payment_type",
    "Amount",
    "hour",
    "day",
    "month",
    "weekday"
]

CATEGORICAL_FEATURES = [
    "Sender_account",
    "Receiver_account",
    "Payment_currency",
    "Received_currency",
    "Sender_bank_location",
    "Receiver_bank_location",
    "Payment_type"
]

cat_features_index = [FEATURE_COLUMNS.index(col) for col in CATEGORICAL_FEATURES]

transaction = {
    "Sender_account": "7421451752",
    "Receiver_account": "2755709071",
    "Payment_currency": "Indian rupee",
    "Received_currency": "UK pounds",
    "Sender_bank_location": "UK",
    "Receiver_bank_location": "UK",
    "Payment_type": "Credit card",
    "Amount": 5883.87,
    "Date": "10/7/2022",
    "Time": "10:35:29"
}

timestamp = pd.to_datetime(
    transaction["Date"] + " " + transaction["Time"],
    dayfirst=True,
    errors="raise"
)

transaction["hour"] = timestamp.hour
transaction["day"] = timestamp.day
transaction["month"] = timestamp.month
transaction["weekday"] = timestamp.weekday()

transaction.pop("Date")
transaction.pop("Time")

X_input = pd.DataFrame([transaction], columns=FEATURE_COLUMNS)

for col in CATEGORICAL_FEATURES:
    X_input[col] = X_input[col].astype(str)

input_pool = Pool(
    data=X_input,
    cat_features=cat_features_index
)

prob_fraud = model.predict_proba(input_pool)[0][1]
prediction = int(prob_fraud >= THRESHOLD)

print("\n========== AML FRAUD PREDICTION ==========")
print(f"Fraud probability : {prob_fraud:.6f}")
print(f"Decision threshold: {THRESHOLD}")
print("Prediction        :", "FRAUD" if prediction else "LEGIT")
print("=========================================")
