from flask import Flask, render_template, request
import pandas as pd
from catboost import CatBoostClassifier, Pool

app = Flask(__name__)

MODEL_PATH = "aml_fraud_catboost_practical.cbm"
THRESHOLD = 0.20
DATASET_PATH = "dataset.csv"

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

# Load model
model = CatBoostClassifier()
model.load_model(MODEL_PATH)

# Load dataset to get unique values
df = pd.read_csv(DATASET_PATH)

payment_currency_options = sorted(df["Payment_currency"].dropna().unique())
received_currency_options = sorted(df["Received_currency"].dropna().unique())
sender_bank_location_options = sorted(df["Sender_bank_location"].dropna().unique())
receiver_bank_location_options = sorted(df["Receiver_bank_location"].dropna().unique())
payment_type_options = sorted(df["Payment_type"].dropna().unique())

@app.route("/", methods=["GET", "POST"])
def index():
    result = None

    if request.method == "POST":
        transaction = {
            "Sender_account": request.form["Sender_account"],
            "Receiver_account": request.form["Receiver_account"],
            "Payment_currency": request.form["Payment_currency"],
            "Received_currency": request.form["Received_currency"],
            "Sender_bank_location": request.form["Sender_bank_location"],
            "Receiver_bank_location": request.form["Receiver_bank_location"],
            "Payment_type": request.form["Payment_type"],
            "Amount": float(request.form["Amount"]),
            "Date": request.form["Date"],
            "Time": request.form["Time"]
        }

        timestamp = pd.to_datetime(transaction["Date"] + " " + transaction["Time"], dayfirst=True)

        transaction["hour"] = timestamp.hour
        transaction["day"] = timestamp.day
        transaction["month"] = timestamp.month
        transaction["weekday"] = timestamp.weekday()

        transaction.pop("Date")
        transaction.pop("Time")

        X_input = pd.DataFrame([transaction], columns=FEATURE_COLUMNS)

        for col in CATEGORICAL_FEATURES:
            X_input[col] = X_input[col].astype(str)

        input_pool = Pool(data=X_input, cat_features=cat_features_index)

        prob_fraud = model.predict_proba(input_pool)[0][1]
        prediction = prob_fraud >= THRESHOLD

        result = {
            "probability": round(prob_fraud, 6),
            "prediction": "FRAUD" if prediction else "LEGIT"
        }

    return render_template(
        "index.html",
        result=result,
        payment_currency_options=payment_currency_options,
        received_currency_options=received_currency_options,
        sender_bank_location_options=sender_bank_location_options,
        receiver_bank_location_options=receiver_bank_location_options,
        payment_type_options=payment_type_options
    )

if __name__ == "__main__":
    app.run(debug=True)
