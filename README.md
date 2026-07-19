# 💳 AML Fraud Detection System using CatBoost & Flask

<p align="center">

A Machine Learning-powered **Anti-Money Laundering (AML) Fraud Detection System** built with **Python, CatBoost, and Flask** to classify financial transactions as **Fraudulent** or **Legitimate** based on transaction characteristics.

</p>

---

## 📌 Overview

Financial institutions process millions of transactions every day, making manual fraud detection inefficient and error-prone. This project applies Machine Learning to identify potentially suspicious transactions by learning patterns from historical transaction data.

The application provides:

* A CatBoost-based fraud detection model
* Automated feature engineering
* Fraud probability prediction
* Interactive Flask web interface
* Standalone prediction script
* Model evaluation with multiple performance metrics

---

## ✨ Features

* 🚀 CatBoost-based binary classification model
* 📊 Fraud probability prediction
* 🌐 Interactive Flask web application
* ⚡ Automatic feature engineering from transaction timestamps
* 📈 Threshold optimization using Fraud F1-Score
* 📉 ROC-AUC and Precision-Recall evaluation
* 🔍 Handles both numerical and categorical transaction features
* 🧩 Easy to extend for real-world AML applications

---

## 🛠️ Tech Stack

| Category         | Technology    |
| ---------------- | ------------- |
| Language         | Python        |
| Machine Learning | CatBoost      |
| Backend          | Flask         |
| Data Processing  | Pandas, NumPy |
| Model Evaluation | Scikit-learn  |
| Version Control  | Git & GitHub  |

---

# 📂 Project Structure

```text
AML-Fraud-Detection/
│
├── app.py                         # Flask web application
├── train.py                       # Model training pipeline
├── predict.py                     # Standalone prediction script
├── requirements.txt               # Project dependencies
├── README.md
│
├── templates/
│   └── index.html
│
├── static/
│   └── images/

```

---

# 🧠 Machine Learning Workflow

```text
                 Transaction Dataset
                         │
                         ▼
                Data Preprocessing
                         │
                         ▼
               Feature Engineering
        ┌────────────┬─────────────┐
        │ Hour       │ Weekday     │
        │ Day        │ Month       │
        └────────────┴─────────────┘
                         │
                         ▼
               Stratified Train/Test Split
                         │
                         ▼
                CatBoost Classifier
                         │
                         ▼
          Threshold Optimization (F1)
                         │
                         ▼
                 Performance Evaluation
                         │
                         ▼
              Fraud Probability Prediction
                         │
            ┌────────────┴─────────────┐
            ▼                          ▼
      Flask Web App            predict.py
```

---

# 📊 Features Used

The model uses transaction information including:

* Sender Account
* Receiver Account
* Payment Currency
* Received Currency
* Sender Bank Location
* Receiver Bank Location
* Payment Type
* Transaction Amount
* Transaction Timestamp
* Hour
* Day
* Month
* Weekday

---

# ⚙️ Installation

## 1. Clone the Repository

```bash
git clone https:/Growthnook-VKP/github.com//AML-Fraud-Detection.git

cd AML-Fraud-Detection
```

---

## 2. Create a Virtual Environment

### Windows

```bash
python -m venv venv

venv\Scripts\activate
```

### Linux / macOS

```bash
python3 -m venv venv

source venv/bin/activate
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

# 📁 Dataset

The dataset used for this project is **not included** in this repository.

It has been excluded because it may be:

* Proprietary
* Confidential
* Too large for GitHub

To train the model, place the dataset in the project root with the following filename:

```text
dataset.csv
```

---

# 📦 Trained Model

The trained CatBoost model is **not included** in this repository.

Generate it by running:

```bash
python train.py
```

This will create:

```text
aml_fraud_catboost_practical.cbm
```

---

# 🚀 Running the Project

## Step 1 — Generate the Model

```bash
python train.py
```

This trains the CatBoost model and creates:

```text
aml_fraud_catboost_practical.cbm
```

---

## Step 2 — Start the Flask Application

```bash
python app.py
```

Open your browser:

```text
http://127.0.0.1:5000
```

---

## Step 3 — Run Command-Line Prediction

```bash
python predict.py
```

---

## ⚠️ Important Note

The current Flask application reads **dataset.csv** to populate dropdown values in the user interface.

Therefore, **dataset.csv must remain in the project root** while running the application, even after the trained model has been generated.

---

# 📈 Model Evaluation

The training pipeline evaluates the model using:

* Classification Report
* ROC-AUC Score
* Precision-Recall AUC
* Fraud F1-Score
* Optimized Decision Threshold

---


# 📋 Example Workflow

```text
User enters transaction details
            │
            ▼
 Flask receives request
            │
            ▼
 Feature Engineering
            │
            ▼
 CatBoost Prediction
            │
            ▼
 Fraud Probability
            │
            ▼
Fraud / Legitimate Result
```

---

# 🔮 Future Improvements

* REST API using Flask RESTful or FastAPI
* Docker support
* Cloud deployment (AWS, Azure, GCP)
* SHAP Explainable AI integration
* User authentication
* Real-time transaction monitoring
* Dashboard with analytics and visualizations
* Automated model retraining pipeline
* Logging and monitoring

---

# 🤝 Contributing

Contributions are welcome!

1. Fork this repository
2. Create a new feature branch

```bash
git checkout -b feature-name
```

3. Commit your changes

```bash
git commit -m "Add new feature"
```

4. Push to your branch

```bash
git push origin feature-name
```

5. Open a Pull Request

---

# 📄 License

This project is licensed under the **MIT License**.

---

# 👨‍💻 Author

**Vishal Kumar Prasad**

* GitHub: https://github.com/Growthnook-VKP
* LinkedIn: www.linkedin.com/in/vishal-kumar-prasad-a1b25b26b

---

# 🌟 Support

If you found this project useful:

⭐ Star this repository

🍴 Fork this repository

💡 Share your feedback or suggestions by opening an Issue.

---

<p align="center">
Made with ❤️ using Python, CatBoost and Flask
</p>

