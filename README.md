# 📡 Telecom Customer Churn Prediction App

A full-stack machine learning application that predicts customer churn in the telecom industry. Built with **scikit-learn** pipelines and a polished **Streamlit** interface, it lets you upload data, train multiple classifiers, evaluate their performance, and get instant single-customer churn predictions.

---

## ✨ Features

- **5 ML Classifiers** — Logistic Regression, Gradient Boosting, Random Forest, SVM, and Naive Bayes
- **Automated Preprocessing Pipeline** — handles missing values, numeric scaling (MinMax), and categorical one-hot encoding via `sklearn.pipeline`
- **Class Imbalance Handling** — uses `class_weight="balanced"` and `compute_sample_weight` for Gradient Boosting
- **Rich Evaluation Dashboard** — ROC-AUC score, accuracy, precision, recall, F1, ROC curve, confusion matrix, and a styled classification report
- **Single-Customer Prediction** — interactive form to enter customer details and get a real-time churn probability with a visual risk gauge
- **Dataset Preview Tab** — inspect raw data, missing values, and churn distribution chart
- **Model Persistence** — save the trained pipeline as a `.pkl` file and reload it later for inference without retraining
- **Dark-themed UI** — custom CSS with IBM Plex fonts, GitHub-inspired dark palette

---

## 🗂️ Project Structure

```
Customer-Churn-APP-with-ML/
│
├── app.py                          # Main Streamlit application
├── main.py                         # Entry point / runner script
├── ML_telecom_full_pipeline.ipynb  # Jupyter notebook — full ML pipeline & EDA
└── requirements.txt                # Python dependencies
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.9 or higher
- pip

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/WafaaMSawan/Customer-Churn-APP-with-ML.git
cd Customer-Churn-APP-with-ML

# 2. (Recommended) Create a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

### Run the App

```bash
streamlit run app.py
```

The app will open at `http://localhost:8501` in your browser.

---

## 📊 How to Use

### Tab 1 — Model Evaluation

1. **Upload** your Telco Churn CSV file via the sidebar (`WA_Fn-UseC_-Telco-Customer-Churn.csv` format).
2. **Select** a classifier from the dropdown.
3. Click **🚀 Train & Evaluate** — the pipeline preprocesses the data, trains the model, and displays:
   - Key metrics: ROC-AUC, Accuracy, Precision, Recall, F1-Score
   - ROC Curve and Confusion Matrix charts
   - Full classification report table
4. Optionally **💾 Save Pipeline** to export the trained model as `churn_pipeline.pkl`.

### Tab 2 — Single Prediction

Fill in the customer detail form (account info, phone/internet services, charges) and click **🔍 Predict Churn** to get:
- A **Churn / Stay** verdict
- Churn probability and retention probability
- A color-coded **risk gauge bar** (green → yellow → red)

> If you have a saved `.pkl` model, the app auto-loads it — no retraining needed.

### Tab 3 — Data Preview

Inspect the first 100 rows of the uploaded dataset, check for missing values, and view the churn class distribution chart.

---

## 🤖 Supported Models

| Model | Notes |
|---|---|
| Logistic Regression | `class_weight="balanced"`, `max_iter=1000` |
| Gradient Boosting | `learning_rate=0.05`, `n_estimators=150`, sample weights for imbalance |
| Random Forest | `n_estimators=150`, `class_weight="balanced"` |
| SVM | `probability=True`, `class_weight="balanced"` |
| Naive Bayes | GaussianNB — fast baseline |

---

## 🛠️ Preprocessing Pipeline

The `sklearn` preprocessing pipeline applied to the training data:

- **Numeric features** → `SimpleImputer(median)` → `MinMaxScaler`
- **Categorical features** → `SimpleImputer(most_frequent)` → `OneHotEncoder(drop="if_binary")`
- `"No internet service"` and `"No phone service"` values are normalized to `"No"` before fitting
- `customerID` and `gender` columns are dropped; `TotalCharges` is coerced to numeric

---

## 📋 Dataset

This app is built for the **IBM Telco Customer Churn** dataset. You can download it from [Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn).

**Key columns used:**

| Column | Type | Description |
|---|---|---|
| `tenure` | Numeric | Months the customer has stayed |
| `MonthlyCharges` | Numeric | Current monthly fee |
| `TotalCharges` | Numeric | Total amount charged |
| `Contract` | Categorical | Month-to-month / One year / Two year |
| `InternetService` | Categorical | DSL / Fiber optic / No |
| `PaymentMethod` | Categorical | Electronic check, etc. |
| `Churn` | Target | Yes / No |

---

## 📦 Key Dependencies

| Package | Purpose |
|---|---|
| `streamlit` | Web application framework |
| `scikit-learn` | ML models and preprocessing pipelines |
| `pandas` / `numpy` | Data manipulation |
| `matplotlib` | Plots (ROC curve, confusion matrix, bar chart) |
| `imbalanced-learn` | Class imbalance utilities |
| `pickle` | Model serialization |

See `requirements.txt` for the full pinned dependency list.

---

## 📓 Notebook

`ML_telecom_full_pipeline.ipynb` contains the exploratory data analysis, feature engineering, and end-to-end pipeline experimentation that informed the app design.

---

## 📄 License

This project is open-source. Feel free to use, modify, and distribute it with attribution.
