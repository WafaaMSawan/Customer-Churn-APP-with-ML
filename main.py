import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_sample_weight
import pickle

# 1. Load data
df= pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')

# 2. Replace specific string values for simplification
df.replace(["No internet service", "No phone service"], "No", inplace=True)

# 3. Convert TotalCharges to numeric and handle missing values
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df["TotalCharges"] = df["TotalCharges"].fillna(0)

# 4. Drop irrelevant columns
df.drop(["customerID", "gender"], axis=1, inplace=True)

# 5. Encode target variable
df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

# 6. Split features and target
X = df.drop("Churn", axis=1)
y = df["Churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# 7. Identify column types (based on training set)
categorical_cols = X_train.select_dtypes(include=["object"]).columns.tolist()
numeric_cols = X_train.select_dtypes(include=["int64", "float64"]).columns.tolist()

# 8. Define transformers
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", MinMaxScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(drop="if_binary", sparse_output=False))
])

# 9. ColumnTransformer
preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_transformer, numeric_cols),
    ("cat", categorical_transformer, categorical_cols)
])

# 10. Full pipeline
pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(
        max_iter=1000,
        random_state=42,
        class_weight="balanced"   # handles imbalance directly
    ))
])
# 11. Handle class imbalance
sample_weights = compute_sample_weight(class_weight="balanced", y=y_train)
pipeline.fit(X_train, y_train, classifier__sample_weight=sample_weights)

# 12. Evaluation
y_pred = pipeline.predict(X_test)
print("\n Classification Report:\n")
print(classification_report(y_test, y_pred))

# 13. Save model
with open("pipeline.pkl", "wb") as f:
    pickle.dump(pipeline, f)

print("\n Model saved successfully to gbm_pipeline.pkl")
