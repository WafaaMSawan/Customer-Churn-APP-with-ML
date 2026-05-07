import streamlit as st
import numpy as np
import pandas as pd
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    classification_report, roc_auc_score, confusion_matrix,
    roc_curve, ConfusionMatrixDisplay
)
from sklearn.utils.class_weight import compute_sample_weight
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Telecom Churn Predictor",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'IBM Plex Sans', sans-serif;
    }
    .main { background-color: #0d1117; }
    .block-container { padding-top: 2rem; }

    h1, h2, h3 { font-family: 'IBM Plex Mono', monospace; color: #58a6ff; }

    .metric-card {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 10px;
        padding: 1.2rem 1.5rem;
        text-align: center;
        margin-bottom: 1rem;
    }
    .metric-card h4 { color: #8b949e; font-size: 0.8rem; margin: 0; letter-spacing: 1px; text-transform: uppercase; }
    .metric-card p  { color: #58a6ff; font-size: 2rem; font-weight: 700; margin: 0.3rem 0 0 0; font-family: 'IBM Plex Mono', monospace; }

    .churn-yes { background: #3d1a1a; border-color: #f85149; }
    .churn-yes p { color: #f85149 !important; }
    .churn-no  { background: #0d2618; border-color: #3fb950; }
    .churn-no  p { color: #3fb950 !important; }

    .stButton>button {
        background: linear-gradient(135deg, #1f6feb, #58a6ff);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.6rem 2rem;
        font-family: 'IBM Plex Mono', monospace;
        font-weight: 600;
        letter-spacing: 0.5px;
        width: 100%;
        transition: opacity 0.2s;
    }
    .stButton>button:hover { opacity: 0.85; }

    .stSelectbox label, .stSlider label, .stNumberInput label, .stRadio label {
        color: #c9d1d9 !important; font-size: 0.85rem;
    }

    div[data-testid="stSidebar"] {
        background-color: #161b22;
        border-right: 1px solid #30363d;
    }

    .info-box {
        background: #161b22;
        border-left: 4px solid #58a6ff;
        border-radius: 0 8px 8px 0;
        padding: 1rem 1.2rem;
        margin: 1rem 0;
        color: #c9d1d9;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
CLASSIFIERS = {
    "Logistic Regression": "lr",
    "Gradient Boosting": "gbm",
    "Random Forest": "rf",
    "SVM": "svm",
    "Naive Bayes": "nb",
}

def preprocess(df):
    df = df.copy()
    df.replace(["No internet service", "No phone service"], "No", inplace=True)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(0)
    if "customerID" in df.columns:
        df.drop(["customerID"], axis=1, inplace=True)
    if "gender" in df.columns:
        df.drop(["gender"], axis=1, inplace=True)
    if "Churn" in df.columns:
        df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})
    return df

def build_pipeline(clf_key):
    if clf_key == "lr":
        clf = LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced")
    elif clf_key == "gbm":
        clf = GradientBoostingClassifier(learning_rate=0.05, n_estimators=150, random_state=42)
    elif clf_key == "rf":
        clf = RandomForestClassifier(n_estimators=150, random_state=42, class_weight="balanced")
    elif clf_key == "svm":
        clf = SVC(probability=True, random_state=42, class_weight="balanced")
    else:
        clf = GaussianNB()
    return clf

@st.cache_resource(show_spinner=False)
def train_model(clf_key, data_hash):
    df = st.session_state["raw_df"].copy()
    df = preprocess(df)

    X = df.drop("Churn", axis=1)
    y = df["Churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    categorical_cols = X_train.select_dtypes(include=["object"]).columns.tolist()
    numeric_cols     = X_train.select_dtypes(include=["int64","float64"]).columns.tolist()

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  MinMaxScaler())
    ])
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot",  OneHotEncoder(drop="if_binary", sparse_output=False, handle_unknown="ignore"))
    ])
    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_transformer,    numeric_cols),
        ("cat", categorical_transformer, categorical_cols)
    ])

    clf = build_pipeline(clf_key)
    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier",   clf)
    ])

    if clf_key == "gbm":
        sw = compute_sample_weight(class_weight="balanced", y=y_train)
        pipeline.fit(X_train, y_train, classifier__sample_weight=sw)
    else:
        pipeline.fit(X_train, y_train)

    y_pred      = pipeline.predict(X_test)
    y_prob      = pipeline.predict_proba(X_test)[:, 1]
    roc_auc     = roc_auc_score(y_test, y_prob)
    report_dict = classification_report(y_test, y_pred, output_dict=True)
    cm          = confusion_matrix(y_test, y_pred)
    fpr, tpr, _ = roc_curve(y_test, y_prob)

    return pipeline, roc_auc, report_dict, cm, fpr, tpr, X_test, categorical_cols, numeric_cols


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📡 Churn Predictor")
    st.markdown("---")

    st.markdown("### 1. Upload Dataset")
    uploaded = st.file_uploader("Upload CSV (Telco Churn format)", type=["csv"])

    if uploaded:
        raw_df = pd.read_csv(uploaded)
        st.session_state["raw_df"] = raw_df
        st.success(f"✅ Loaded {len(raw_df):,} rows")

    st.markdown("---")
    st.markdown("### 2. Select Classifier")
    clf_label = st.selectbox("Model", list(CLASSIFIERS.keys()))
    clf_key   = CLASSIFIERS[clf_label]

    st.markdown("---")
    train_btn = st.button("🚀 Train & Evaluate")

    st.markdown("---")
    st.markdown("### About")
    st.markdown("""
    <div class="info-box">
    Telecom customer churn prediction using sklearn pipelines.<br><br>
    • 5 classifiers supported<br>
    • Handles class imbalance<br>
    • Full evaluation metrics
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# MAIN AREA
# ─────────────────────────────────────────────
st.markdown("# 📡 Telecom Customer Churn Prediction")
st.markdown("---")

tab1, tab2, tab3 = st.tabs(["📊 Model Evaluation", "🔍 Single Prediction", "📋 Data Preview"])

# ── TAB 1: EVALUATION ──────────────────────────
with tab1:
    if "raw_df" not in st.session_state:
        st.markdown("""
        <div class="info-box">
        👈 Upload your <b>WA_Fn-UseC_-Telco-Customer-Churn.csv</b> file from the sidebar to get started.
        </div>
        """, unsafe_allow_html=True)
    else:
        if train_btn or "pipeline" in st.session_state:
            if train_btn:
                with st.spinner("Training pipeline..."):
                    data_hash = str(len(st.session_state["raw_df"]))
                    result = train_model(clf_key, data_hash)
                    st.session_state["pipeline"]      = result[0]
                    st.session_state["roc_auc"]       = result[1]
                    st.session_state["report_dict"]   = result[2]
                    st.session_state["cm"]            = result[3]
                    st.session_state["fpr"]           = result[4]
                    st.session_state["tpr"]           = result[5]
                    st.session_state["X_test"]        = result[6]
                    st.session_state["categorical_cols"] = result[7]
                    st.session_state["numeric_cols"]     = result[8]
                    st.session_state["clf_key"]       = clf_key

            if "pipeline" in st.session_state:
                rd  = st.session_state["report_dict"]
                auc = st.session_state["roc_auc"]
                cm  = st.session_state["cm"]
                fpr = st.session_state["fpr"]
                tpr = st.session_state["tpr"]

                # ── Metric Cards ──
                c1, c2, c3, c4, c5 = st.columns(5)
                metrics = [
                    ("ROC-AUC",   f"{auc:.3f}"),
                    ("Accuracy",  f"{rd['accuracy']:.3f}"),
                    ("Precision", f"{rd['1']['precision']:.3f}"),
                    ("Recall",    f"{rd['1']['recall']:.3f}"),
                    ("F1-Score",  f"{rd['1']['f1-score']:.3f}"),
                ]
                for col, (label, val) in zip([c1,c2,c3,c4,c5], metrics):
                    col.markdown(f"""
                    <div class="metric-card">
                        <h4>{label}</h4>
                        <p>{val}</p>
                    </div>""", unsafe_allow_html=True)

                st.markdown("---")

                # ── ROC + Confusion Matrix ──
                col_a, col_b = st.columns(2)

                with col_a:
                    st.markdown("#### ROC Curve")
                    fig, ax = plt.subplots(figsize=(5, 4))
                    fig.patch.set_facecolor("#161b22")
                    ax.set_facecolor("#0d1117")
                    ax.plot(fpr, tpr, color="#58a6ff", lw=2, label=f"AUC = {auc:.3f}")
                    ax.plot([0,1],[0,1], color="#30363d", lw=1, linestyle="--")
                    ax.set_xlabel("False Positive Rate", color="#8b949e")
                    ax.set_ylabel("True Positive Rate",  color="#8b949e")
                    ax.tick_params(colors="#8b949e")
                    ax.spines[:].set_color("#30363d")
                    ax.legend(facecolor="#161b22", labelcolor="#c9d1d9")
                    st.pyplot(fig)
                    plt.close(fig)

                with col_b:
                    st.markdown("#### Confusion Matrix")
                    fig2, ax2 = plt.subplots(figsize=(5, 4))
                    fig2.patch.set_facecolor("#161b22")
                    ax2.set_facecolor("#0d1117")
                    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No Churn","Churn"])
                    disp.plot(ax=ax2, colorbar=False, cmap="Blues")
                    ax2.set_title("", color="#8b949e")
                    ax2.tick_params(colors="#c9d1d9")
                    ax2.xaxis.label.set_color("#8b949e")
                    ax2.yaxis.label.set_color("#8b949e")
                    fig2.patch.set_facecolor("#161b22")
                    st.pyplot(fig2)
                    plt.close(fig2)

                # ── Full Classification Report ──
                st.markdown("---")
                st.markdown("#### Classification Report")
                report_df = pd.DataFrame(rd).transpose().round(3)
                st.dataframe(report_df.style.background_gradient(cmap="Blues", subset=["precision","recall","f1-score"]), use_container_width=True)

                # ── Save Model ──
                st.markdown("---")
                col_save1, col_save2 = st.columns([2,1])
                with col_save1:
                    if st.button("💾 Save Pipeline (.pkl)"):
                        with open("churn_pipeline.pkl", "wb") as f:
                            pickle.dump(st.session_state["pipeline"], f)
                        st.success("Model saved as `churn_pipeline.pkl`")
        else:
            st.info("Click **Train & Evaluate** in the sidebar to start.")


# ── TAB 2: SINGLE PREDICTION ───────────────────
with tab2:
    # Try to load from saved pkl if no pipeline trained yet
    if "pipeline" not in st.session_state:
        auto_names = ["churn_pipeline.pkl", "gbm_pipeline.pkl", "pipeline.pkl"]
        auto_found = next((p for p in auto_names if os.path.exists(p)), None)
        if auto_found:
            with open(auto_found, "rb") as f:
                st.session_state["pipeline"] = pickle.load(f)
            st.success(f"✅ Loaded saved model: `{auto_found}`")

    if "pipeline" not in st.session_state:
        st.info("No model found. Either train one in Tab 1 or place your `.pkl` file in the same folder as app.py.")
    else:
        st.markdown("### Enter Customer Details")
        st.markdown("Fill in the fields below and click **Predict Churn**.")

        with st.form("predict_form"):
            c1, c2, c3 = st.columns(3)

            with c1:
                st.markdown("**Account Info**")
                senior       = st.selectbox("Senior Citizen", [0, 1])
                partner      = st.selectbox("Partner", ["Yes","No"])
                dependents   = st.selectbox("Dependents", ["Yes","No"])
                tenure       = st.slider("Tenure (months)", 0, 72, 12)
                contract     = st.selectbox("Contract", ["Month-to-month","One year","Two year"])
                paperless    = st.selectbox("Paperless Billing", ["Yes","No"])
                payment      = st.selectbox("Payment Method", [
                    "Electronic check","Mailed check",
                    "Bank transfer (automatic)","Credit card (automatic)"
                ])

            with c2:
                st.markdown("**Phone Services**")
                phone        = st.selectbox("Phone Service", ["Yes","No"])
                multi_lines  = st.selectbox("Multiple Lines", ["Yes","No"])
                st.markdown("**Internet Services**")
                internet     = st.selectbox("Internet Service", ["DSL","Fiber optic","No"])
                online_sec   = st.selectbox("Online Security", ["Yes","No"])
                online_bkp   = st.selectbox("Online Backup", ["Yes","No"])
                device_prot  = st.selectbox("Device Protection", ["Yes","No"])

            with c3:
                st.markdown("**Streaming & Support**")
                tech_sup     = st.selectbox("Tech Support", ["Yes","No"])
                streaming_tv = st.selectbox("Streaming TV", ["Yes","No"])
                streaming_mv = st.selectbox("Streaming Movies", ["Yes","No"])
                st.markdown("**Charges**")
                monthly_chg  = st.number_input("Monthly Charges ($)", 0.0, 200.0, 65.0, step=0.5)
                total_chg    = st.number_input("Total Charges ($)", 0.0, 10000.0, float(tenure * monthly_chg), step=1.0)

            submitted = st.form_submit_button("🔍 Predict Churn")

        if submitted:
            input_data = pd.DataFrame([{
                "SeniorCitizen":    senior,
                "Partner":          partner,
                "Dependents":       dependents,
                "tenure":           tenure,
                "PhoneService":     phone,
                "MultipleLines":    multi_lines,
                "InternetService":  internet,
                "OnlineSecurity":   online_sec,
                "OnlineBackup":     online_bkp,
                "DeviceProtection": device_prot,
                "TechSupport":      tech_sup,
                "StreamingTV":      streaming_tv,
                "StreamingMovies":  streaming_mv,
                "Contract":         contract,
                "PaperlessBilling": paperless,
                "PaymentMethod":    payment,
                "MonthlyCharges":   monthly_chg,
                "TotalCharges":     total_chg,
            }])

            pipeline    = st.session_state["pipeline"]
            prediction  = pipeline.predict(input_data)[0]
            probability = pipeline.predict_proba(input_data)[0][1]

            st.markdown("---")
            col_res1, col_res2, col_res3 = st.columns(3)

            churn_class = "churn-yes" if prediction == 1 else "churn-no"
            churn_label = "⚠️ WILL CHURN" if prediction == 1 else "✅ WILL STAY"

            col_res1.markdown(f"""
            <div class="metric-card {churn_class}">
                <h4>Prediction</h4>
                <p style="font-size:1.3rem">{churn_label}</p>
            </div>""", unsafe_allow_html=True)

            col_res2.markdown(f"""
            <div class="metric-card">
                <h4>Churn Probability</h4>
                <p>{probability:.1%}</p>
            </div>""", unsafe_allow_html=True)

            col_res3.markdown(f"""
            <div class="metric-card">
                <h4>Retention Probability</h4>
                <p>{1-probability:.1%}</p>
            </div>""", unsafe_allow_html=True)

            # Risk gauge bar
            st.markdown("**Risk Level**")
            color = "#f85149" if probability > 0.6 else "#d29922" if probability > 0.35 else "#3fb950"
            st.markdown(f"""
            <div style="background:#21262d;border-radius:8px;height:20px;overflow:hidden;margin-top:0.3rem">
                <div style="background:{color};width:{probability*100:.1f}%;height:100%;border-radius:8px;
                            transition:width 0.5s ease;display:flex;align-items:center;justify-content:center;">
                    <span style="color:white;font-size:0.75rem;font-weight:600">{probability*100:.1f}%</span>
                </div>
            </div>
            """, unsafe_allow_html=True)


# ── TAB 3: DATA PREVIEW ────────────────────────
with tab3:
    if "raw_df" not in st.session_state:
        st.info("Upload a dataset from the sidebar to preview it here.")
    else:
        df_raw = st.session_state["raw_df"]
        st.markdown(f"### Dataset — {len(df_raw):,} rows × {len(df_raw.columns)} columns")

        col_p1, col_p2, col_p3 = st.columns(3)
        col_p1.metric("Total Customers", f"{len(df_raw):,}")
        churn_counts = df_raw["Churn"].value_counts()
        col_p2.metric("Churned",     f"{churn_counts.get('Yes', 0):,}")
        col_p3.metric("Retained",    f"{churn_counts.get('No',  0):,}")

        st.markdown("---")
        st.dataframe(df_raw.head(100), use_container_width=True)

        st.markdown("---")
        st.markdown("#### Missing Values")
        missing = df_raw.isnull().sum()
        missing = missing[missing > 0]
        if missing.empty:
            st.success("No missing values found.")
        else:
            st.dataframe(missing.rename("Missing Count"), use_container_width=True)

        st.markdown("#### Churn Distribution")
        fig3, ax3 = plt.subplots(figsize=(4,3))
        fig3.patch.set_facecolor("#161b22")
        ax3.set_facecolor("#0d1117")
        counts = df_raw["Churn"].value_counts()
        bars = ax3.bar(counts.index, counts.values, color=["#3fb950","#f85149"], width=0.4)
        ax3.set_xlabel("Churn", color="#8b949e")
        ax3.set_ylabel("Count",  color="#8b949e")
        ax3.tick_params(colors="#8b949e")
        ax3.spines[:].set_color("#30363d")
        for bar in bars:
            ax3.text(bar.get_x()+bar.get_width()/2, bar.get_height()+20,
                     f"{bar.get_height():,}", ha="center", color="#c9d1d9", fontsize=9)
        st.pyplot(fig3)
        plt.close(fig3)
