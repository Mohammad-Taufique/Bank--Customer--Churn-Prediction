
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

# ---------------------------------
# 1. PAGE CONFIGURATION
# ---------------------------------

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"

print("APP FILE:", Path(__file__).resolve())
print("MODEL DIRECTORY:", MODEL_DIR)
print("MODEL EXISTS:", MODEL_DIR.exists())
print("MODEL FILES:", list(MODEL_DIR.iterdir()))

st.write("Checking model file paths...")

for filename in [
    "best_random_forest.pkl",
    "encoder.pkl",
    "scaler.pkl",
    "feature_column.pkl"
]:
    file_path = MODEL_DIR / filename

    st.write("File:", repr(str(file_path)))
    st.write("Exists:", file_path.exists())
    st.write("Is file:", file_path.is_file())

st.write("Project directory:", BASE_DIR)
st.write("Model directory:", MODEL_DIR)
st.write("Model folder exists:", MODEL_DIR.exists())
st.write("Model files:", [f.name for f in MODEL_DIR.glob("*")])

@st.cache_resource
def load_artifacts():

    # Find model files safely
    files = {
        p.name.strip().lower(): p
        for p in MODEL_DIR.iterdir()
        if p.is_file()
    }

    required_files = [
        "best_random_forest.pkl",
        "encoder.pkl",
        "scaler.pkl",
        "feature_column.pkl"
    ]

    # Check all required files
    for filename in required_files:
        if filename not in files:
            raise FileNotFoundError(
                f"Missing model file: {filename}"
            )

    # Load artifacts
    model = joblib.load(files["best_random_forest.pkl"])
    encoder = joblib.load(files["encoder.pkl"])
    scaler = joblib.load(files["scaler.pkl"])
    feature_columns = joblib.load(files["feature_column.pkl"])

    return model, encoder, scaler, feature_columns

# ---------------------------------
# 2. LOAD SAVED MODEL AND ARTIFACTS
# ---------------------------------

@st.cache_resource
def load_artifacts():
    model = joblib.load(
        MODEL_DIR / "best_random_forest.pkl"
    )
    encoder = joblib.load(
        MODEL_DIR / "encoder.pkl"
    )
    scaler = joblib.load(
        MODEL_DIR / "scaler.pkl"
    )
    feature_columns = joblib.load(
        MODEL_DIR / "feature_column.pkl"
    )

    return model, encoder, scaler, feature_columns

try:
    model, encoder, scaler, feature_columns = load_artifacts()
except Exception as e:
    st.error(f"Unable to load model files: {e}")
    st.stop()

# ---------------------------------
# 3. DASHBOARD HEADER
# ---------------------------------

st.title("🏦 Bank Customer Churn Prediction")
st.markdown(
    "### AI-Powered Customer Retention Analytics"
)

st.write(
    "Enter customer information below to predict "
    "the probability of customer churn."
)

st.divider()

# ---------------------------------
# 4. CUSTOMER INPUT FORM
# ---------------------------------

st.subheader("👤 Customer Information")

with st.form("customer_form"):

    col1, col2, col3 = st.columns(3)

    with col1:
        credit_score = st.number_input(
            "Credit Score",
            min_value=300,
            max_value=850,
            value=650
        )

        age = st.number_input(
            "Age",
            min_value=18,
            max_value=100,
            value=35
        )

        tenure = st.number_input(
            "Tenure (Years)",
            min_value=0,
            max_value=10,
            value=5
        )

    with col2:
        balance = st.number_input(
            "Account Balance",
            min_value=0.0,
            value=50000.0,
            step=1000.0
        )

        num_products = st.number_input(
            "Number of Products",
            min_value=1,
            max_value=4,
            value=2
        )

        geography = st.selectbox(
            "Geography",
            ["France", "Germany", "Spain"]
        )

    with col3:
        gender = st.selectbox(
            "Gender",
            ["Female", "Male"]
        )

        has_card = st.selectbox(
            "Has Credit Card?",
            ["Yes", "No"]
        )

        active_member = st.selectbox(
            "Is Active Member?",
            ["Yes", "No"]
        )

        salary = st.number_input(
            "Estimated Salary",
            min_value=0.0,
            value=60000.0,
            step=1000.0
        )

    submitted = st.form_submit_button(
        "🔍 Predict Customer Churn",
        use_container_width=True
    )

# ---------------------------------
# 5. PREPROCESS CUSTOMER INPUT
# ---------------------------------

if submitted:

    numerical_cols = [
        "CreditScore",
        "Age",
        "Tenure",
        "Balance",
        "NumOfProducts",
        "EstimatedSalary"
    ]

    categorical_cols = [
        "Geography",
        "Gender"
    ]

    input_data = pd.DataFrame([{
        "CreditScore": credit_score,
        "Age": age,
        "Tenure": tenure,
        "Balance": balance,
        "NumOfProducts": num_products,
        "HasCrCard": 1 if has_card == "Yes" else 0,
        "IsActiveMember": (
            1 if active_member == "Yes" else 0
        ),
        "EstimatedSalary": salary,
        "Geography": geography,
        "Gender": gender
    }])

    # Scale numerical features
    scaled_values = scaler.transform(
        input_data[numerical_cols]
    )

    scaled_df = pd.DataFrame(
        scaled_values,
        columns=numerical_cols
    )

    # Encode categorical features
    encoded_values = encoder.transform(
        input_data[categorical_cols]
    )

    encoded_cols = encoder.get_feature_names_out(
        categorical_cols
    )

    encoded_df = pd.DataFrame(
        encoded_values,
        columns=encoded_cols
    )

    # Combine all processed features
    processed_data = pd.concat(
        [
            scaled_df,
            input_data[["HasCrCard", "IsActiveMember"]],
            encoded_df
        ],
        axis=1
    )

    # Match exact model training feature order
    processed_data = processed_data.reindex(
        columns=feature_columns,
        fill_value=0
    )

    # ---------------------------------
    # 6. MAKE PREDICTION
    # ---------------------------------

    prediction = model.predict(processed_data)[0]

    churn_probability = model.predict_proba(
        processed_data
    )[0][1]

    churn_percentage = churn_probability * 100

    # ---------------------------------
    # 7. RISK CLASSIFICATION
    # ---------------------------------

    if churn_percentage >= 60:
        risk = "HIGH"
    elif churn_percentage >= 30:
        risk = "MEDIUM"
    else:
        risk = "LOW"

    st.divider()

    st.subheader("📊 Prediction Results")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Churn Probability",
            f"{churn_percentage:.2f}%"
        )

    with col2:
        st.metric(
            "Risk Classification",
            risk
        )

    with col3:
        st.metric(
            "Prediction",
            "Will Churn" if prediction == 1
            else "Will Stay"
        )

    st.progress(float(churn_probability))

    if prediction == 1:
        st.error(
            "⚠️ This customer is predicted to churn."
        )
    else:
        st.success(
            "✅ This customer is predicted to stay."
        )

    st.info(
        f"Customer churn probability: "
        f"{churn_percentage:.2f}%"
    )

    st.caption(
        "Risk bands are illustrative dashboard thresholds: "
        "Low <30%, Medium 30–59.99%, High ≥60%. "
        "They are not independently calibrated business thresholds."
    )

    st.divider()

    st.subheader("💡 Customer Retention Recommendation")

    if risk == "HIGH":
        st.write(
            "Prioritize this customer for retention. "
            "Consider personalized offers, proactive "
            "support and a relationship review."
        )

    elif risk == "MEDIUM":
        st.write(
            "Monitor customer engagement and consider "
            "targeted loyalty benefits."
        )

    else:
        st.write(
            "Maintain regular engagement and continue "
            "providing a positive customer experience."
        )

st.divider()

st.caption(
    "Developed using Python, Scikit-learn, "
    "Random Forest and Streamlit."
)