import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Customer Churn Prediction", layout="centered")

st.title("Customer Churn Prediction")
st.write(
    "This app predicts whether a customer will churn (Yes or No) "
    "using a trained Machine Learning pipeline."
)

# Load saved preprocessing + model pipeline
try:
    model = joblib.load("churn_pipeline.pkl")
except Exception:
    st.error("Could not load churn_pipeline.pkl. Train and save the model first.")
    st.stop()

st.header("Enter Customer Details")

# Numeric inputs
age = st.number_input("Age", min_value=18, max_value=80, value=40, step=1)

tenure_months = st.slider("Tenure (months)", min_value=1, max_value=72, value=12)

monthly_charges = st.number_input(
    "Monthly Charges", min_value=0.0, max_value=150.0, value=70.0, step=0.1
)

total_charges = st.number_input(
    "Total Charges", min_value=0.0, max_value=8000.0, value=1000.0, step=1.0
)

num_support_calls = st.slider(
    "Number of Support Calls", min_value=0, max_value=8, value=1
)

late_payments_last_year = st.slider(
    "Late Payments Last Year", min_value=0, max_value=6, value=1
)

avg_monthly_usage_gb = st.number_input(
    "Average Monthly Usage (GB)", min_value=0.0, max_value=700.0, value=200.0, step=0.1
)

# Categorical inputs
gender = st.selectbox("Gender", ["Female", "Male"])
region = st.selectbox("Region", ["East", "North", "South", "West"])
contract_type = st.selectbox(
    "Contract Type", ["Month-to-month", "One year", "Two year"]
)
internet_service = st.selectbox(
    "Internet Service", ["DSL", "Fiber optic", "No"]
)
tech_support = st.selectbox(
    "Tech Support", ["Yes", "No", "No internet service"]
)
online_security = st.selectbox(
    "Online Security", ["Yes", "No", "No internet service"]
)
paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
payment_method = st.selectbox(
    "Payment Method",
    ["Electronic check", "Bank transfer", "Credit card", "Mailed check"],
)

if st.button("Predict Churn"):
    try:
        # Same column names as training X
        new_customer = pd.DataFrame(
            {
                "age": [age],
                "gender": [gender],
                "region": [region],
                "tenure_months": [tenure_months],
                "monthly_charges": [monthly_charges],
                "total_charges": [total_charges],
                "contract_type": [contract_type],
                "internet_service": [internet_service],
                "tech_support": [tech_support],
                "online_security": [online_security],
                "paperless_billing": [paperless_billing],
                "payment_method": [payment_method],
                "num_support_calls": [num_support_calls],
                "late_payments_last_year": [late_payments_last_year],
                "avg_monthly_usage_gb": [avg_monthly_usage_gb],
            }
        )

        prediction = model.predict(new_customer)[0]

        # Bonus: show confidence if available
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(new_customer)[0]
            classes = list(model.classes_)
            confidence = float(max(proba))
            st.write(f"Confidence: {confidence:.1%}")

        if prediction == "Yes":
            st.error("Prediction: Yes — Customer is likely to churn")
        else:
            st.success("Prediction: No — Customer is likely to stay")

    except Exception as e:
        st.error("Something went wrong with the prediction. Please check your inputs.")
        st.caption(str(e))
