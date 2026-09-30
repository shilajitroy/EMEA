from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st


MODEL_PATH = Path(__file__).resolve().with_name(
    "loan_default_random_forest.joblib"
)


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


st.set_page_config(page_title="Loan Repayment Risk", layout="wide")
st.title("Loan Repayment Risk")
st.caption("LendingClub Random Forest model")

st.warning(
    "Educational demo only. The model had very low recall for loans not fully "
    "paid in its evaluation. Do not use it for real credit or lending decisions."
)

try:
    model = load_model()
except FileNotFoundError:
    st.error(f"Model file not found: {MODEL_PATH.name}")
    st.stop()

model_features = list(model.feature_names_in_)
purpose_options = ["all_other"] + sorted(
    feature.split("purpose_", 1)[1]
    for feature in model_features
    if feature.startswith("purpose_")
)

with st.form("loan_prediction_form"):
    borrower_col, loan_col = st.columns(2)

    with borrower_col:
        st.subheader("Borrower profile")
        purpose = st.selectbox("Loan purpose", purpose_options)
        credit_policy = st.selectbox(
            "Meets credit policy criteria",
            ["Yes", "No"],
        )
        annual_income = st.number_input(
            "Annual income ($)",
            min_value=1.0,
            value=60000.0,
            step=5000.0,
        )
        dti = st.number_input(
            "Debt-to-income ratio (%)",
            min_value=0.0,
            max_value=100.0,
            value=15.0,
            step=0.5,
        )
        fico = st.number_input(
            "FICO score",
            min_value=300,
            max_value=850,
            value=700,
            step=1,
        )
        days_with_credit_line = st.number_input(
            "Days with credit line",
            min_value=0,
            value=5000,
            step=100,
        )

    with loan_col:
        st.subheader("Loan and credit")
        interest_rate = st.number_input(
            "Interest rate (%)",
            min_value=0.0,
            max_value=100.0,
            value=12.0,
            step=0.1,
        )
        installment = st.number_input(
            "Monthly installment ($)",
            min_value=0.0,
            value=350.0,
            step=25.0,
        )
        revolving_balance = st.number_input(
            "Revolving balance ($)",
            min_value=0,
            value=15000,
            step=500,
        )
        revolving_utilization = st.number_input(
            "Revolving utilization (%)",
            min_value=0.0,
            max_value=200.0,
            value=50.0,
            step=1.0,
        )
        inquiries = st.number_input(
            "Credit inquiries in last 6 months",
            min_value=0,
            value=1,
            step=1,
        )
        delinquencies = st.number_input(
            "Delinquencies in last 2 years",
            min_value=0,
            value=0,
            step=1,
        )
        public_records = st.number_input(
            "Derogatory public records",
            min_value=0,
            value=0,
            step=1,
        )

    submitted = st.form_submit_button(
        "Estimate repayment risk",
        type="primary",
        use_container_width=True,
    )

if submitted:
    input_values = {
        "credit.policy": 1 if credit_policy == "Yes" else 0,
        "int.rate": interest_rate / 100,
        "installment": installment,
        "log.annual.inc": np.log(annual_income),
        "dti": dti,
        "fico": fico,
        "days.with.cr.line": days_with_credit_line,
        "revol.bal": revolving_balance,
        "revol.util": revolving_utilization,
        "inq.last.6mths": inquiries,
        "delinq.2yrs": delinquencies,
        "pub.rec": public_records,
    }

    model_input = pd.DataFrame(
        0.0,
        index=[0],
        columns=model_features,
    )

    for feature, value in input_values.items():
        model_input.at[0, feature] = value

    purpose_feature = f"purpose_{purpose}"
    if purpose_feature in model_input.columns:
        model_input.at[0, purpose_feature] = 1

    prediction = int(model.predict(model_input)[0])
    positive_class_index = list(model.classes_).index(1)
    risk_probability = float(
        model.predict_proba(model_input)[0, positive_class_index]
    )

    st.subheader("Model estimate")
    result_col, probability_col = st.columns(2)

    with result_col:
        estimate = "Not fully paid" if prediction == 1 else "Fully paid"
        st.metric("Predicted class", estimate)

    with probability_col:
        st.metric("Estimated probability of not being fully paid",
                  f"{risk_probability:.1%}")

    if prediction == 1:
        st.warning("The model predicts the loan may not be fully paid.")
    else:
        st.success("The model predicts the loan will be fully paid.")