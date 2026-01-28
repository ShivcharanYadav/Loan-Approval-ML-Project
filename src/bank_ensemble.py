import streamlit as st
import numpy as np
import joblib

# ===============================
# LOAD MODEL & CHATBOT
# ===============================
model = joblib.load("models/Gradient.pkl")
loan_chatbot_corpus = joblib.load("chatbot/loan_chatbot_corpus.pkl")

FEATURES = [
    'person_age',
    'person_gender',
    'person_education',
    'person_income',
    'person_emp_exp',
    'person_home_ownership',
    'loan_amnt',
    'loan_intent',
    'loan_int_rate',
    'loan_percent_income',
    'cb_person_cred_hist_length',
    'credit_score',
    'previous_loan_defaults_on_file'
]

st.set_page_config(page_title="Loan Approval System", page_icon="🏦")
st.title("🏦 Loan Approval System")

tabs = st.tabs(["🔮 Loan Prediction", "💬 Loan Chatbot"])

# ==================================================
# TAB 1: LOAN PREDICTION
# ==================================================
with tabs[0]:
    st.subheader("Enter Applicant Details")

    # ===== USER INPUTS (IMPORTANT ONLY) =====
    person_age = st.number_input("Age", min_value=18, max_value=100)

    gender = st.selectbox("Gender", ["female", "male"])
    education = st.selectbox(
        "Education",
        ["Associate", "Bachelor", "Doctorate", "High School", "Master"]
    )

    person_income = st.number_input("Annual Income", min_value=1)

    home_ownership = st.selectbox(
        "Home Ownership",
        ["MORTGAGE", "OTHER", "OWN", "RENT"]
    )

    loan_amnt = st.number_input("Loan Amount", min_value=0)

    loan_intent = st.selectbox(
        "Loan Intent",
        ["EDUCATION", "HOMEIMPROVEMENT", "MEDICAL", "PERSONAL", "VENTURE"]
    )

    loan_int_rate = st.number_input("Loan Interest Rate (%)", min_value=0.0)

    credit_score = st.number_input("Credit Score", min_value=300, max_value=900)

    previous_default = st.selectbox("Previous Loan Default", ["No", "Yes"])

    # ===== DEFAULT / AUTO VALUES =====
    person_emp_exp = 2                    # default
    cb_person_cred_hist_length = 5        # default

    # ===== LABEL ENCODING =====
    encoded = {
        'person_age': person_age,
        'person_gender': 0 if gender == "female" else 1,
        'person_education': {
            "Associate": 0,
            "Bachelor": 1,
            "Doctorate": 2,
            "High School": 3,
            "Master": 4
        }[education],
        'person_income': person_income,
        'person_emp_exp': person_emp_exp,
        'person_home_ownership': {
            "MORTGAGE": 0,
            "OTHER": 1,
            "OWN": 2,
            "RENT": 3
        }[home_ownership],
        'loan_amnt': loan_amnt,
        'loan_intent': {
            "EDUCATION": 0,
            "HOMEIMPROVEMENT": 1,
            "MEDICAL": 2,
            "PERSONAL": 3,
            "VENTURE": 4
        }[loan_intent],
        'loan_int_rate': loan_int_rate,
        'loan_percent_income': loan_amnt / person_income,
        'cb_person_cred_hist_length': cb_person_cred_hist_length,
        'credit_score': credit_score,
        'previous_loan_defaults_on_file': 0 if previous_default == "No" else 1
    }

    if st.button("Predict Loan Status"):
        input_data = np.array([[encoded[f] for f in FEATURES]])
        prediction = model.predict(input_data)[0]

        if prediction == 1:
            st.success("✅ Loan Approved")
        else:
            st.error("❌ Loan Not Approved")

# ==================================================
# TAB 2: CHATBOT
# ==================================================
with tabs[1]:
    st.subheader("💬 Loan Assistance Chatbot")

    user_question = st.text_input("Ask a question")

    if st.button("Get Answer"):
        answer = next(
            (item["answer"] for item in loan_chatbot_corpus
             if user_question.lower() in item["question"].lower()),
            None
        )

        if answer:
            st.success(answer)
        else:
            st.warning("Sorry, I don't have an answer yet.")
