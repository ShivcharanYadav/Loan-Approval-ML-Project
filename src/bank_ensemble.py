import streamlit as st
import numpy as np
import joblib

# ===============================
# LOAD MODEL & CHATBOT
# ===============================
model = joblib.load("models/Gradient.pkl")
loan_chatbot_corpus = joblib.load("chatbot/loan_chatbot_corpus.pkl")

st.set_page_config(page_title="Loan Approval System", page_icon="🏦")

st.title("🏦 Loan Approval System")

tabs = st.tabs(["🔮 Loan Prediction", "💬 Loan Chatbot"])

# ==================================================
# TAB 1: LOAN PREDICTION (FIXED FEATURE ORDER)
# ==================================================
with tabs[0]:
    st.subheader("Enter Applicant Details")

    person_age = st.number_input("Age", min_value=18, max_value=100)
    gender = st.selectbox("Gender", ["female", "male"])
    education = st.selectbox(
        "Education",
        ["Associate", "Bachelor", "Doctorate", "High School", "Master"]
    )
    person_income = st.number_input("Annual Income", min_value=0)
    person_emp_exp = st.number_input("Employment Experience (Years)", min_value=0)

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

    # ===== Label Encoding (MATCH NOTEBOOK) =====
    gender = 0 if gender == "female" else 1

    education_map = {
        "Associate": 0,
        "Bachelor": 1,
        "Doctorate": 2,
        "High School": 3,
        "Master": 4
    }

    home_ownership_map = {
        "MORTGAGE": 0,
        "OTHER": 1,
        "OWN": 2,
        "RENT": 3
    }

    loan_intent_map = {
        "EDUCATION": 0,
        "HOMEIMPROVEMENT": 1,
        "MEDICAL": 2,
        "PERSONAL": 3,
        "VENTURE": 4
    }

    previous_default = 0 if previous_default == "No" else 1

    education = education_map[education]
    home_ownership = home_ownership_map[home_ownership]
    loan_intent = loan_intent_map[loan_intent]

    if st.button("Predict Loan Status"):
        # ⚠️ EXACT FEATURE ORDER (11 features)
        input_data = np.array([[
            person_age,
            gender,
            education,
            person_income,
            person_emp_exp,
            home_ownership,
            loan_amnt,
            loan_intent,
            loan_int_rate,
            credit_score,
            previous_default
            loan_percent_income
        ]])

        prediction = model.predict(input_data)[0]

        if prediction == 1:
            st.success("✅ Loan Approved")
        else:
            st.error("❌ Loan Not Approved")

# ==================================================
# TAB 2: CHATBOT (SAFE & SIMPLE)
# ==================================================
with tabs[1]:
    st.subheader("💬 Loan Assistance Chatbot")
    st.write("Ask questions related to loans, approval process, credit score, etc.")

    user_question = st.text_input("Type your question here")

    if st.button("Get Answer"):
        response = None

        for item in loan_chatbot_corpus:
            if user_question.lower() in item["question"].lower():
                response = item["answer"]
                break

        if response:
            st.success(response)
        else:
            st.warning("Sorry, I don't have an answer for that yet.")
