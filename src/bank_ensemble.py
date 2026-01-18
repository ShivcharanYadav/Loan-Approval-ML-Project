import streamlit as st
import numpy as np
import joblib

# -------------------------------
#  LOAD MODEL + CHATBOT CORPUS
# -------------------------------

model = joblib.load("models/Gradient.pkl")
loan_chatbot_corpus = joblib.load("chatbot/loan_chatbot_corpus.pkl")
  # your saved dictionary corpus

# -------------------------------
#  SIDEBAR OPTIONS
# -------------------------------
st.sidebar.title("Menu")
option = st.sidebar.radio(
    "Select an option:",
    ["Loan Prediction", "Loan Chatbot"]
)

# =======================================================================
#                     OPTION 1: LOAN PREDICTION
# =======================================================================
if option == "Loan Prediction":

    st.title("Loan Approval Prediction")

    # Encodings
    gender_encoding = {"female": 0, "male": 1}
    education_encoding = {"Associate": 0, "Bachelor": 1, "Doctorate": 2, "High School": 3, "Master": 4}
    home_encoding = {"MORTGAGE": 0, "OTHER": 1, "OWN": 2, "RENT": 3}
    intent_encoding = {
        "DEBTCONSOLIDATION": 0, "EDUCATION": 1, "HOMEIMPROVEMENT": 2,
        "MEDICAL": 3, "PERSONAL": 4, "VENTURE": 5
    }
    defaults_encoding = {"No": 0, "Yes": 1}

    # Inputs
    person_age = st.number_input("person_age", value=30)

    person_gender_label = st.radio("person_gender", options=["female", "male"])
    person_gender = gender_encoding[person_gender_label]

    person_education_label = st.radio(
        "person_education",
        options=["Associate", "Bachelor", "Doctorate", "High School", "Master"]
    )
    person_education = education_encoding[person_education_label]

    person_income = st.number_input("person_income", value=50000)
    person_emp_exp = st.number_input("person_emp_exp", value=3)

    person_home_label = st.radio("person_home_ownership", options=["MORTGAGE", "OTHER", "OWN", "RENT"])
    person_home_ownership = home_encoding[person_home_label]

    loan_amnt = st.number_input("loan_amnt", value=10000)

    loan_intent_label = st.radio(
        "loan_intent",
        options=["DEBTCONSOLIDATION", "EDUCATION", "HOMEIMPROVEMENT", "MEDICAL", "PERSONAL", "VENTURE"]
    )
    loan_intent = intent_encoding[loan_intent_label]

    loan_int_rate = st.number_input("loan_int_rate", value=10.0, format="%.3f")
    loan_percent_income = st.number_input("loan_percent_income", value=0.2, format="%.3f")
    cb_person_cred_hist_length = st.number_input("cb_person_cred_hist_length", value=5)
    credit_score = st.number_input("credit_score", value=650)

    previous_loan_label = st.radio("previous_loan_defaults_on_file", options=["No", "Yes"])
    previous_loan_defaults_on_file = defaults_encoding[previous_loan_label]

    # Create feature array
    features = np.array([[
        person_age,
        person_gender,
        person_education,
        person_income,
        person_emp_exp,
        person_home_ownership,
        loan_amnt,
        loan_intent,
        loan_int_rate,
        loan_percent_income,
        cb_person_cred_hist_length,
        credit_score,
        previous_loan_defaults_on_file
    ]], dtype=float)

    # Predict
    if st.button("Predict"):
        pred = model.predict(features)[0]

        if pred == 1:
            st.success("Loan Approved (1)")
        else:
            st.error("Loan Not Approved (0)")

# =======================================================================
#                     OPTION 2: CHATBOT
# =======================================================================
else:
    st.title("Loan Project Chatbot 🤖")
    st.write("Ask me anything about dataset, encoding, models, or the project.")

    user_q = st.text_input("Ask your question:")

    if st.button("Chat"):
        user_q_lower = user_q.lower()
        reply = "Sorry, I didn't understand. Try asking about dataset, models, encoding, or accuracy."

        for key in loan_chatbot_corpus:
            if key in user_q_lower:
                reply = loan_chatbot_corpus[key]
                break

        st.write("**Bot:**", reply)
