 🚀 Live Demo (Streamlit App)

🔗 https://loan-approval-ml-project.streamlit.app

Loan Approval Prediction System with Chatbot

Project Overview  
This project is a Machine Learning–based Loan Approval Prediction system.  
It predicts whether a loan application will be approved or not based on applicant details.  

The project also includes a simple chatbot to answer basic questions about the project, dataset, and model used.  


Project Structure  

Loan_Approval_AI_Project/  
│  
├── data/  
│   └── loan_data.csv  
│  
├── notebooks/  
│   └── Bank_project.ipynb  
│  
├── models/  
│   └── Gradient.pkl  
│  
├── chatbot/  
│   └── loan_chatbot_corpus.pkl  
│  
├── bank_ensemble.py  
├── app.py  
├── README.md  
└── requirements.txt  

---

Files Description  

- Bank_project.ipynb  
  Used for data analysis, feature engineering, model experimentation, and creating the chatbot corpus.  

- bank_ensemble.py  
  Main Python file that loads the trained ML model and chatbot corpus for usage.  

- loan_data.csv  
  Dataset containing loan applicant information.  

- Gradient.pkl  
  Trained Gradient Boosting machine learning model.  

- loan_chatbot_corpus.pkl  
  Serialized chatbot knowledge base created in the Jupyter Notebook.  

- app.py  
  Documentation / explanation file describing the project logic.  

---

Machine Learning Algorithms Used  
- AdaBoost Classifier  
- Gradient Boosting Classifier  
- XGBoost Classifier  

Gradient Boosting was selected as the final model based on performance.  

---

How to Run the Project  

Step 1: Install dependencies  
pip install -r requirements.txt  


