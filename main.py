import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression  # Example model
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt  # Optional for visualization
import streamlit as st
import google.generativeai as genai
import os

# Data (Replace with  data acquisition and preprocessing logic)
data = pd.read_csv(r"diabetes.csv")
st.set_page_config(page_title="Diabetes")

st.header("Diabetes Prediction")

# Feature selection (Replace with your analysis and selection)


# Target variable (Replace with the actual target variable name)
target_variable = "Outcome"

# Features (Replace with the actual feature names)
features = list(data.columns)
features.remove(target_variable)

X = data[features]  # Features DataFrame
y = data[target_variable]  # Target variable Series

# Model training and evaluation
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Example model: Logistic Regression (Choose a suitable model based on your data)
model = LogisticRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate model performance
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1)

# Visualize feature importance
if isinstance(model, LogisticRegression):
    coefficients = model.coef_.flatten()
    feature_names = X.columns
    plt.bar(feature_names, coefficients)
    plt.xlabel("Features")
    plt.ylabel("Coefficients")
    plt.title("Feature Importance")
    plt.show()

x = ""
y = ""

# form creation

with st.form("my_form"):
    preg = st.number_input("Enter the number of pregnancies: ")
    glu = st.number_input("Enter the plasma glucose concentration (mg/dL): ")
    pr = st.number_input("Enter the diastolic blood pressure (mmHg): ")
    thick = st.number_input("Enter the triceps skinfold thickness (mm): ")
    insu = st.number_input("Enter the insulin level (uU/mL): ")
    bmi = st.number_input("Enter the body mass index (kg/m^2): ")
    fn = st.number_input("Enter the diabetes pedigree function: ")
    age = st.number_input("Enter your age (years): ")

    if st.form_submit_button("Predict"):

        features = {
            "Pregnancies": preg,
            "Glucose": glu,
            "BloodPressure": pr,
            "SkinThickness": thick,
            "Insulin": insu,
            "BMI": bmi,
            "DiabetesPedigreeFunction": fn,
            "Age": age
        }

        new_data_df = pd.DataFrame([features])  # Convert to DataFrame for prediction
        prediction = model.predict(new_data_df)[0]
        if prediction == 1:
            x = "You are predicted to have diabetes. Please consult a doctor for diagnosis and treatment."
            st.markdown(x)
            x = x+"\nAnd the factors for diabetes are as follows: "+"\nnumber of pregnancies: "+str(preg)+ "\nplasma glucose concentration (mg/dL): "+str(glu)+"\ndiastolic blood pressure (mmHg):"+str(pr)+"\ntriceps skinfold thickness (mm): "+str(thick)+"\nthe insulin level (uU/mL): "+str(insu)+"\nthe body mass index (kg/m^2): "+str(bmi)+"\nthe diabetes pedigree function: "+str(fn)+"\nage (years):"+str(age)
        else:
            y = "You are not predicted to have diabetes based on this model. However, this is not a substitute for professional medical advice. It's important to consult a doctor for a complete evaluation."
            st.markdown(y)

# gemini model initialize

genai.configure(api_key="AIzaSyDPy5RrrNDU7Quge5JSsoiAqa4XhwX2aqo")
model = genai.GenerativeModel('gemini-pro')

if "chat" not in st.session_state:
    st.session_state.chat = model.start_chat(history=[])

st.title("Mitigation Measures")

# model response - mitigation measures

if prompt := x:
    new_prompt = prompt + "\nYou should analyze the prediction and provide mitigation measures"
    response = st.session_state.chat.send_message(new_prompt)
    st.markdown(response.text)



