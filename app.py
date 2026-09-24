import streamlit as st
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Page configuration
st.set_page_config(
    page_title="Car Purchase Prediction",
    page_icon="🚗"
)

# Title
st.title("🚗 Car Purchase Prediction")
st.write("Predict whether a customer is likely to purchase a car using Logistic Regression.")

# Dataset
data = {
    "Age": [22, 25, 28, 30, 32, 35, 38, 40, 45, 50],
    "EstimatedSalary": [
        25000, 30000, 40000, 50000, 55000,
        60000, 65000, 70000, 80000, 90000
    ],
    "Purchased": [0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
}

df = pd.DataFrame(data)

# Input and output
X = df[["Age", "EstimatedSalary"]]
y = df["Purchased"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create and train model
model = LogisticRegression()
model.fit(X_train, y_train)

# Accuracy
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# User input
st.subheader("Enter Customer Details")

age = st.number_input(
    "Age",
    min_value=18,
    max_value=100,
    value=35
)

salary = st.number_input(
    "Estimated Salary",
    min_value=10000,
    max_value=1000000,
    value=60000,
    step=5000
)

# Prediction button
if st.button("Predict Purchase"):

    prediction = model.predict([[age, salary]])

    st.subheader("Prediction Result")

    if prediction[0] == 1:
        st.success("Customer is likely to purchase a car.")
    else:
        st.warning("Customer is unlikely to purchase a car.")

    st.info(f"Model Accuracy: {accuracy * 100:.2f}%")

# Project information
st.divider()

st.subheader("About the Project")

st.write(
    "This project uses Logistic Regression, a supervised "
    "machine learning algorithm, to predict car purchase "
    "based on customer age and estimated salary."
)

st.write("0 = Not Purchased")
st.write("1 = Purchased")
