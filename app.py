import streamlit as st
import pandas as pd
import pickle
from datetime import datetime

# Page configuration
st.set_page_config(page_title="Car Price Predictor", page_icon="🚗", layout="centered")

st.title("🚗 Car Price Prediction App")
st.write("Predict the **resale price of a car** based on its features using Machine Learning.")

# Load trained model
model = pickle.load(open("car_price_model.pkl", "rb"))

# Sidebar
st.sidebar.header("About Project")
st.sidebar.write("""
This Machine Learning app predicts the **resale price of a car**.

Model Used:
Linear Regression

Built With:
Python • Pandas • Scikit-Learn • Streamlit
""")

st.sidebar.success("Model Accuracy (R² Score): 0.89")

# User Inputs
st.subheader("Enter Car Details")

present_price = st.number_input("Present Price (Lakhs)", min_value=0.0, step=0.5)

kms_driven = st.number_input("Kilometers Driven", min_value=0)

year = st.selectbox(
    "Model Year",
    list(range(2000, 2027))
)

fuel_type = st.selectbox("Fuel Type", ["Petrol", "Diesel", "CNG"])

seller_type = st.selectbox("Seller Type", ["Dealer", "Individual"])

transmission = st.selectbox("Transmission", ["Manual", "Automatic"])

owner = st.selectbox("Number of Previous Owners", [0,1,2,3])

# Feature Encoding
fuel_petrol = 1 if fuel_type == "Petrol" else 0
fuel_diesel = 1 if fuel_type == "Diesel" else 0

seller_individual = 1 if seller_type == "Individual" else 0

transmission_manual = 1 if transmission == "Manual" else 0

car_age = datetime.now().year - year

# Prediction Button
if st.button("Predict Car Price"):

    input_data = [[
        present_price,
        kms_driven,
        owner,
        car_age,
        fuel_diesel,
        fuel_petrol,
        seller_individual,
        transmission_manual
    ]]

    prediction = model.predict(input_data)

    st.success(f"💰 Estimated Resale Price: ₹ {round(prediction[0],2)} Lakhs")

    # Save prediction history
    history = pd.DataFrame({
        "Present Price": [present_price],
        "Kms Driven": [kms_driven],
        "Year": [year],
        "Fuel": [fuel_type],
        "Seller": [seller_type],
        "Transmission": [transmission],
        "Owner": [owner],
        "Predicted Price": [round(prediction[0],2)]
    })

    if "history" not in st.session_state:
        st.session_state.history = history
    else:
        st.session_state.history = pd.concat([st.session_state.history, history])

# Show History
if "history" in st.session_state:
    st.subheader("📊 Prediction History")
    st.dataframe(st.session_state.history)

# Footer
st.markdown("---")
st.write("Developed by **Naveen Kumar** | Machine Learning Project")
