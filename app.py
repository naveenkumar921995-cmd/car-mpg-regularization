import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
import math

# -------------------------------
# Page Configuration
# -------------------------------
st.set_page_config(page_title="Car MPG Prediction", layout="wide")

st.title("🚗 Car MPG Prediction using Regularization")
st.markdown("Compare **Linear, Ridge & Lasso Regression** on Car MPG Dataset")

# -------------------------------
# Load Dataset
# -------------------------------
@st.cache_data
def load_data():
    data = pd.read_csv("car-mpg.csv")
    data = data.drop(['car_name'], axis=1)
    data = data.replace('?', np.nan)
    data = data.apply(pd.to_numeric, errors='coerce')
    data['origin'] = data['origin'].replace({1: 'america', 2: 'europe', 3: 'asia'})
    data = pd.get_dummies(data, columns=['origin'])
    data = data.fillna(data.median(numeric_only=True))
    return data

data = load_data()

# -------------------------------
# Feature & Target
# -------------------------------
X = data.drop(['mpg'], axis=1)
y = data[['mpg']]

# -------------------------------
# Scaling
# -------------------------------
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_scaled, test_size=0.30, random_state=1
)

# -------------------------------
# Sidebar Controls
# -------------------------------
st.sidebar.header("⚙ Model Settings")

model_choice = st.sidebar.selectbox(
    "Choose Model",
    ("Linear Regression", "Ridge Regression", "Lasso Regression")
)

alpha = st.sidebar.slider(
    "Select Alpha (Regularization Strength)",
    min_value=0.01,
    max_value=10.0,
    value=0.3
)

# -------------------------------
# Model Selection
# -------------------------------
if model_choice == "Linear Regression":
    model = LinearRegression()
elif model_choice == "Ridge Regression":
    model = Ridge(alpha=alpha)
else:
    model = Lasso(alpha=alpha)

# -------------------------------
# Train Model
# -------------------------------
model.fit(X_train, y_train)

# Predictions
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# Metrics
train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)
mse = mean_squared_error(y_test, y_test_pred)
rmse = math.sqrt(mse)

# -------------------------------
# Display Metrics
# -------------------------------
col1, col2, col3 = st.columns(3)

col1.metric("Train R²", round(train_r2, 4))
col2.metric("Test R²", round(test_r2, 4))
col3.metric("RMSE", round(rmse, 4))

st.markdown("---")

# -------------------------------
# Coefficient Visualization
# -------------------------------
st.subheader("📊 Model Coefficients")

coef = pd.Series(model.coef_.flatten(), index=X.columns)
coef_df = coef.sort_values()

fig, ax = plt.subplots(figsize=(10,6))
coef_df.plot(kind='barh', ax=ax)
ax.set_title("Feature Coefficient Importance")
st.pyplot(fig)

# -------------------------------
# Prediction Section
# -------------------------------
st.markdown("---")
st.subheader("🔮 Predict MPG for New Car")

input_data = {}

col1, col2 = st.columns(2)

with col1:
    input_data["cyl"] = st.number_input("Cylinders", 3, 12, 4)
    input_data["disp"] = st.number_input("Displacement", 50.0, 500.0, 150.0)
    input_data["hp"] = st.number_input("Horsepower", 40.0, 300.0, 100.0)
    input_data["wt"] = st.number_input("Weight", 1500.0, 5000.0, 2500.0)

with col2:
    input_data["acc"] = st.number_input("Acceleration", 8.0, 25.0, 15.0)
    input_data["yr"] = st.number_input("Model Year", 70, 82, 76)

origin = st.selectbox("Origin", ["america", "europe", "asia"])

# Create dataframe for prediction
if st.button("Predict MPG"):

    input_df = pd.DataFrame([input_data])

    # Add origin columns
    input_df["origin_america"] = 1 if origin == "america" else 0
    input_df["origin_europe"] = 1 if origin == "europe" else 0
    input_df["origin_asia"] = 1 if origin == "asia" else 0

    # Ensure column order
    input_df = input_df[X.columns]

    # Scale input
    input_scaled = scaler_X.transform(input_df)

    # Predict
    prediction_scaled = model.predict(input_scaled)

    # Inverse transform
    prediction = scaler_y.inverse_transform(prediction_scaled)

    st.success(f"🚗 Predicted MPG: {prediction[0][0]:.2f}")

# -------------------------------
# Residual Plot
# -------------------------------
st.markdown("---")
st.subheader("📉 Residual Analysis")

fig2, ax2 = plt.subplots(figsize=(8,5))
sns.residplot(x=y_test.flatten(), y=y_test_pred.flatten(), lowess=True, ax=ax2)
ax2.set_xlabel("Actual MPG (Scaled)")
ax2.set_ylabel("Residuals")
st.pyplot(fig2)
