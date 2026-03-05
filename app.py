import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------

st.set_page_config(
    page_title="Car MPG Enterprise Dashboard",
    page_icon="🚗",
    layout="wide"
)

st.title("🚗 Car MPG Enterprise Analytics Dashboard")
st.markdown("### Linear vs Ridge vs Lasso Regression Comparison")

# ---------------------------------------------------
# LOAD DATA
# ---------------------------------------------------

@st.cache_data
def load_data():
    data = pd.read_csv("car-mpg.csv")
    data = data.drop(['car_name'], axis=1)

    data = data.replace('?', np.nan)
    data = data.apply(pd.to_numeric, errors='coerce')

    data['origin'] = data['origin'].replace({
        1: 'america',
        2: 'europe',
        3: 'asia'
    })

    data = pd.get_dummies(data, columns=['origin'])
    data = data.fillna(data.median(numeric_only=True))

    return data


data = load_data()

# ---------------------------------------------------
# PREPROCESSING
# ---------------------------------------------------

@st.cache_resource
def prepare_data(data):

    X = data.drop("mpg", axis=1)
    y = data["mpg"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled,
        y,
        test_size=0.30,
        random_state=42
    )

    return X, y, X_train, X_test, y_train, y_test, scaler


X, y, X_train, X_test, y_train, y_test, scaler = prepare_data(data)

# ---------------------------------------------------
# SIDEBAR CONTROLS
# ---------------------------------------------------

st.sidebar.header("⚙ Model Settings")

alpha = st.sidebar.slider(
    "Regularization Strength (Alpha)",
    0.01,
    5.0,
    0.3
)

st.sidebar.markdown("---")
st.sidebar.header("🚗 Car Details")

cyl = st.sidebar.number_input("Cylinders", 3, 12, 4)

disp = st.sidebar.number_input(
    "Displacement",
    50.0,
    500.0,
    150.0
)

hp = st.sidebar.number_input(
    "Horsepower",
    40.0,
    300.0,
    100.0
)

wt = st.sidebar.number_input(
    "Weight",
    1500.0,
    5000.0,
    2500.0
)

acc = st.sidebar.number_input(
    "Acceleration",
    8.0,
    25.0,
    15.0
)

yr = st.sidebar.number_input(
    "Model Year",
    1970,
    2026,
    2020
)

origin = st.sidebar.selectbox(
    "Origin",
    ["america", "europe", "asia"]
)

predict_btn = st.sidebar.button("Predict MPG")

# ---------------------------------------------------
# MODEL TRAINING
# ---------------------------------------------------

@st.cache_resource
def train_models(alpha):

    linear = LinearRegression()
    ridge = Ridge(alpha=alpha)
    lasso = Lasso(alpha=alpha, max_iter=5000)

    linear.fit(X_train, y_train)
    ridge.fit(X_train, y_train)
    lasso.fit(X_train, y_train)

    return linear, ridge, lasso


linear, ridge, lasso = train_models(alpha)

# ---------------------------------------------------
# PREDICTIONS
# ---------------------------------------------------

lin_pred = linear.predict(X_test)
ridge_pred = ridge.predict(X_test)
lasso_pred = lasso.predict(X_test)

lin_r2 = r2_score(y_test, lin_pred)
ridge_r2 = r2_score(y_test, ridge_pred)
lasso_r2 = r2_score(y_test, lasso_pred)

# ---------------------------------------------------
# KPI METRICS
# ---------------------------------------------------

st.subheader("📊 Model Performance (R² Score)")

col1, col2, col3 = st.columns(3)

col1.metric("Linear Regression", round(lin_r2, 4))
col2.metric("Ridge Regression", round(ridge_r2, 4))
col3.metric("Lasso Regression", round(lasso_r2, 4))

# ---------------------------------------------------
# R2 COMPARISON CHART
# ---------------------------------------------------

st.subheader("📈 Model Comparison")

fig1, ax1 = plt.subplots()

ax1.bar(
    ["Linear", "Ridge", "Lasso"],
    [lin_r2, ridge_r2, lasso_r2]
)

ax1.set_ylabel("R² Score")
ax1.set_ylim(0, 1)

st.pyplot(fig1)

# ---------------------------------------------------
# ALPHA IMPACT
# ---------------------------------------------------

@st.cache_data
def compute_alpha():

    alphas = np.linspace(0.01, 5, 20)

    ridge_scores = []
    lasso_scores = []

    for a in alphas:

        r = Ridge(alpha=a).fit(X_train, y_train)
        l = Lasso(alpha=a, max_iter=5000).fit(X_train, y_train)

        ridge_scores.append(r2_score(y_test, r.predict(X_test)))
        lasso_scores.append(r2_score(y_test, l.predict(X_test)))

    return alphas, ridge_scores, lasso_scores


alphas, ridge_scores, lasso_scores = compute_alpha()

st.subheader("📉 Alpha Impact on Model Performance")

fig2, ax2 = plt.subplots()

ax2.plot(alphas, ridge_scores, label="Ridge")
ax2.plot(alphas, lasso_scores, label="Lasso")

ax2.set_xlabel("Alpha")
ax2.set_ylabel("R² Score")

ax2.legend()

st.pyplot(fig2)

# ---------------------------------------------------
# COEFFICIENT COMPARISON
# ---------------------------------------------------

st.subheader("📌 Feature Coefficient Comparison")

coef_df = pd.DataFrame({

    "Feature": X.columns,
    "Linear": linear.coef_,
    "Ridge": ridge.coef_,
    "Lasso": lasso.coef_

}).set_index("Feature")

fig3, ax3 = plt.subplots(figsize=(10, 5))

coef_df.plot(kind="bar", ax=ax3)

st.pyplot(fig3)

# ---------------------------------------------------
# PREDICTION
# ---------------------------------------------------

if predict_btn:

    input_df = pd.DataFrame([{

        "cyl": cyl,
        "disp": disp,
        "hp": hp,
        "wt": wt,
        "acc": acc,
        "yr": yr,

        "origin_america": 1 if origin == "america" else 0,
        "origin_europe": 1 if origin == "europe" else 0,
        "origin_asia": 1 if origin == "asia" else 0

    }])

    input_df = input_df[X.columns]

    input_scaled = scaler.transform(input_df)

    prediction = ridge.predict(input_scaled)

    st.success(f"🚗 Estimated MPG: {round(prediction[0],2)}")

# ---------------------------------------------------
# RESIDUAL ANALYSIS
# ---------------------------------------------------

st.subheader("📉 Residual Analysis (Ridge Model)")

residuals = y_test - ridge_pred

fig4, ax4 = plt.subplots()

ax4.scatter(y_test, residuals)

ax4.axhline(y=0)

ax4.set_xlabel("Actual MPG")
ax4.set_ylabel("Residuals")

st.pyplot(fig4)
