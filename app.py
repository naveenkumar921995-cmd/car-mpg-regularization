import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------
st.set_page_config(
    page_title="Car MPG ML Dashboard",
    page_icon="🚗",
    layout="wide"
)

st.title("🚗 Car MPG Machine Learning Dashboard")
st.markdown("### Linear vs Ridge vs Lasso Regression Analysis")

# ---------------------------------------------------
# LOAD DATA
# ---------------------------------------------------
@st.cache_data
def load_data():

    data = pd.read_csv("car-mpg.csv")

    data = data.drop(["car_name"], axis=1)

    data = data.replace("?", np.nan)
    data = data.apply(pd.to_numeric, errors="coerce")

    data["origin"] = data["origin"].replace({
        1:"america",
        2:"europe",
        3:"asia"
    })

    data = pd.get_dummies(data, columns=["origin"])

    data = data.fillna(data.median(numeric_only=True))

    return data

data = load_data()

# ---------------------------------------------------
# CORRELATION HEATMAP (PLOTLY)
# ---------------------------------------------------
st.subheader("📊 Feature Correlation Heatmap")

corr = data.corr()

fig_corr = px.imshow(
    corr,
    text_auto=True,
    aspect="auto",
    color_continuous_scale="RdBu_r"
)

st.plotly_chart(fig_corr, use_container_width=True)

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
st.sidebar.header("⚙ Model Controls")

alpha = st.sidebar.slider(
    "Regularization Strength (Alpha)",
    0.01,
    5.0,
    0.3
)

st.sidebar.markdown("---")
st.sidebar.header("🚗 Car Input")

cyl = st.sidebar.number_input("Cylinders",3,12,4)
disp = st.sidebar.number_input("Displacement",50.0,500.0,150.0)
hp = st.sidebar.number_input("Horsepower",40.0,300.0,100.0)
wt = st.sidebar.number_input("Weight",1500.0,5000.0,2500.0)
acc = st.sidebar.number_input("Acceleration",8.0,25.0,15.0)

yr = st.sidebar.number_input("Model Year",1970,2026,2020)

origin = st.sidebar.selectbox(
    "Origin",
    ["america","europe","asia"]
)

predict_button = st.sidebar.button("Predict MPG")

# ---------------------------------------------------
# MODEL TRAINING
# ---------------------------------------------------
@st.cache_resource
def train_models(alpha):

    linear = LinearRegression()
    ridge = Ridge(alpha=alpha)
    lasso = Lasso(alpha=alpha, max_iter=5000)

    linear.fit(X_train,y_train)
    ridge.fit(X_train,y_train)
    lasso.fit(X_train,y_train)

    return linear,ridge,lasso

linear,ridge,lasso = train_models(alpha)

# ---------------------------------------------------
# MODEL PREDICTIONS
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
st.subheader("📊 Model Performance")

col1,col2,col3 = st.columns(3)

col1.metric("Linear Regression R²",round(lin_r2,4))
col2.metric("Ridge Regression R²",round(ridge_r2,4))
col3.metric("Lasso Regression R²",round(lasso_r2,4))

# ---------------------------------------------------
# MODEL COMPARISON CHART
# ---------------------------------------------------
st.subheader("📈 Model Comparison")

fig_model = px.bar(
    x=["Linear","Ridge","Lasso"],
    y=[lin_r2,ridge_r2,lasso_r2],
    labels={"x":"Model","y":"R² Score"},
    color=["Linear","Ridge","Lasso"]
)

st.plotly_chart(fig_model, use_container_width=True)

# ---------------------------------------------------
# ALPHA IMPACT ANALYSIS
# ---------------------------------------------------
@st.cache_data
def alpha_analysis():

    alphas = np.linspace(0.01,5,30)

    ridge_scores=[]
    lasso_scores=[]

    for a in alphas:

        r = Ridge(alpha=a).fit(X_train,y_train)
        l = Lasso(alpha=a,max_iter=5000).fit(X_train,y_train)

        ridge_scores.append(
            r2_score(y_test,r.predict(X_test))
        )

        lasso_scores.append(
            r2_score(y_test,l.predict(X_test))
        )

    return alphas,ridge_scores,lasso_scores

alphas,ridge_scores,lasso_scores = alpha_analysis()

st.subheader("📉 Alpha Impact on Models")

fig_alpha = go.Figure()

fig_alpha.add_trace(
    go.Scatter(
        x=alphas,
        y=ridge_scores,
        mode="lines+markers",
        name="Ridge"
    )
)

fig_alpha.add_trace(
    go.Scatter(
        x=alphas,
        y=lasso_scores,
        mode="lines+markers",
        name="Lasso"
    )
)

fig_alpha.update_layout(
    xaxis_title="Alpha",
    yaxis_title="R² Score"
)

st.plotly_chart(fig_alpha,use_container_width=True)

# ---------------------------------------------------
# FEATURE IMPORTANCE
# ---------------------------------------------------
st.subheader("🏆 Feature Importance (Ridge)")

importance = pd.DataFrame({
    "Feature":X.columns,
    "Importance":abs(ridge.coef_)
})

importance = importance.sort_values(
    by="Importance",
    ascending=True
)

fig_imp = px.bar(
    importance,
    x="Importance",
    y="Feature",
    orientation="h"
)

st.plotly_chart(fig_imp,use_container_width=True)

# ---------------------------------------------------
# ACTUAL VS PREDICTED
# ---------------------------------------------------
st.subheader("📊 Actual vs Predicted MPG")

fig_scatter = px.scatter(
    x=y_test,
    y=ridge_pred,
    labels={"x":"Actual MPG","y":"Predicted MPG"}
)

st.plotly_chart(fig_scatter,use_container_width=True)

# ---------------------------------------------------
# RESIDUAL ANALYSIS
# ---------------------------------------------------
st.subheader("📉 Residual Analysis")

residuals = y_test - ridge_pred

fig_res = px.scatter(
    x=y_test,
    y=residuals,
    labels={"x":"Actual MPG","y":"Residual"}
)

fig_res.add_hline(y=0)

st.plotly_chart(fig_res,use_container_width=True)

# ---------------------------------------------------
# SIDEBAR PREDICTION
# ---------------------------------------------------
if predict_button:

    input_data = {
        "cylinders":cyl,
        "displacement":disp,
        "horsepower":hp,
        "weight":wt,
        "acceleration":acc,
        "model_year":yr
    }

    input_df = pd.DataFrame([input_data])

    input_df["origin_america"] = 1 if origin=="america" else 0
    input_df["origin_europe"] = 1 if origin=="europe" else 0
    input_df["origin_asia"] = 1 if origin=="asia" else 0

    for col in X.columns:
        if col not in input_df.columns:
            input_df[col] = 0

    input_df = input_df[X.columns]

    input_scaled = scaler.transform(input_df)

    lin_prediction = linear.predict(input_scaled)[0]
    ridge_prediction = ridge.predict(input_scaled)[0]
    lasso_prediction = lasso.predict(input_scaled)[0]

    st.sidebar.success(
        f"🚗 Estimated MPG (Ridge): {round(ridge_prediction,2)}"
    )
