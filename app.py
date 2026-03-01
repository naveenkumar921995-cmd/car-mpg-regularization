import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import seaborn as sns

# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.metric("Linear Regression R²", round(lin_r2,4))
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.metric("Ridge Regression R²", round(ridge_r2,4))
    st.markdown('</div>', unsafe_allow_html=True)

with col3:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.metric("Lasso Regression R²", round(lasso_r2,4))
    st.markdown('</div>', unsafe_allow_html=True)
st.title("🚗 Car MPG Enterprise Analytics Dashboard")
st.markdown("### Comparative Analysis of Linear, Ridge & Lasso Regression")

# -------------------------------------------------
# LOAD DATA
# -------------------------------------------------
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

X = data.drop("mpg", axis=1)
y = data["mpg"]

# Scaling
scaler_X = StandardScaler()
X_scaled = scaler_X.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.30, random_state=42
)

# -------------------------------------------------
# SIDEBAR
# -------------------------------------------------
st.sidebar.header("⚙ Model Controls")
alpha = st.sidebar.slider("Alpha (Regularization Strength)", 0.01, 10.0, 0.3)

# -------------------------------------------------
# TRAIN MODELS
# -------------------------------------------------
linear = LinearRegression()
ridge = Ridge(alpha=alpha)
lasso = Lasso(alpha=alpha, max_iter=10000)

linear.fit(X_train, y_train)
ridge.fit(X_train, y_train)
lasso.fit(X_train, y_train)

# Predictions
lin_pred = linear.predict(X_test)
ridge_pred = ridge.predict(X_test)
lasso_pred = lasso.predict(X_test)

# R2 Scores
lin_r2 = r2_score(y_test, lin_pred)
ridge_r2 = r2_score(y_test, ridge_pred)
lasso_r2 = r2_score(y_test, lasso_pred)

# -------------------------------------------------
# KPI SECTION
# -------------------------------------------------
col1, col2, col3 = st.columns(3)

col1.metric("Linear Regression R²", round(lin_r2,4))
col2.metric("Ridge Regression R²", round(ridge_r2,4))
col3.metric("Lasso Regression R²", round(lasso_r2,4))

st.markdown("---")

# -------------------------------------------------
# R2 COMPARISON CHART
# -------------------------------------------------
st.subheader("📊 Model Performance Comparison")

r2_df = pd.DataFrame({
    "Model": ["Linear", "Ridge", "Lasso"],
    "R2 Score": [lin_r2, ridge_r2, lasso_r2]
})

fig1, ax1 = plt.subplots(figsize=(8,5))
ax1.bar(r2_df["Model"], r2_df["R2 Score"])
ax1.set_ylim(0,1)
ax1.set_ylabel("R² Score")
st.pyplot(fig1)

# -------------------------------------------------
# ALPHA IMPACT VISUALIZATION
# -------------------------------------------------
st.subheader("📈 Impact of Alpha on Ridge & Lasso")

alphas = np.linspace(0.01, 5, 50)
ridge_scores = []
lasso_scores = []

for a in alphas:
    ridge_temp = Ridge(alpha=a)
    lasso_temp = Lasso(alpha=a, max_iter=10000)
    ridge_temp.fit(X_train, y_train)
    lasso_temp.fit(X_train, y_train)
    ridge_scores.append(r2_score(y_test, ridge_temp.predict(X_test)))
    lasso_scores.append(r2_score(y_test, lasso_temp.predict(X_test)))

fig2, ax2 = plt.subplots(figsize=(8,5))
ax2.plot(alphas, ridge_scores, label="Ridge")
ax2.plot(alphas, lasso_scores, label="Lasso")
ax2.set_xlabel("Alpha")
ax2.set_ylabel("R² Score")
ax2.legend()
st.pyplot(fig2)

# -------------------------------------------------
# COEFFICIENT ANALYSIS
# -------------------------------------------------
st.subheader("📌 Coefficient Shrinkage Analysis")

coef_df = pd.DataFrame({
    "Feature": X.columns,
    "Linear": linear.coef_,
    "Ridge": ridge.coef_,
    "Lasso": lasso.coef_
})

coef_df = coef_df.set_index("Feature")

fig3, ax3 = plt.subplots(figsize=(10,6))
coef_df.plot(kind="bar", ax=ax3)
st.pyplot(fig3)

# -------------------------------------------------
# PREDICTION MODULE
# -------------------------------------------------
st.markdown("---")
st.subheader("🔮 Predict MPG")

input_data = {}
cols = st.columns(2)

with cols[0]:
    input_data["cyl"] = st.number_input("Cylinders", 3, 12, 4)
    input_data["disp"] = st.number_input("Displacement", 50.0, 500.0, 150.0)
    input_data["hp"] = st.number_input("Horsepower", 40.0, 300.0, 100.0)
    input_data["wt"] = st.number_input("Weight", 1500.0, 5000.0, 2500.0)

with cols[1]:
    input_data["acc"] = st.number_input("Acceleration", 8.0, 25.0, 15.0)
    input_data["yr"] = st.number_input("Model Year", 70, 82, 76)
    origin = st.selectbox("Origin", ["america", "europe", "asia"])

if st.button("Predict MPG"):
    input_df = pd.DataFrame([input_data])
    input_df["origin_america"] = 1 if origin=="america" else 0
    input_df["origin_europe"] = 1 if origin=="europe" else 0
    input_df["origin_asia"] = 1 if origin=="asia" else 0
    input_df = input_df[X.columns]
    input_scaled = scaler_X.transform(input_df)
    prediction = ridge.predict(input_scaled)
    st.success(f"🚗 Estimated MPG: {round(prediction[0],2)}")

# -------------------------------------------------
# RESIDUAL PLOT
# -------------------------------------------------
st.subheader("📉 Residual Analysis (Ridge Model)")

fig4, ax4 = plt.subplots(figsize=(8,5))
sns.residplot(x=y_test, y=ridge_pred, ax=ax4)
ax4.set_xlabel("Actual MPG")
ax4.set_ylabel("Residuals")
st.pyplot(fig4)
