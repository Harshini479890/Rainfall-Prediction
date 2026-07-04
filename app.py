import streamlit as st
import pandas as pd
import numpy as np
import random
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

st.set_page_config(page_title="Rainfall Prediction", layout="centered")

st.title("Rainfall Prediction System")
st.write("Predicting rainfall (mm) from climate parameters using ML")

# -----------------------------
# 1. Generate synthetic dataset (with added noise for realism)
# -----------------------------
random.seed(42)
np.random.seed(42)

states = ["Maharashtra", "Kerala", "Tamil Nadu", "Rajasthan", "Punjab"]
months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct"]

data = []

for state in states:
    for month_index, month in enumerate(months):
        for _ in range(20):  # increased from 4 to 20 records per month for a larger dataset
            if month in ["Jun", "Jul", "Aug", "Sep"]:
                rainfall = 150 + month_index * 5
                humidity = 75
                temp = 28
            else:
                rainfall = 20 + month_index * 2
                humidity = 45
                temp = 35

            if state == "Kerala":
                rainfall += 40
            elif state == "Rajasthan":
                rainfall -= 15

            wind_speed = 10 + month_index

            # Add random noise so the data isn't perfectly deterministic
            rainfall += random.uniform(-10, 10)
            temp += random.uniform(-2, 2)
            humidity += random.uniform(-5, 5)
            wind_speed += random.uniform(-1, 1)

            rainfall = max(rainfall, 0)  # rainfall can't be negative

            data.append([state, month, round(temp, 1), round(humidity, 1), round(wind_speed, 1), round(rainfall, 1)])

df = pd.DataFrame(
    data,
    columns=["State", "Month", "Avg_Temperature", "Humidity", "Wind_Speed", "Rainfall_mm"]
)
df.to_csv("rainfall_dataset.csv", index=False)

if st.checkbox("Show Dataset"):
    st.write(df.head(20))
    st.write("Total Records:", df.shape[0])

# -----------------------------
# 2. Encode categorical features
# -----------------------------
df_encoded = pd.get_dummies(df, columns=["State", "Month"], drop_first=True)

x = df_encoded.drop("Rainfall_mm", axis=1)
y = df_encoded["Rainfall_mm"]

# Shuffled split (fixed from previous version which used shuffle=False)
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, shuffle=True, random_state=42
)

# -----------------------------
# 3. Train multiple models
# -----------------------------
lr_model = LinearRegression()
lr_model.fit(x_train, y_train)
lr_pred = lr_model.predict(x_test)

rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(x_train, y_train)
rf_pred = rf_model.predict(x_test)

def get_metrics(y_true, y_pred):
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = np.mean(np.abs(y_true - y_pred))
    return r2, rmse, mae

lr_r2, lr_rmse, lr_mae = get_metrics(y_test, lr_pred)
rf_r2, rf_rmse, rf_mae = get_metrics(y_test, rf_pred)

# -----------------------------
# 4. Model comparison table
# -----------------------------
st.subheader("Model Comparison")

comparison_df = pd.DataFrame({
    "Model": ["Linear Regression", "Random Forest"],
    "R2 Score": [round(lr_r2, 3), round(rf_r2, 3)],
    "RMSE (mm)": [round(lr_rmse, 2), round(rf_rmse, 2)],
    "MAE (mm)": [round(lr_mae, 2), round(rf_mae, 2)]
})

st.table(comparison_df)

best_model_name = "Random Forest" if rf_r2 > lr_r2 else "Linear Regression"
best_model = rf_model if rf_r2 > lr_r2 else lr_model

st.info(f"Best performing model: **{best_model_name}**")

# -----------------------------
# 5. Feature importance (only meaningful for Random Forest / tree models)
# -----------------------------
st.subheader("Feature Importance (Random Forest)")

importance_df = pd.DataFrame({
    "Feature": x.columns,
    "Importance": rf_model.feature_importances_
}).sort_values("Importance", ascending=False)

st.bar_chart(importance_df.set_index("Feature"))

# -----------------------------
# 6. Prediction interface
# -----------------------------
st.subheader("Enter Climate Details")

model_choice = st.radio("Select Model for Prediction", ["Random Forest", "Linear Regression"])

state_input = st.selectbox("Select State", states)
month_input = st.selectbox("Select Month", months)

temperature_input = st.slider("Average Temperature (C)", 20, 45, 30)
humidity_input = st.slider("Humidity (%)", 30, 90, 60)
wind_input = st.slider("Wind Speed (km/h)", 5, 25, 12)

input_data = pd.DataFrame([{
    "State": state_input,
    "Month": month_input,
    "Avg_Temperature": temperature_input,
    "Humidity": humidity_input,
    "Wind_Speed": wind_input
}])

input_encoded = pd.get_dummies(input_data, columns=["State", "Month"])
input_encoded = input_encoded.reindex(columns=x.columns, fill_value=0)

if st.button("Predict Rainfall"):
    chosen_model = rf_model if model_choice == "Random Forest" else lr_model
    prediction = chosen_model.predict(input_encoded)[0]
    prediction = max(prediction, 0)
    st.success(f"Predicted Rainfall ({model_choice}): {prediction:.2f} mm")

st.write("---")
st.caption("Synthetic India Climate Data | Linear Regression vs Random Forest")