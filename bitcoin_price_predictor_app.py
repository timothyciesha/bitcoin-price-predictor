import requests
import pandas as pd
from datetime import datetime, timedelta
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt

# Function to fetch historical Bitcoin data
def get_bitcoin_data(days):
    base_url = "https://api.coingecko.com/api/v3"
    endpoint = f"/coins/bitcoin/market_chart"
    params = {"vs_currency": "usd", "days": days, "interval": "daily"}
    response = requests.get(base_url + endpoint, params=params)
    if response.status_code != 200:
        st.error(f"API request failed with status code {response.status_code}: {response.text}")
        return pd.DataFrame()  # Return empty DataFrame to avoid crashing
    data = response.json()
    st.write("API Response Structure:", data.keys())  # Debug: Show available keys
    if 'prices' not in data or not data['prices']:
        st.error("No 'prices' data found in API response!")
        return pd.DataFrame()
    prices = data["prices"]  # Ensure this line is correct
    df = pd.DataFrame(prices, columns=["timestamp", "price"])
    if df.empty or df["timestamp"].isnull().all() or df["price"].isnull().all():
        st.error("DataFrame construction failed - empty or invalid data!")
        return pd.DataFrame()
    df["date"] = pd.to_datetime(df["timestamp"], unit="ms")
    df["price"] = df["price"].astype(float)
    df = df[["date", "price"]]
    return df

# Function to prepare data with moving average
def prepare_data(df, lookback=7):
    for i in range(1, lookback + 1):
        df[f"price_lag_{i}"] = df["price"].shift(i)
    df["moving_avg_14"] = df["price"].rolling(window=14).mean()
    df["next_price"] = df["price"].shift(-1)
    df = df.dropna()
    return df

# Streamlit app
st.title("Bitcoin Price Predictor")

# Sidebar for user input
st.sidebar.header("Settings")
days = st.sidebar.slider("Days of Historical Data", 30, 365, 365)
lookback = st.sidebar.slider("Lookback Period (days)", 1, 14, 7)
custom_date = st.sidebar.checkbox("Set Custom Prediction Date", value=False)
if custom_date:
    pred_date = st.sidebar.date_input("Select Prediction Date", value=datetime.now() + timedelta(days=1))
else:
    pred_date = None

# Fetch and prepare data
btc_data = get_bitcoin_data(days)
btc_prepared = prepare_data(btc_data, lookback)

# Split features and target
X = btc_prepared[[f"price_lag_{i}" for i in range(1, lookback + 1)] + ["moving_avg_14"]]
y = btc_prepared["next_price"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest model
rf_model = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)
rf_model.fit(X_train, y_train)

# Predict on test set
y_pred = rf_model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

# Predict the next day's price
latest_date = btc_prepared["date"].iloc[-1].date()  # Convert Timestamp to date
if custom_date and pred_date:
    days_ahead = (pred_date - latest_date).days
    if days_ahead < 0:
        st.error("Prediction date must be after the latest data date!")
    else:
        # For simplicity, we'll use the latest data to predict; for future dates, you'd need a time series model
        next_day_pred = rf_model.predict(btc_prepared.tail(1)[[f"price_lag_{i}" for i in range(1, lookback + 1)] + ["moving_avg_14"]])[0]
else:
    days_ahead = 1
    next_day_pred = rf_model.predict(btc_prepared.tail(1)[[f"price_lag_{i}" for i in range(1, lookback + 1)] + ["moving_avg_14"]])[0]

# Display results
st.write(f"**Root Mean Squared Error: ${rmse:.2f}**")
if custom_date and pred_date:
    st.write(f"**Predicted Price for {pred_date:%Y-%m-%d}: ${next_day_pred:.2f}** (Note: Based on latest data, not future trends)")
else:
    st.write(f"**Predicted Price for {latest_date + timedelta(days=days_ahead):%Y-%m-%d}: ${next_day_pred:.2f}**")

# Plot
st.subheader("Actual vs Predicted Prices (Test Set)")
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(y_test.values[:50], label="Actual Price", marker="o")
ax.plot(y_pred[:50], label="Predicted Price", marker="x")
ax.set_title("Actual vs Predicted Bitcoin Prices (Test Set)")
ax.set_xlabel("Days")
ax.set_ylabel("Price (USD)")
ax.legend()
st.pyplot(fig)