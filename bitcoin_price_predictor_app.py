import requests
import pandas as pd
from datetime import datetime, timedelta
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt

# Function to fetch historical data for any cryptocurrency
def get_crypto_data(coin_id, days):
    base_url = "https://api.coingecko.com/api/v3"
    endpoint = f"/coins/{coin_id}/market_chart"
    params = {"vs_currency": "usd", "days": days, "interval": "daily"}
    response = requests.get(base_url + endpoint, params=params)
    if response.status_code != 200:
        st.error(f"API request failed with status code {response.status_code}: {response.text}")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    data = response.json()
    if 'prices' not in data or not data['prices']:
        st.error(f"No 'prices' data found for {coin_id}!")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    prices = data["prices"]
    market_caps = data["market_caps"]
    volumes = data["total_volumes"]
    
    df_prices = pd.DataFrame(prices, columns=["timestamp", "price"])
    df_market_caps = pd.DataFrame(market_caps, columns=["timestamp", "market_cap"])
    df_volumes = pd.DataFrame(volumes, columns=["timestamp", "volume"])
    
    for df in [df_prices, df_market_caps, df_volumes]:
        if df.empty or df["timestamp"].isnull().all() or (df.columns[1].lower() != "price" and df[df.columns[1]].isnull().all()):
            st.error(f"DataFrame construction failed for {coin_id} - empty or invalid data!")
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        df["date"] = pd.to_datetime(df["timestamp"], unit="ms")
        if df.columns[1].lower() == "price":
            df["price"] = df["price"].astype(float)
        elif df.columns[1].lower() == "market_cap":
            df["market_cap"] = df["market_cap"].astype(float)
        elif df.columns[1].lower() == "volume":
            df["volume"] = df["volume"].astype(float)
    
    df_prices = df_prices[["date", "price"]]
    df_market_caps = df_market_caps[["date", "market_cap"]]
    df_volumes = df_volumes[["date", "volume"]]
    
    return df_prices, df_market_caps, df_volumes

# Function to prepare data with moving average
def prepare_data(df_prices, lookback=7):
    df = df_prices.copy()
    for i in range(1, lookback + 1):
        df[f"price_lag_{i}"] = df["price"].shift(i)
    df["moving_avg_14"] = df["price"].rolling(window=14).mean()
    df["next_price"] = df["price"].shift(-1)
    df = df.dropna()
    return df

# Streamlit app
st.title("Web3 Crypto Price Predictor")

# Sidebar for user input
st.sidebar.header("Settings")
coin_id = st.sidebar.selectbox("Select Cryptocurrency", ["bitcoin", "ethereum", "solana"], index=0)
days = st.sidebar.slider("Days of Historical Data", 30, 365, 365)
lookback = st.sidebar.slider("Lookback Period (days)", 1, 14, 7)
custom_date = st.sidebar.checkbox("Set Custom Prediction Date", value=False)
if custom_date:
    pred_date = st.sidebar.date_input("Select Prediction Date", value=datetime.now() + timedelta(days=1))
else:
    pred_date = None

# Refresh button
if st.sidebar.button("Refresh Data"):
    st.rerun()

# Fetch and prepare data for the selected coin
df_prices, df_market_caps, df_volumes = get_crypto_data(coin_id, days)
btc_prepared = prepare_data(df_prices, lookback)

# Split features and target
X = btc_prepared[[f"price_lag_{i}" for i in range(1, lookback + 1)] + ["moving_avg_14"]]
y = btc_prepared["next_price"]

# Train-test split (only if data exists)
if not btc_prepared.empty:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train Random Forest model
    rf_model = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)
    rf_model.fit(X_train, y_train)

    # Predict on test set
    y_pred = rf_model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    # Predict the next day's price
    latest_date = btc_prepared["date"].iloc[-1].date()
    if custom_date and pred_date:
        days_ahead = (pred_date - latest_date).days
        if days_ahead < 0:
            st.error("Prediction date must be after the latest data date!")
        else:
            next_day_pred = rf_model.predict(btc_prepared.tail(1)[[f"price_lag_{i}" for i in range(1, lookback + 1)] + ["moving_avg_14"]])[0]
            st.markdown(f"**Predicted Price for {pred_date:%Y-%m-%d}:** <span style='font-size: 36px; color: #2ecc71;'>${next_day_pred:.2f}</span> (Note: Based on latest data up to {latest_date:%Y-%m-%d} UTC, not a true forecast for this date)", unsafe_allow_html=True)
    else:
        days_ahead = 1
        next_day_pred = rf_model.predict(btc_prepared.tail(1)[[f"price_lag_{i}" for i in range(1, lookback + 1)] + ["moving_avg_14"]])[0]

    # Display results
    st.write(f"**Root Mean Squared Error: ${rmse:.2f}**")
    if custom_date and pred_date:
        pass  # Note already displayed above
    else:
        st.markdown(f"**Predicted Price for {latest_date + timedelta(days=days_ahead):%Y-%m-%d}:** <span style='font-size: 36px; color: #2ecc71;'>${next_day_pred:.2f}</span>", unsafe_allow_html=True)
    st.write("**Last 5 Days of Data (UTC):**")
    st.dataframe(btc_prepared[["date", "price"]].tail())
    st.write("**Market Caps (Last 5 Days, UTC):**")
    st.dataframe(df_market_caps.tail().set_index("date"))
    st.write("**Volumes (Last 5 Days, UTC):**")
    st.dataframe(df_volumes.tail().set_index("date"))

    # Plot
    st.subheader("Actual vs Predicted Prices (Test Set)")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(y_test.values[:50], label="Actual Price", marker="o")
    ax.plot(y_pred[:50], label="Predicted Price", marker="x")
    ax.set_title(f"Actual vs Predicted {coin_id.capitalize()} Prices (Test Set)")
    ax.set_xlabel("Days")
    ax.set_ylabel("Price (USD)")
    ax.legend()
    st.pyplot(fig)
else:
    st.error(f"No data available for {coin_id}. Please try again later or check the API.")