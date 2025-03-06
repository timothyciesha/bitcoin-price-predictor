import requests
import pandas as pd
from datetime import datetime, timedelta
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt

# Custom CSS for styling
st.markdown("""
    <style>
    .main-title {
        font-size: 40px;
        font-weight: bold;
        color: #2c3e50;
        text-align: center;
        margin-bottom: 20px;
    }
    .metric-box {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        text-align: center;
    }
    .current-price {
        font-size: 36px;
        color: #e74c3c;
        font-weight: bold;
    }
    .predicted-price {
        font-size: 36px;
        color: #2ecc71;
        font-weight: bold;
    }
    .data-table {
        background-color: #ffffff;
        border-radius: 5px;
        padding: 10px;
        margin-top: 10px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    .stSlider > div > div > div > div {
        background-color: #3498db !important;
    }
    .stButton > button {
        background-color: #3498db;
        color: white;
        border-radius: 5px;
        padding: 10px 20px;
    }
    </style>
""", unsafe_allow_html=True)

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

# Function to fetch current price
def get_current_price(coin_id):
    base_url = "https://api.coingecko.com/api/v3"
    endpoint = "/simple/price"
    params = {"ids": coin_id, "vs_currencies": "usd"}
    response = requests.get(base_url + endpoint, params=params)
    if response.status_code == 200:
        data = response.json()
        return data[coin_id]["usd"]
    else:
        st.error(f"Failed to fetch current price for {coin_id}: {response.text}")
        return None

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
st.markdown('<div class="main-title">Crypto Price Predictor</div>', unsafe_allow_html=True)

# Sidebar for user input
st.sidebar.header("Settings")
coin_id = st.sidebar.selectbox("Select Cryptocurrency", ["bitcoin", "ethereum", "solana"], index=0)
days = st.sidebar.slider("Days of Historical Data", 30, 365, 365)
lookback = st.sidebar.slider("Lookback Period (days)", 1, 14, 7)
forecast_days = st.sidebar.slider("Days Ahead to Predict", 1, 7, 1)
custom_date = st.sidebar.checkbox("Set Custom Prediction Date", value=False)
if custom_date:
    pred_date = st.sidebar.date_input("Select Prediction Date", value=datetime.now() + timedelta(days=1))
else:
    pred_date = None

# Initialize session state for current price
if 'current_price' not in st.session_state:
    st.session_state.current_price = None

# Refresh button and fetch current price
if st.sidebar.button("Refresh Data"):
    st.session_state.current_price = get_current_price(coin_id)
    st.rerun()

# Fetch and prepare data for the selected coin
df_prices, df_market_caps, df_volumes = get_crypto_data(coin_id, days)
btc_prepared = prepare_data(df_prices, lookback)

# Define latest_date
latest_date = btc_prepared["date"].iloc[-1].date() if not btc_prepared.empty else datetime.now().date()

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

    # Predict multi-day prices
    latest_data = btc_prepared.tail(1)[[f"price_lag_{i}" for i in range(1, lookback + 1)] + ["moving_avg_14"]].copy()
    predictions = []
    for _ in range(forecast_days):
        pred = rf_model.predict(latest_data)[0]
        predictions.append(pred)
        # Shift features for next prediction (simplified)
        latest_data.iloc[0, :-1] = latest_data.iloc[0, 1:].values
        latest_data.iloc[0, -2] = pred  # Update moving average roughly
        latest_data.iloc[0, -1] = (latest_data.iloc[0, -1] * 13 + pred) / 14  # Approx moving avg

    # Prepare data for multi-day prediction plot
    historical_days = 5  # Show last 5 days of historical data
    hist_dates = btc_prepared["date"].tail(historical_days).dt.date.tolist()
    hist_prices = btc_prepared["price"].tail(historical_days).tolist()
    forecast_dates = [latest_date + timedelta(days=i) for i in range(1, forecast_days + 1)]
    all_dates = hist_dates + forecast_dates
    all_prices = hist_prices + predictions

    # Display results with layout
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"<div class='metric-box'><b>Root Mean Squared Error:</b> ${rmse:.2f}</div>", unsafe_allow_html=True)
    with col2:
        if st.session_state.current_price:
            st.markdown(f"<div class='metric-box'><b>Current Price (as of refresh):</b> <span class='current-price'>${st.session_state.current_price:.2f}</span> UTC</div>", unsafe_allow_html=True)

    # Display multi-day predictions
    if custom_date and pred_date:
        days_ahead = (pred_date - latest_date).days
        if days_ahead < 0:
            st.error("Prediction date must be after the latest data date!")
        else:
            st.markdown(f"<div class='metric-box'><b>Predicted Price for {pred_date:%Y-%m-%d}:</b> <span class='predicted-price'>${predictions[-1]:.2f}</span> (Note: Based on latest data up to {latest_date:%Y-%m-%d} UTC, not a true forecast)</div>", unsafe_allow_html=True)
    else:
        for i, pred in enumerate(predictions, 1):
            st.markdown(f"<div class='metric-box'><b>Predicted Price for {latest_date + timedelta(days=i):%Y-%m-%d}:</b> <span class='predicted-price'>${pred:.2f}</span></div>", unsafe_allow_html=True)
            if i == 1:  # Only show the first day's note
                st.markdown(f"<div class='metric-box' style='font-size: 12px;'>(Note: Based on latest data up to {latest_date:%Y-%m-%d} UTC, accuracy decreases for further days)</div>", unsafe_allow_html=True)

    # Multi-day prediction plot with enhanced styling
    st.subheader("Price Forecast Trend")
    fig, ax = plt.subplots(figsize=(10, 6), facecolor='#f8f9fa')
    ax.set_facecolor('#ffffff')
    # Historical prices
    ax.plot(range(len(hist_dates)), hist_prices, label="Historical Price", marker="o", markersize=8, linewidth=2, color="#1f77b4")
    # Predicted prices (start from the last historical point)
    ax.plot(range(len(hist_dates) - 1, len(all_dates)), all_prices[len(hist_dates) - 1:], label="Predicted Price", marker="^", markersize=8, linestyle="--", linewidth=2, color="#27ae60")
    # Add annotations for the last historical and final predicted points
    ax.annotate(f"${hist_prices[-1]:.2f}", (len(hist_dates) - 1, hist_prices[-1]), textcoords="offset points", xytext=(0, 10), ha='center', fontsize=10, color="#1f77b4")
    ax.annotate(f"${predictions[-1]:.2f}", (len(all_dates) - 1, predictions[-1]), textcoords="offset points", xytext=(0, 10), ha='center', fontsize=10, color="#27ae60")
    # Customize plot
    ax.set_xticks(range(len(all_dates)))
    ax.set_xticklabels([d.strftime("%Y-%m-%d") for d in all_dates], rotation=45)
    ax.set_title(f"{coin_id.capitalize()} Price Forecast (Historical + Predicted)", fontsize=14, pad=20)
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Price (USD)", fontsize=12)
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    st.pyplot(fig)

    # Display historical data and metrics
    st.subheader("Historical Data")
    st.write("**Last 5 Days of Data (UTC):**")
    st.dataframe(btc_prepared[["date", "price"]].tail(), use_container_width=True)
    st.write("**Market Caps (Last 5 Days, UTC):**")
    st.dataframe(df_market_caps.tail().set_index("date"), use_container_width=True)
    st.write("**Volumes (Last 5 Days, UTC):**")
    st.dataframe(df_volumes.tail().set_index("date"), use_container_width=True)

    # Plot for test set
    st.subheader("Actual vs Predicted Prices (Test Set)")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(y_test.values[:50], label="Actual Price", marker="o")
    ax.plot(y_pred[:50], label="Predicted Price", marker="x")
    ax.set_title(f"Actual vs Predicted {coin_id.capitalize()} Prices (Test Set)")
    ax.set_xlabel("Days")
    ax.set_ylabel("Price (USD)")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)
else:
    st.error(f"No data available for {coin_id}. Please try again later or check the API.")