# Bitcoin Price Predictor

A data science project to predict Bitcoin prices using historical data from the CoinGecko API and a Random Forest model.

## Features
- Fetches 365 days of Bitcoin price data from CoinGecko.
- Engineers features: 7-day lagged prices and 14-day moving average.
- Trains a Random Forest Regressor (200 trees, max_depth=10) to predict the next day's price.
- Evaluates with RMSE: **$2,659.40**.
- Visualizes actual vs. predicted prices.

## Results
- Predicted price for 2025-03-04: **$91,160.99** (based on March 3 data of $94,261.53).
- See `prediction.txt` for details and `btc_prediction_plot.png` for the plot.

## How to Run
1. Install dependencies: `pip install requests pandas numpy scikit-learn matplotlib`
2. Run the script: `python bitcoin_price_predictor.py`

## Skills
- Data fetching (API)
- Feature engineering
- Machine learning (Random Forest)
- Visualization
- Python programming

## Future Work
- Add more cryptocurrencies (e.g., Ethereum).
- Incorporate additional features (e.g., trading volume).

## Contact
Open to web3/AI freelancing gigs! Reach me at timothyciesha@gmail.com