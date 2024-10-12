import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import datetime
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
import os

# List of Nifty-50 tickers
nifty50_tickers = [
    "ADANIENT", "ADANIPORTS", "ASIANPAINT", "AXISBANK", "BAJAJ-AUTO", 
    "BAJFINANCE", "BAJAJFINSV", "BHARTIARTL", "BPCL", "BRITANNIA",
    "CIPLA", "COALINDIA", "DIVISLAB", "DRREDDY", "EICHERMOT", 
    "GRASIM", "HCLTECH", "HDFC", "HDFCBANK", "HDFCLIFE",
    "HEROMOTOCO", "HINDALCO", "HINDUNILVR", "ICICIBANK", "INDUSINDBK", 
    "INFY", "ITC", "JSWSTEEL", "KOTAKBANK", "LT", 
    "M&M", "MARUTI", "NESTLEIND", "NTPC", "ONGC", 
    "POWERGRID", "RELIANCE", "SBILIFE", "SBIN", "SUNPHARMA", 
    "TATACONSUM", "TATAMOTORS", "TATASTEEL", "TCS", "TECHM", 
    "TITAN", "ULTRACEMCO", "UPL", "WIPRO"
]

st.title("Nifty-50 Stock Price Prediction with Moving Averages")

# Select a stock ticker from dropdown
ticker = st.selectbox("Select a stock ticker:", nifty50_tickers)

# Function to fetch historical stock data and preprocess it
def fetch_data(ticker):
    today = datetime.date.today()
    data = yf.download(ticker + ".NS", start='2000-01-01', end=today)
    return data

# Load the model for the selected ticker
def load_trained_model(ticker):
    model_path = os.path.join('Keras_models', f'Latest_stock_price_model_{ticker}.keras')
    model = load_model(model_path)
    return model

# Fetch stock data for the selected ticker
data = fetch_data(ticker)

today = datetime.date.today()

if not data.empty:
    st.write(f"Displaying stock data for {ticker}")

    # Calculate 100-day and 250-day moving averages
    data['100MA'] = data['Close'].rolling(window=100).mean()
    data['250MA'] = data['Close'].rolling(window=250).mean()

    # Close price chart
    st.subheader(f"{ticker} - Closing Price Chart")
    st.line_chart(data['Close'])

    # 100MA chart
    st.subheader(f"{ticker} - 100-Day Moving Average")
    st.line_chart(data['100MA'])

    # 250MA chart
    st.subheader(f"{ticker} - 250-Day Moving Average")
    st.line_chart(data['250MA'])

    # Preprocess the close price data for prediction
    Close_price = data[["Close"]]
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(Close_price)

    # Prepare the last 100 days of data to make a prediction
    last_100_days = scaled_data[-100:].reshape(1, 100, 1)

    # Load the trained model
    model = load_trained_model(ticker)

    # Make a prediction
    prediction = model.predict(last_100_days)
    
    # Inverse scale the prediction to get the original price
    predicted_price = scaler.inverse_transform(prediction)[0][0]

    # Display the predicted stock price
    st.write(f"Predicted closing price for {ticker} of {today} is: ₹{predicted_price:.2f}")
else:
    st.write(f"No data available for {ticker}")
