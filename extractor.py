import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.callbacks import EarlyStopping



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

today = datetime.date.today()
dataframes = {}
for ticker in nifty50_tickers:
    # Fetch historical data
    data = yf.download(ticker+'.NS', start='2000-01-01', end=today)
    print(ticker)    
    # Check if data is empty
    if data.empty:
        print(f"No data found for {ticker}")
        continue
    
    dataframes[ticker] = data
    df = data

    Close_price = df[["Close"]]
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(Close_price)
    
    # Prepare data for LSTM
    X_data, y_data = [], []
    for i in range(100, len(scaled_data)):
        X_data.append(scaled_data[i-100:i])
        y_data.append(scaled_data[i])

    X_data, y_data = np.array(X_data), np.array(y_data)

    # Split data into train and test sets
    X_train_val, X_test, y_train_val, y_test = train_test_split(X_data, y_data, test_size=0.2, random_state=42)
    
    # Split training + validation into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.2, random_state=42)

    # Assuming X_train, y_train, X_val, and y_val are your training and validation data and labels
    from keras import layers, models, optimizers, callbacks

    # Updated model architecture
    model = models.Sequential([
        # LSTM layer to capture temporal dependencies
        layers.LSTM(128, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        
        # Dense layers for feature learning
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        
        layers.Dense(64, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        
        layers.Dense(32, activation='relu'),
        
        # Output layer for regression
        layers.Dense(1)
    ])

    # Compile the model with Adam optimizer and learning rate scheduler
    adam_optimizer = optimizers.Adam(learning_rate=0.001)
    lr_scheduler = callbacks.ReduceLROnPlateau(monitor='val_loss', patience=3, factor=0.5, min_lr=0.0001)

    model.compile(optimizer=adam_optimizer, loss='mean_squared_error')

    # Early stopping
    early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    # Fit the model
    history = model.fit(X_train, y_train, 
                        epochs=200, 
                        batch_size=16, 
                        validation_data=(X_val, y_val), 
                        callbacks=[early_stopping, lr_scheduler])
        # Save the model
    model.save(f"Keras_models\Latest_stock_price_model_{ticker}.keras")

    print(f"Model trained and saved for {ticker}")
