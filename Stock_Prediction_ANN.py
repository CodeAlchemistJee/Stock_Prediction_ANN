import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

def fetch_data(ticker, start_date, end_date):
    print(f"Fetching live data for {ticker}...")
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    return stock_data[['Close']].values

def create_dataset(dataset, window_size=60):
    X, y = [], []
    for i in range(window_size, len(dataset)):
        X.append(dataset[i-window_size:i, 0])
        y.append(dataset[i, 0])
    return np.array(X), np.array(y)

def build_model(input_shape):
    print("Building the Neural Network Architecture...")
    model = Sequential()
    # Input & First Hidden Layer
    model.add(Dense(64, activation='relu', input_dim=input_shape))
    # Dropout layer prevents overfitting by randomly turning off neurons during training
    model.add(Dropout(0.2))
    # Second Hidden Layer
    model.add(Dense(32, activation='relu'))
    # Output Layer (Predicts a single continuous value: Price)
    model.add(Dense(1))
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def main():
    # 1. Fetch & Preprocess Data
    data = fetch_data('AAPL', '2020-01-01', '2023-12-31')
    
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    
    window_size = 60
    X, y = create_dataset(scaled_data, window_size)
    
    # 2. Train / Test Split (80/20)
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    # 3. Build & Train Model
    model = build_model(window_size)
    print("Training the model (this may take a moment)...")
    model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test), verbose=1)
    
    # 4. Make Predictions
    print("Evaluating Model Performance...")
    predicted_scaled = model.predict(X_test)
    
    # Inverse transform to convert back to USD ($)
    predicted_prices = scaler.inverse_transform(predicted_scaled)
    actual_prices = scaler.inverse_transform(y_test.reshape(-1, 1))
    
    # 5. Evaluate Metrics
    rmse = np.sqrt(mean_squared_error(actual_prices, predicted_prices))
    mape = mean_absolute_percentage_error(actual_prices, predicted_prices) * 100
    
    print("-" * 30)
    print(f"RESULTS:")
    print(f"Root Mean Squared Error (RMSE): ${rmse:.2f}")
    print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
    print("-" * 30)
    
    # 6. Visualize
    plt.figure(figsize=(12,6))
    plt.plot(actual_prices, color='black', label='Actual AAPL Price', linewidth=2)
    plt.plot(predicted_prices, color='green', label='ANN Predicted Price', linewidth=1.5, alpha=0.8)
    plt.title('Stock Price Prediction using Feedforward ANN')
    plt.xlabel('Time (Testing Data Period)')
    plt.ylabel('Price (USD)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

if __name__ == "__main__":
    main()