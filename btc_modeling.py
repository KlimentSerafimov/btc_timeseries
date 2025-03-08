import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.tsa.arima.model import ARIMA
import os
import pickle

def load_data(filepath='data/btc_price_data.csv'):
    """Load Bitcoin data from CSV file"""
    data = pd.read_csv(filepath, index_col=0, parse_dates=True)
    print(f"Loaded data with shape: {data.shape}")
    return data

def prepare_data_for_modeling(data, target_col='Close', test_size=0.2):
    """Prepare data for modeling by splitting into train/test sets"""
    # Get the target column
    series = data[target_col].values.reshape(-1, 1)
    
    # Scale the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(series)
    
    # Split into train and test sets
    train_size = int(len(scaled_data) * (1 - test_size))
    train_data = scaled_data[:train_size]
    test_data = scaled_data[train_size:]
    
    print(f"Training data size: {train_data.shape}")
    print(f"Testing data size: {test_data.shape}")
    
    return train_data, test_data, scaler

def build_arima_model(data, order=(5,1,0)):
    """Build and train an ARIMA model"""
    print(f"Building ARIMA model with order {order}...")
    
    # Fit the model
    model = ARIMA(data, order=order)
    model_fit = model.fit()
    
    print("ARIMA model summary:")
    print(model_fit.summary())
    
    return model_fit

def save_model(model, filepath='models/arima_model.pkl'):
    """Save the trained model to a file"""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    with open(filepath, 'wb') as f:
        pickle.dump(model, f)
    
    print(f"Model saved to {filepath}")

def evaluate_model(model, test_data, scaler, original_data, save_path='figures/model_evaluation.png'):
    """Evaluate the model and plot predictions vs actual values"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Make predictions
    predictions = model.forecast(steps=len(test_data))
    
    # Reshape predictions for inverse scaling
    predictions_reshaped = predictions.reshape(-1, 1)
    
    # Inverse transform to get actual price values
    predictions_actual = scaler.inverse_transform(predictions_reshaped)
    test_actual = scaler.inverse_transform(test_data)
    
    # Calculate error metrics
    mse = mean_squared_error(test_actual, predictions_actual)
    mae = mean_absolute_error(test_actual, predictions_actual)
    rmse = np.sqrt(mse)
    
    print(f"Model Evaluation Metrics:")
    print(f"MSE: {mse:.2f}")
    print(f"MAE: {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    
    # Plot predictions vs actual
    plt.figure(figsize=(12, 6))
    
    # Get the dates for the test period
    test_dates = original_data.index[-len(test_data):]
    
    plt.plot(test_dates, test_actual, label='Actual Prices')
    plt.plot(test_dates, predictions_actual, label='Predicted Prices', alpha=0.7)
    
    plt.title('Bitcoin Price Prediction: Actual vs Predicted')
    plt.xlabel('Date')
    plt.ylabel('Price (USD)')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()
    
    print(f"Evaluation plot saved to {save_path}")
    
    return mse, mae, rmse

if __name__ == "__main__":
    # This script is a template for future modeling
    # Uncomment and run when ready to implement predictive models
    
    """
    # Load data
    btc_data = load_data()
    
    # Prepare data for modeling
    train_data, test_data, scaler = prepare_data_for_modeling(btc_data)
    
    # Build ARIMA model
    arima_model = build_arima_model(train_data.flatten())
    
    # Save model
    save_model(arima_model)
    
    # Evaluate model
    mse, mae, rmse = evaluate_model(arima_model, test_data, scaler, btc_data)
    
    print("\nModeling complete!")
    """
    
    print("This is a template for future modeling. Uncomment the code when ready to implement predictive models.") 