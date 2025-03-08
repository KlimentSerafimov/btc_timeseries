import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.tsa.arima.model import ARIMA
import os
import pickle
import matplotlib.dates as mdates

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
    
    # Set dark style for plots
    plt.style.use('dark_background')
    
    # Define colors for better visibility on dark background
    line_color1 = '#00a8ff'  # Bright blue
    line_color2 = '#00ff7f'  # Green
    title_color = '#e0e0e0'  # Light gray
    grid_color = '#555555'   # Medium gray
    
    # Plot predictions vs actual
    fig, ax = plt.subplots(figsize=(12, 6), facecolor='#1e1e1e')
    ax.set_facecolor('#2d2d2d')
    
    # Get the dates for the test period
    test_dates = original_data.index[-len(test_data):]
    
    ax.plot(test_dates, test_actual, label='Actual Prices', color=line_color1, linewidth=2)
    ax.plot(test_dates, predictions_actual, label='Predicted Prices', color=line_color2, alpha=0.7, linewidth=2)
    
    # Set title and labels
    ax.set_title('Bitcoin Price Prediction: Actual vs Predicted', color=title_color, fontsize=16)
    ax.set_xlabel('Date', color=title_color, fontsize=12)
    ax.set_ylabel('Price (USD)', color=title_color, fontsize=12)
    
    # Configure legend, grid and spines
    legend = ax.legend(facecolor='#2d2d2d', edgecolor=grid_color, labelcolor=title_color)
    ax.grid(True, alpha=0.2, color=grid_color)
    ax.tick_params(colors=title_color)
    for spine in ax.spines.values():
        spine.set_color(grid_color)
    
    # Set date ticks
    date_range = [test_dates[0], test_dates[-1]]
    ax.set_xlim(date_range)
    date_interval = (date_range[1] - date_range[0]) / 20
    locator = mdates.DayLocator(interval=max(1, int(date_interval.days)))
    ax.xaxis.set_major_locator(locator)
    
    # Rotate tick labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', color=title_color)
    
    plt.tight_layout()
    plt.savefig(save_path, facecolor='#1e1e1e')
    plt.close()
    
    # Reset to default style
    plt.style.use('default')
    
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