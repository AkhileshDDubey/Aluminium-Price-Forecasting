import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from statsmodels.tsa.statespace.sarimax import SARIMAX
import pickle
# ---------------------
# Data Loading and Preprocessing
# ---------------------
file_path = r"C:\Users\dubey\Project_244\Data Collection\Dataset_244\Aluminium_Data_cleaned.csv"  
df = pd.read_csv(file_path, parse_dates=['Date'], index_col='Date')
df = df.sort_index()
# ---------------------
# AutoEDA
# ---------------------
import sweetviz as sv

# Sweetviz Report
sweetviz_report = sv.analyze(df)
sweetviz_report.show_html("sweetviz_report.html")
# ---------------------
# Decompose the Time Series
# ---------------------
from statsmodels.tsa.seasonal import seasonal_decompose
decomposition = seasonal_decompose(df['Price'], model='additive', period=30)
plt.figure(figsize=(12, 10))
plt.subplot(411)
plt.plot(df['Price'], label='Original Price', color='blue')
plt.legend(loc='upper left')
plt.subplot(412)
plt.plot(decomposition.seasonal, label='Seasonality', color='green')
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()

# ---------------------
# Stationarity Test (ADF)
# ---------------------
from statsmodels.tsa.stattools import adfuller
adf_test = adfuller(df['Price'])
print("ADF Statistic:", adf_test[0])
print("p-value:", adf_test[1])
if adf_test[1] < 0.05:
    print("Data is stationary (No seasonality detected)")
else:
    print("Data is non-stationary (Seasonality might be present)")

# ---------------------
# Plot the Time Series Data
# ---------------------
plt.figure(figsize=(12, 6))
plt.plot(df['Price'], label='Aluminium Price', color='blue')
plt.title("Aluminium Price Over Time")
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()
plt.show()

# ---------------------
# Define a helper function to calculate RMSE and MAPE
# ---------------------
def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

# ---------------------
# Feature Engineering
# ---------------------
df['Day'] = df.index.day
df['Month'] = df.index.month
df['Year'] = df.index.year
df['DayOfWeek'] = df.index.dayofweek
df['Days'] = (df.index - df.index.min()).days

# ---------------------
# Split Data for Models
# ---------------------
features = ['Days', 'Day', 'Month', 'Year', 'DayOfWeek']
X = df[features]
y = df['Price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# ---------------------
# Train and Evaluate Models
# ---------------------
def evaluate_model(name, model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    print(f"{name} (Train) - MAE: {mean_absolute_error(y_train, y_train_pred):.4f}, MSE: {mean_squared_error(y_train, y_train_pred):.4f}, RMSE: {rmse(y_train, y_train_pred):.4f}, R2: {r2_score(y_train, y_train_pred):.4f}, MAPE: {mean_absolute_percentage_error(y_train, y_train_pred):.4f}%")
    print(f"{name} (Test)  - MAE: {mean_absolute_error(y_test, y_test_pred):.4f}, MSE: {mean_squared_error(y_test, y_test_pred):.4f}, RMSE: {rmse(y_test, y_test_pred):.4f}, R2: {r2_score(y_test, y_test_pred):.4f}, MAPE: {mean_absolute_percentage_error(y_test, y_test_pred):.4f}%")
    print("-" * 100)

# SARIMA
sarima_model = SARIMAX(df['Price'], order=(1, 1, 1), seasonal_order=(1, 1, 1, 12), enforce_stationarity=False, enforce_invertibility=False)
sarima_fit = sarima_model.fit()
df['SARIMA_Prediction'] = sarima_fit.predict(start=len(df) - len(X_test), end=len(df) - 1, dynamic=False)

# SARIMA Predictions
sarima_train_pred = sarima_fit.fittedvalues
sarima_test_pred = sarima_fit.predict(start=len(df) - len(X_test), end=len(df) - 1, dynamic=False)

# Print SARIMA Evaluation Metrics
print(f"SARIMA (Train) - MAE: {mean_absolute_error(y_train, sarima_train_pred[:len(y_train)]):.4f}, "
      f"MSE: {mean_squared_error(y_train, sarima_train_pred[:len(y_train)]):.4f}, "
      f"RMSE: {rmse(y_train, sarima_train_pred[:len(y_train)]):.4f}, "
      f"R2: {r2_score(y_train, sarima_train_pred[:len(y_train)]):.4f}, "
      f"MAPE: {mean_absolute_percentage_error(y_train, sarima_train_pred[:len(y_train)]):.4f}%")

print(f"SARIMA (Test)  - MAE: {mean_absolute_error(y_test, sarima_test_pred):.4f}, "
      f"MSE: {mean_squared_error(y_test, sarima_test_pred):.4f}, "
      f"RMSE: {rmse(y_test, sarima_test_pred):.4f}, "
      f"R2: {r2_score(y_test, sarima_test_pred):.4f}, "
      f"MAPE: {mean_absolute_percentage_error(y_test, sarima_test_pred):.4f}%")

# ---------------------
# 2-Month Predictions
# ---------------------
future_dates = pd.date_range(start=df.index[-1], periods=61, freq='D')[1:]
future_X = pd.DataFrame({'Days': (future_dates - df.index.min()).days,
                          'Day': future_dates.day,
                          'Month': future_dates.month,
                          'Year': future_dates.year,
                          'DayOfWeek': future_dates.dayofweek})

future_sarima = sarima_fit.forecast(steps=60)

# Print predicted values
predictions_df = pd.DataFrame({
    'Date': future_dates,
    'SARIMA': future_sarima,
})
predictions_df["Date"] = predictions_df["Date"].dt.strftime("%Y-%m-%d")
print(predictions_df)
# ---------------------
# Plot Predictions
# ---------------------
plt.figure(figsize=(12, 6))
plt.plot(df.index, df['Price'], label='Actual Prices', color='blue')
plt.plot(future_dates, future_sarima, label='SARIMA Forecast', linestyle='dashed', color='purple')
plt.xlabel('Date')
plt.ylabel('Price')
plt.title('2-Month Forecast')
plt.legend()
plt.show()


# Save models as pickle files
with open("sarima.pkl", "wb") as f:
    pickle.dump(sarima_fit, f)
print("Models saved successfully!")