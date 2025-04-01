import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load SARIMA Model
with open("sarima.pkl", "rb") as f:
    sarima_model = pickle.load(f)

# Load dataset
file_path = r"C:\Users\dubey\Project_244\Data Collection\Dataset_244\Aluminium_Data_cleaned.csv"
df = pd.read_csv(file_path, parse_dates=['Date'], index_col='Date')
df = df.sort_index()
print(df.info())
# Feature Engineering
df['Day'] = df.index.day
df['Month'] = df.index.month
df['Year'] = df.index.year
df['DayOfWeek'] = df.index.dayofweek
df['Days'] = (df.index - df.index.min()).days

# Split Data
features = ['Days', 'Day', 'Month', 'Year', 'DayOfWeek']
X = df[features]
y = df['Price']
train_size = int(len(X) * 0.8)
X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]

# SARIMA Predictions
sarima_train_pred = sarima_model.fittedvalues
sarima_test_pred = sarima_model.predict(start=len(df) - len(X_test), end=len(df) - 1, dynamic=False)

# Evaluation Metrics
def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

sarima_rmse = rmse(y_test, sarima_test_pred)
sarima_mape = mean_absolute_percentage_error(y_test, sarima_test_pred)
sarima_r2 = r2_score(y_test, sarima_test_pred)

# Streamlit UI
st.set_page_config(page_title="Aluminium Price Forecasting", layout="wide")

st.markdown("<h1 style='text-align: center;'>📈 Aluminium Price Forecasting</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center;'>Using SARIMA Model</h3>", unsafe_allow_html=True)

st.write("---")

# User Input: Number of Days for Prediction
forecast_days = st.slider("Select Number of Days to Forecast:", min_value=7, max_value=180, step=7, value=60)

st.write("---")

# Show Evaluation Metrics in Center
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("RMSE", f"{sarima_rmse:.4f}")
with col2:
    st.metric("MAPE", f"{sarima_mape:.4f}%")
with col3:
    st.metric("R² Score", f"{sarima_r2:.4f}")

st.write("---")

# Predict Button
if st.button("Predict"):
    # Future Predictions
    future_dates = pd.date_range(start=df.index[-1], periods=forecast_days + 1, freq='D')[1:]
    future_sarima = sarima_model.forecast(steps=forecast_days)

    # Show Predicted Values in Center
    st.markdown(f"<h3 style='text-align: center;'>📊 {forecast_days}-Day Predicted Prices</h3>", unsafe_allow_html=True)

    # Display centered table
    styled_df = pd.DataFrame({'Date': future_dates.strftime("%Y-%m-%d"), 'SARIMA Forecast': future_sarima})
    styled_df_html = styled_df.style.set_properties(**{'text-align': 'center'}).hide(axis="index").to_html()

    st.markdown(
        f"<div style='display: flex; justify-content: center;'>{styled_df_html}</div>",
        unsafe_allow_html=True
    )

    st.write("---")

    # Plot Graph
    st.markdown(f"<h3 style='text-align: center;'>📉 {forecast_days}-Day Forecast Visualization</h3>", unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df.index, df['Price'], label='Actual Prices', color='blue')
    ax.plot(future_dates, future_sarima, label='SARIMA Forecast', linestyle='dashed', color='purple')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price')
    ax.set_title(f'{forecast_days}-Day Forecast')
    ax.legend()
    st.pyplot(fig)

    st.write("---")

    st.success(f"✅ Forecast for {forecast_days} days generated successfully!")
