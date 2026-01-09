import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
from statsmodels.tsa.statespace.sarimax import SARIMAX

warnings.filterwarnings("ignore")

st.title("Apple Stock Price Prediction (SARIMA)")
st.write("SARIMA-based time series forecasting")
@st.cache_data
def load_data():
    df = pd.read_csv("P625 DATASET.csv")
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')
    df.set_index('Date', inplace=True)
    df = df.asfreq('B')   # Business day frequency (IMPORTANT)
    df['Close'].fillna(method='ffill', inplace=True)
    return df

 df = load_data("P625 DATASET.csv")
# Historical Plot
st.subheader("📊 Historical Close Prices")
st.line_chart(df['Close'])

# SARIMA Parameters

st.subheader("SARIMA Model Parameters")

p = st.number_input("AR (p)", 0, 5, 5)
d = st.number_input("Differencing (d)", 0, 2, 1)
q = st.number_input("MA (q)", 0, 5, 0)

P = st.number_input("Seasonal AR (P)", 0, 2, 1)
D = st.number_input("Seasonal Differencing (D)", 0, 2, 1)
Q = st.number_input("Seasonal MA (Q)", 0, 2, 1)
m = st.number_input("Seasonal Period (m)", 5, 30, 5)

# Train SARIMA

model = SARIMAX(
    df['Close'],
    order=(p, d, q),
    seasonal_order=(P, D, Q, m),
    enforce_stationarity=False,
    enforce_invertibility=False
)

model_fit = model.fit()

# Forecast Section

st.subheader("🔮 Forecast Future Prices")

forecast_days = st.slider("Select forecast days", 1, 60, 30)

forecast = model_fit.forecast(steps=forecast_days)

future_dates = pd.date_range(
    start=df.index[-1] + pd.Timedelta(days=1),
    periods=forecast_days,
    freq='B'
)

forecast_df = pd.DataFrame({
    "Date": future_dates,
    "Predicted Close Price": forecast.values
}).set_index("Date")

# Plot Forecast

fig, ax = plt.subplots(figsize=(10,5))
ax.plot(df['Close'], label="Historical")
ax.plot(forecast_df, label="SARIMA Forecast", color="red")
ax.legend()
ax.set_title("Apple Stock Price Forecast (SARIMA)")

st.pyplot(fig)

# Display Table

st.subheader("📅 Forecasted Prices")
st.dataframe(forecast_df)

st.success("SARIMA Forecast Generated Successfully ✅")


