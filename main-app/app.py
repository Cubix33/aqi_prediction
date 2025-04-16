import streamlit as st
import pandas as pd
import numpy as np
from prophet import Prophet
import matplotlib.pyplot as plt
import seaborn as sns
from utils import get_city_data, forecast_with_prophet, plot_city_trend

st.set_page_config(page_title="AQI Forecasting Dashboard", layout="wide")

st.title("🌫️ Air Quality Index (AQI) Forecast Dashboard")
st.markdown("Predict and visualize future AQI trends using machine learning and time series models.")

# Load and cache data
@st.cache_data
def load_data():
    df = pd.read_csv("main-app/data/city_day.csv")
    df['Date'] = pd.to_datetime(df['Date'])
    return df

df = load_data()
cities = df['City'].dropna().unique()
selected_city = st.selectbox("📍 Choose a City", sorted(cities))

city_df = get_city_data(df, selected_city)

# Show latest AQI
latest = city_df.sort_values("Date").iloc[-1]
st.metric("📅 Latest Date", latest['Date'].strftime("%d %b %Y"))
st.metric("🌡️ Latest AQI", f"{latest['AQI']:.2f}")

# Plot historical trend
st.subheader("📉 Historical AQI Trend")
plot_city_trend(city_df)

# Forecast future AQI
st.subheader("🔮 Forecast Future AQI")
n_days = st.slider("Forecast Days", 30, 365, 180)
forecast, model = forecast_with_prophet(city_df, n_days)

# Show forecast plot
st.write("### Forecast Plot")
fig1 = model.plot(forecast)
st.pyplot(fig1)

st.write("### Forecast Components")
fig2 = model.plot_components(forecast)
st.pyplot(fig2)

# AQI Category
def aqi_category(aqi):
    if aqi <= 50:
        return "Good"
    elif aqi <= 100:
        return "Moderate"
    elif aqi <= 200:
        return "Unhealthy"
    elif aqi <= 300:
        return "Very Unhealthy"
    else:
        return "Hazardous"

st.write("### 🚦 AQI Categories (Forecasted)")
forecast_display = forecast[['ds', 'yhat']].copy()
forecast_display['AQI Category'] = forecast_display['yhat'].apply(aqi_category)
st.dataframe(forecast_display.tail(10))

