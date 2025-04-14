import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet

def get_city_data(df, city):
    city_df = df[df['City'] == city][['Date', 'AQI']].dropna()
    city_df = city_df.groupby('Date').mean().reset_index()
    return city_df

def forecast_with_prophet(city_df, periods=180):
    df_prophet = city_df.rename(columns={"Date": "ds", "AQI": "y"})
    model = Prophet()
    model.fit(df_prophet)

    future = model.make_future_dataframe(periods=periods)
    forecast = model.predict(future)
    return forecast, model

def plot_city_trend(city_df):
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(city_df['Date'], city_df['AQI'], color='orange')
    ax.set_title("Historical AQI", fontsize=16)
    ax.set_xlabel("Date")
    ax.set_ylabel("AQI")
    ax.grid(True)
    plt.tight_layout()
    st.pyplot(fig)

