import streamlit as st
import pandas as pd
import numpy as np
import datetime
from xgboost import XGBRegressor
import pydeck as pdk
import matplotlib.pyplot as plt
import time

############################################################
# AUTO REFRESH EVERY 60 SECONDS
############################################################

refresh_interval = 60  # seconds
last_run = st.session_state.get("last_run", 0)

current_time = time.time()
if current_time - last_run > refresh_interval:
    st.session_state["last_run"] = current_time
    st.experimental_rerun()


############################################################
# SENSOR CONFIGURATION
############################################################

SENSORS = {
    1: {
        "street": "sensor1",
        "lat": 41.9025,
        "lon": 12.4950,
        "model_json": "air_quality_model_sensor1.json",
        "df": "air-quality-data_sensor1.csv"
    },
    2: {
        "street": "sensor2",
        "lat": 41.8967,
        "lon": 12.4822,
        "model_json": "air_quality_model_sensor2.json",
        "df": "air-quality-data_sensor2.csv"
    },
    3: {
        "street": "sensor3",
        "lat": 41.9100,
        "lon": 12.4600,
        "model_json": "air_quality_model_sensor3.json",
        "df": "air-quality-data_sensor3.csv"
    },
    4: {
        "street": "sensor4",
        "lat": 41.8850,
        "lon": 12.5000,
        "model_json": "air_quality_model_sensor4.json",
        "df": "air-quality-data_sensor4.csv"
    }
}

############################################################
# SIDEBAR
############################################################
st.sidebar.title("Sensor & Page Selection")

sensor_option = st.sidebar.selectbox(
    "Select Sensor",
    ["All Sensors", "Sensor 1", "Sensor 2", "Sensor 3", "Sensor 4"]
)

page = st.sidebar.radio(
    "Navigation",
    ["Forecasts", "Historical Charts", "Map of Rome"]
)

############################################################
# UTILITY FUNCTIONS
############################################################

@st.cache_data
def load_xgboost_model(path):
    model = XGBRegressor()
    model.load_model(path)
    return model

@st.cache_data
def load_sensor_data(df_path):
    df = pd.read_csv(df_path)
    df["date"] = pd.to_datetime(df["date"])
    return df

def run_prediction(sensor_id: int):
    sensor = SENSORS[sensor_id]
    df = load_sensor_data(sensor["df"])
    model = load_xgboost_model(sensor["model_json"])

    # Use last row for lags
    last = df.tail(1).copy()
    lag1 = last["lag_1"].values[0]
    lag2 = last["lag_2"].values[0]
    lag3 = last["lag_3"].values[0]

    # Predict next day
    tomorrow = last["date"].values[0] + np.timedelta64(1, "D")
    next_row = last.copy()
    next_row["date"] = tomorrow
    next_row["lag_1"] = lag1
    next_row["lag_2"] = lag2
    next_row["lag_3"] = lag3

    feature_cols = [
        "lag_1", "lag_2", "lag_3",
        "temperature_2m_mean",
        "precipitation_sum",
        "wind_speed_10m_max",
        "wind_direction_10m_dominant"
    ]

    X = next_row[feature_cols]
    pred = float(model.predict(X)[0])
    next_row["predicted_pm25"] = pred

    return pred, df, next_row

def pm25_to_color(pm):
    if pm <= 50:
        return [0, 255, 0]  # green
    elif pm <= 100:
        return [255, 165, 0]  # orange
    elif pm <= 130:
        return [255, 0, 0]  # red
    else:
        return [148, 0, 211]  # violet

############################################################
# HELPER TO DETERMINE WHICH SENSORS TO SHOW
############################################################
def selected_sensors():
    if sensor_option == "All Sensors":
        return range(1, 5)
    else:
        return [int(sensor_option[-1])]

############################################################
# PAGE 1: FORECASTS
############################################################
if page == "Forecasts":
    st.title("📈 PM2.5 Forecasts for Rome Sensors")
    for sensor_id in selected_sensors():
        st.subheader(f"Sensor {sensor_id} — {SENSORS[sensor_id]['street']}")
        pred, df_hist, pred_row = run_prediction(sensor_id)
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.bar(["Prediction"], [pred], color=np.array(pm25_to_color(pred))/255)
        ax.set_ylabel("PM2.5")
        ax.set_title(f"Predicted PM2.5: {pred:.2f}")
        st.pyplot(fig)

############################################################
# PAGE 2: HISTORICAL CHARTS
############################################################
if page == "Historical Charts":
    st.title("📉 Historical PM2.5 Levels")
    for sensor_id in selected_sensors():
        st.subheader(f"Sensor {sensor_id} — {SENSORS[sensor_id]['street']}")
        _, df_hist, _ = run_prediction(sensor_id)
        fig, ax = plt.subplots(figsize=(7, 3))
        ax.plot(df_hist["date"], df_hist["pm25"], label="Observed PM2.5", color="blue")
        ax.set_ylabel("PM2.5")
        ax.set_xlabel("Date")
        ax.set_title("Historical PM2.5 Levels")
        ax.legend()
        st.pyplot(fig)

############################################################
# PAGE 3: MAP OF ROME
############################################################
if page == "Map of Rome":
    st.title("🗺️ Air Quality Map of Rome")

    map_rows = []
    for sensor_id in selected_sensors():
        pred, _, _ = run_prediction(sensor_id)
        s = SENSORS[sensor_id]
        map_rows.append({
            "sensor": sensor_id,
            "street": s["street"],
            "lat": s["lat"],
            "lon": s["lon"],
            "pm25": pred,
            "color": pm25_to_color(pred)
        })

    map_df = pd.DataFrame(map_rows)

    layer = pdk.Layer(
        "ScatterplotLayer",
        data=map_df,
        get_position=["lon", "lat"],
        get_radius=150,
        get_fill_color="color",
        pickable=True,
    )

    view_state = pdk.ViewState(latitude=41.9028, longitude=12.4964, zoom=11)

    deck = pdk.Deck(
        layers=[layer],
        initial_view_state=view_state,
        tooltip={"text": "Sensor {sensor}\nStreet: {street}\nPM2.5: {pm25}"}
    )

    st.pydeck_chart(deck)
