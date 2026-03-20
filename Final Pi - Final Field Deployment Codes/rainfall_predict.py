import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
import joblib
import warnings
warnings.filterwarnings('ignore')

COLOMBO_LAT = 6.9271
COLOMBO_LON = 79.8612

def get_rainfall_predicted_mm():
    # Fetches last 24 days of hourly weather, engineers features,
    # and returns 24h rainfall forecast in mm using XGBoost pipeline.
    # Returns None if models fail to load or API call fails.
    try:
        stage1_model = joblib.load('rainfall/stage1_rain_classifier.pkl')
        stage2_model = joblib.load('rainfall/stage2_amount_regressor.pkl')
        FEATURE_NAMES = stage1_model.get_booster().feature_names
    except Exception as e:
        print(f"ERROR [rainfall_predict]: Failed to load models - {e}")
        return None

    try:
        current_time = datetime.now()
        current_date = current_time.date()
        hourly_vars = ["temperature_2m", "relative_humidity_2m", "dew_point_2m",
                       "precipitation", "surface_pressure", "cloud_cover",
                       "wind_speed_10m", "wind_direction_10m"]

        def fetch_df(url, params):
            r = requests.get(url, params=params, timeout=30)
            r.raise_for_status()
            data = r.json()['hourly']
            return pd.DataFrame({'datetime': pd.to_datetime(data['time']),
                                 **{v: data[v] for v in hourly_vars}})

        df_hist = fetch_df("https://archive-api.open-meteo.com/v1/archive", {
            "latitude": COLOMBO_LAT, "longitude": COLOMBO_LON,
            "start_date": str(current_date - timedelta(days=20)),
            "end_date":   str(current_date - timedelta(days=4)),
            "hourly": ",".join(hourly_vars), "timezone": "Asia/Colombo"
        })

        df_recent = fetch_df("https://api.open-meteo.com/v1/forecast", {
            "latitude": COLOMBO_LAT, "longitude": COLOMBO_LON,
            "hourly": ",".join(hourly_vars),
            "past_days": 4, "forecast_days": 1, "timezone": "Asia/Colombo"
        })
        df_recent = df_recent[df_recent['datetime'] <= current_time]

        df = (pd.concat([df_hist, df_recent], ignore_index=True)
                .drop_duplicates(subset=['datetime'], keep='last')
                .sort_values('datetime').reset_index(drop=True))

    except requests.exceptions.ConnectionError:
        print("ERROR [rainfall_predict]: No internet connection")
        return None
    except requests.exceptions.Timeout:
        print("ERROR [rainfall_predict]: API request timed out")
        return None
    except Exception as e:
        print(f"ERROR [rainfall_predict]: API fetch failed - {e}")
        return None

    try:
        W, MP = 24, 18
        df['hour']        = df['datetime'].dt.hour
        df['month']       = df['datetime'].dt.month
        df['year']        = df['datetime'].dt.year
        df['day_of_year'] = df['datetime'].dt.dayofyear
        df['day_of_week'] = df['datetime'].dt.dayofweek
        df['date_dt']     = df['datetime'].dt.normalize()

        df['precip_24h_sum']           = df['precipitation'].rolling(W, min_periods=MP).sum()
        df['precip_24h_max']           = df['precipitation'].rolling(W, min_periods=MP).max()
        df['precip_24h_mean']          = df['precipitation'].rolling(W, min_periods=MP).mean()
        df['precip_24h_std']           = df['precipitation'].rolling(W, min_periods=MP).std()
        df['precip_24h_rainy_hours']   = df['precipitation'].rolling(W, min_periods=MP).apply(lambda x: (x > 0.1).sum(), raw=True)
        df['temp_24h_mean']            = df['temperature_2m'].rolling(W, min_periods=MP).mean()
        df['temp_24h_max']             = df['temperature_2m'].rolling(W, min_periods=MP).max()
        df['temp_24h_min']             = df['temperature_2m'].rolling(W, min_periods=MP).min()
        df['temp_24h_range']           = df['temp_24h_max'] - df['temp_24h_min']
        df['temp_24h_std']             = df['temperature_2m'].rolling(W, min_periods=MP).std()
        df['temp_24h_trend']           = df['temperature_2m'] - df['temperature_2m'].shift(W)
        df['humidity_24h_mean']        = df['relative_humidity_2m'].rolling(W, min_periods=MP).mean()
        df['humidity_24h_max']         = df['relative_humidity_2m'].rolling(W, min_periods=MP).max()
        df['humidity_24h_min']         = df['relative_humidity_2m'].rolling(W, min_periods=MP).min()
        df['humidity_24h_range']       = df['humidity_24h_max'] - df['humidity_24h_min']
        df['humidity_24h_hours_above_80'] = df['relative_humidity_2m'].rolling(W, min_periods=MP).apply(lambda x: (x > 80).sum(), raw=True)
        df['pressure_24h_mean']        = df['surface_pressure'].rolling(W, min_periods=MP).mean()
        df['pressure_24h_min']         = df['surface_pressure'].rolling(W, min_periods=MP).min()
        df['pressure_24h_max']         = df['surface_pressure'].rolling(W, min_periods=MP).max()
        df['pressure_24h_std']         = df['surface_pressure'].rolling(W, min_periods=MP).std()
        df['pressure_24h_trend']       = df['surface_pressure'] - df['surface_pressure'].shift(W)
        df['wind_24h_mean']            = df['wind_speed_10m'].rolling(W, min_periods=MP).mean()
        df['wind_24h_max']             = df['wind_speed_10m'].rolling(W, min_periods=MP).max()
        df['wind_24h_std']             = df['wind_speed_10m'].rolling(W, min_periods=MP).std()
        df['cloud_24h_mean']           = df['cloud_cover'].rolling(W, min_periods=MP).mean()
        df['cloud_24h_max']            = df['cloud_cover'].rolling(W, min_periods=MP).max()
        df['cloud_24h_hours_above_90'] = df['cloud_cover'].rolling(W, min_periods=MP).apply(lambda x: (x > 90).sum(), raw=True)
        df['dewpoint_24h_mean']        = df['dew_point_2m'].rolling(W, min_periods=MP).mean()
        df['temp_current']             = df['temperature_2m']
        df['humidity_current']         = df['relative_humidity_2m']
        df['pressure_current']         = df['surface_pressure']
        df['dewpoint_current']         = df['dew_point_2m']
        df['wind_speed_current']       = df['wind_speed_10m']
        df['wind_direction_current']   = df['wind_direction_10m']
        df['cloud_cover_current']      = df['cloud_cover']
        df['dew_point_depression_current'] = df['temperature_2m'] - df['dew_point_2m']
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)

        daily_agg = df.groupby('date_dt').agg(
            daily_rainfall_total=('precipitation', 'sum'),
            daily_temp_mean=('temperature_2m', 'mean'),
            daily_pressure_mean=('surface_pressure', 'mean'),
            daily_humidity_mean=('relative_humidity_2m', 'mean'),
        ).reset_index().sort_values('date_dt').reset_index(drop=True)

        for lag in [1, 2, 3, 7, 14]:
            daily_agg[f'rainfall_lag_{lag}d'] = daily_agg['daily_rainfall_total'].shift(lag)
        for lag in [1, 3, 7]:
            daily_agg[f'temp_lag_{lag}d'] = daily_agg['daily_temp_mean'].shift(lag)
        for lag in [1, 3]:
            daily_agg[f'pressure_lag_{lag}d'] = daily_agg['daily_pressure_mean'].shift(lag)
        for lag in [1, 3]:
            daily_agg[f'humidity_lag_{lag}d'] = daily_agg['daily_humidity_mean'].shift(lag)

        lag_cols = [c for c in daily_agg.columns if '_lag_' in c]
        df = df.merge(daily_agg[['date_dt'] + lag_cols], on='date_dt', how='left')

        df['month_sin']       = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos']       = np.cos(2 * np.pi * df['month'] / 12)
        df['day_of_year_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365.25)
        df['day_of_year_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365.25)
        df['season_SW_Monsoon']  = df['month'].isin([5, 6, 7]).astype(int)
        df['season_NE_Monsoon']  = df['month'].isin([12, 1, 2]).astype(int)
        df['season_Inter_Heavy'] = df['month'].isin([4, 10, 11]).astype(int)
        df['season_Inter_Light'] = (~df['month'].isin([5,6,7,12,1,2,4,10,11])).astype(int)
        df['days_into_sw_monsoon'] = df.apply(
            lambda r: (r['datetime'] - pd.Timestamp(year=r['year'], month=5, day=1)).days
                      if r['month'] in [5, 6, 7] else 0, axis=1)
        df['days_into_ne_monsoon'] = df.apply(
            lambda r: (r['datetime'] - pd.Timestamp(
                           year=r['year'] if r['month'] == 12 else r['year'] - 1,
                           month=12, day=1)).days
                      if r['month'] in [12, 1, 2] else 0, axis=1)
        df['pressure_tendency_3h']     = (df['surface_pressure'] - df['surface_pressure'].shift(3)) / 3
        df['pressure_tendency_6h']     = (df['surface_pressure'] - df['surface_pressure'].shift(6)) / 6
        df['wind_from_southwest']      = ((df['wind_direction_10m'] >= 180) & (df['wind_direction_10m'] <= 270)).astype(int)
        df['wind_from_northeast']      = ((df['wind_direction_10m'] >= 0)   & (df['wind_direction_10m'] <= 90)).astype(int)
        raw_change = np.abs(df['wind_direction_10m'] - df['wind_direction_10m'].shift(6))
        df['wind_direction_change_6h'] = raw_change.apply(lambda x: min(x, 360 - x) if pd.notna(x) else x)
        df['instability_index']        = (df['temp_24h_range'] * (1 - df['dew_point_depression_current'] / 20)).clip(lower=0)
        df['moisture_flux']            = df['humidity_24h_mean'] * df['wind_24h_mean']

        latest = df.iloc[-1:]
        X_pred = pd.DataFrame(index=[0], columns=FEATURE_NAMES, dtype=float)
        for feat in FEATURE_NAMES:
            X_pred[feat] = latest[feat].values[0] if feat in latest.columns else 0.0
        X_pred = X_pred.fillna(0).astype(float)

        rain_prob   = stage1_model.predict_proba(X_pred)[0, 1]
        amount      = max(0, stage2_model.predict(X_pred)[0])
        prediction  = max(0, rain_prob * amount)
        return round(prediction, 2)

    except Exception as e:
        print(f"ERROR [rainfall_predict]: Feature engineering or prediction failed - {e}")
        return None
