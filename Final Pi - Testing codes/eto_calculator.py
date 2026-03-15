import math
import requests
from datetime import date

LATITUDE = 6.9271
LONGITUDE = 79.8612
ELEVATION_M = 7

def _saturation_vapor_pressure(T):
    return 0.6108 * math.exp((17.27 * T) / (T + 237.3))

def _slope_svp_curve(T):
    return 4098.0 * _saturation_vapor_pressure(T) / ((T + 237.3) ** 2)

def _actual_vapor_pressure(RHmin, RHmax, es_min, es_max):
    return ((RHmax / 100.0) * es_min + (RHmin / 100.0) * es_max) / 2.0

def _psychrometric_constant(elevation_m):
    P = 101.3 * ((293.0 - 0.0065 * elevation_m) / 293.0) ** 5.26
    return 0.000665 * P

def _wind_speed_10m_to_2m(u10):
    return u10 * 4.87 / math.log(67.8 * 10.0 - 5.42)

def _extraterrestrial_radiation(day_of_year, latitude_deg):
    phi = math.radians(latitude_deg)
    dr = 1 + 0.033 * math.cos((2 * math.pi / 365) * day_of_year)
    delta = 0.409 * math.sin((2 * math.pi / 365) * day_of_year - 1.39)
    ws = math.acos(-math.tan(phi) * math.tan(delta))
    return (24 * 60 / math.pi) * 0.0820 * dr * (
        ws * math.sin(phi) * math.sin(delta) +
        math.cos(phi) * math.cos(delta) * math.sin(ws)
    )

def _net_radiation_fao(Rs, Tmax, Tmin, ea, Ra, elevation_m):
    Rns = (1.0 - 0.23) * Rs
    Rso = (0.75 + 2e-5 * elevation_m) * Ra
    Rs_Rso = max(0.0, min(Rs / Rso if Rso > 0 else 0.0, 1.5))
    Tavg4 = ((Tmax + 273.16) ** 4 + (Tmin + 273.16) ** 4) / 2.0
    Rnl = 4.903e-9 * Tavg4 * (0.34 - 0.14 * math.sqrt(max(ea, 0.0))) * (1.35 * Rs_Rso - 0.35)
    return Rns - Rnl

def get_eto_mm_day():
    # Fetches today's weather from Open-Meteo and computes FAO-56 ETo
    # Returns ETo in mm/day, or None if API call fails
    try:
        today = date.today().isoformat()
        doy = date.today().timetuple().tm_yday

        params = {
            "latitude": LATITUDE, "longitude": LONGITUDE,
            "start_date": today, "end_date": today,
            "daily": ["temperature_2m_max", "temperature_2m_min",
                      "relative_humidity_2m_max", "relative_humidity_2m_min",
                      "wind_speed_10m_mean", "shortwave_radiation_sum"],
            "timezone": "Asia/Colombo"
        }

        response = requests.get("https://api.open-meteo.com/v1/forecast",
                                params=params, timeout=30)
        response.raise_for_status()
        data = response.json()["daily"]

        Tmax  = data["temperature_2m_max"][0]
        Tmin  = data["temperature_2m_min"][0]
        RHmax = data["relative_humidity_2m_max"][0]
        RHmin = data["relative_humidity_2m_min"][0]
        u10   = data["wind_speed_10m_mean"][0]
        Rs    = data["shortwave_radiation_sum"][0]

        Tmean = (Tmax + Tmin) / 2.0
        es    = (_saturation_vapor_pressure(Tmax) + _saturation_vapor_pressure(Tmin)) / 2.0
        ea    = _actual_vapor_pressure(RHmin, RHmax,
                                       _saturation_vapor_pressure(Tmin),
                                       _saturation_vapor_pressure(Tmax))
        delta = _slope_svp_curve(Tmean)
        gamma = _psychrometric_constant(ELEVATION_M)
        Ra    = _extraterrestrial_radiation(doy, LATITUDE)
        Rn    = _net_radiation_fao(Rs, Tmax, Tmin, ea, Ra, ELEVATION_M)
        u2    = _wind_speed_10m_to_2m(u10 / 3.6)

        numerator   = 0.408 * delta * Rn + gamma * (900.0 / (Tmean + 273.0)) * u2 * (es - ea)
        denominator = delta + gamma * (1.0 + 0.34 * u2)
        eto = numerator / max(denominator, 1e-6)

        return round(eto, 2)

    except requests.exceptions.ConnectionError:
        print("ERROR [eto_calculator]: No internet connection")
        return None
    except requests.exceptions.Timeout:
        print("ERROR [eto_calculator]: API request timed out")
        return None
    except Exception as e:
        print(f"ERROR [eto_calculator]: Unexpected error - {e}")
        return None
