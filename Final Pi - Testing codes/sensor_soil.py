import time
import board
import busio
from adafruit_ads1x15.ads1115 import ADS1115
from adafruit_ads1x15.analog_in import AnalogIn

DRY_VALUE = 16386
WET_VALUE = 8400
NUM_READINGS = 5
READING_DELAY = 0.02

# Initialize I2C and ADS1115 at module level
try:
    i2c = busio.I2C(board.SCL, board.SDA)
    ads = ADS1115(i2c)
    chan = AnalogIn(ads, 0)
except Exception as e:
    print(f"ERROR [sensor_soil]: Failed to initialize ADS1115 - {e}")
    chan = None

def get_soil_moisture_pct():
    # Returns soil moisture as percentage (0-100)
    # Returns None if sensor fails
    if chan is None:
        print("ERROR [sensor_soil]: ADS1115 not initialized")
        return None

    readings = []
    for _ in range(NUM_READINGS):
        try:
            readings.append(chan.value)
        except OSError as e:
            print(f"ERROR [sensor_soil]: I2C read failed - {e}")
        time.sleep(READING_DELAY)

    if len(readings) == 0:
        print("ERROR [sensor_soil]: No valid readings from soil sensor")
        return None

    raw_avg = sum(readings) / len(readings)
    percent = ((DRY_VALUE - raw_avg) / (DRY_VALUE - WET_VALUE)) * 100
    percent = max(0, min(100, percent))
    return round(percent, 1)
