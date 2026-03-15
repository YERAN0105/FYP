import RPi.GPIO as GPIO
import time

RAIN_SENSOR_PIN = 17  # GPIO 17 (Pin 11)

def setup_rain_sensor():
    GPIO.setup(RAIN_SENSOR_PIN, GPIO.IN)

def is_raining():
    # Returns True if rain detected (LOW signal)
    # Returns False if no rain (HIGH signal)
    # Returns None if read fails
    try:
        state = GPIO.input(RAIN_SENSOR_PIN)
        if state == GPIO.LOW:
            return True   # Rain detected
        else:
            return False  # No rain
    except Exception as e:
        print(f"ERROR [sensor_rain]: Failed to read rain sensor - {e}")
        return None
