import RPi.GPIO as GPIO
import time
import statistics

TRIG_PIN = 23
ECHO_PIN = 24
EMPTY_FIELD_DISTANCE_CM = 8.0
NUM_READINGS = 5
READING_DELAY = 0.06

def setup_field_sensor():
    GPIO.setup(TRIG_PIN, GPIO.OUT)
    GPIO.setup(ECHO_PIN, GPIO.IN)
    GPIO.output(TRIG_PIN, False)
    time.sleep(2)

def _measure_single():
    GPIO.output(TRIG_PIN, True)
    time.sleep(0.00001)
    GPIO.output(TRIG_PIN, False)

    pulse_start = time.time()
    timeout_start = pulse_start
    while GPIO.input(ECHO_PIN) == 0:
        pulse_start = time.time()
        if pulse_start - timeout_start > 0.1:
            return None

    pulse_end = time.time()
    timeout_end = pulse_end
    while GPIO.input(ECHO_PIN) == 1:
        pulse_end = time.time()
        if pulse_end - timeout_end > 0.1:
            return None

    duration = pulse_end - pulse_start
    distance = round(duration * 17150, 2)
    if 2 <= distance <= 400:
        return distance
    return None

def get_water_depth_cm():
    # Returns standing water depth in field in cm
    # Returns None if sensor fails
    readings = []
    for _ in range(NUM_READINGS):
        d = _measure_single()
        if d is not None:
            readings.append(d)
        time.sleep(READING_DELAY)

    if len(readings) == 0:
        print("ERROR [sensor_field]: No valid readings from HC-SR04")
        return None

    distance = round(statistics.median(readings), 2)
    water_depth = max(0, EMPTY_FIELD_DISTANCE_CM - distance)
    return round(water_depth, 2)
