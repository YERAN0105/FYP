import RPi.GPIO as GPIO
import time
import statistics

TRIG_PIN = 27
ECHO_PIN = 22
TANK_HEIGHT_CM = 25.0
NUM_READINGS = 5
READING_DELAY = 0.06

def setup_tank_sensor():
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

def get_tank_level_pct():
    # Returns tank water level as percentage (0-100)
    # Returns None if sensor fails
    readings = []
    for _ in range(NUM_READINGS):
        d = _measure_single()
        if d is not None:
            readings.append(d)
        time.sleep(READING_DELAY)

    if len(readings) == 0:
        print("ERROR [sensor_tank]: No valid readings from HC-SR04")
        return None

    distance = round(statistics.median(readings), 2)
    distance = max(0, min(distance, TANK_HEIGHT_CM))
    water_level_cm = TANK_HEIGHT_CM - distance
    percentage = (water_level_cm / TANK_HEIGHT_CM) * 100
    return round(percentage, 1)

def get_tank_distance_cm():
    # Returns raw distance from sensor to water surface in cm
    # Returns None if sensor fails
    readings = []
    for _ in range(NUM_READINGS):
        d = _measure_single()
        if d is not None:
            readings.append(d)
        time.sleep(READING_DELAY)

    if len(readings) == 0:
        print("ERROR [sensor_tank]: No valid readings from HC-SR04")
        return None

    return round(statistics.median(readings), 2)
