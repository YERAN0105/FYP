import RPi.GPIO as GPIO
from RpiMotorLib import RpiMotorLib
import time

DIR        = 20
STEP       = 21
STEPS      = 1350
STEP_DELAY = 0.003

def setup_motor_pins():
    # Call once at startup to initialise motor pins to LOW
    GPIO.setup(DIR,  GPIO.OUT, initial=GPIO.LOW)
    GPIO.setup(STEP, GPIO.OUT, initial=GPIO.LOW)

def _hold_pins_low():
    # Always drive pins LOW after any movement to prevent motor drift
    GPIO.output(DIR,  GPIO.LOW)
    GPIO.output(STEP, GPIO.LOW)

def open_gate():
    # Drives stepper motor to open the water gate
    # Returns True on success, False on failure
    try:
        motor = RpiMotorLib.A4988Nema(DIR, STEP, (0, 0, 0), "A4988")
        print("[motor_gate] Opening gate...")
        motor.motor_go(False, "Full", STEPS, STEP_DELAY, False, 0.0)
        _hold_pins_low()
        print("[motor_gate] Gate fully opened.")
        return True
    except Exception as e:
        _hold_pins_low()  # Hold low even on failure
        print(f"ERROR [motor_gate]: Failed to open gate - {e}")
        return False

def close_gate():
    # Drives stepper motor to close the water gate
    # Returns True on success, False on failure
    try:
        motor = RpiMotorLib.A4988Nema(DIR, STEP, (0, 0, 0), "A4988")
        print("[motor_gate] Closing gate...")
        motor.motor_go(True, "Full", STEPS, STEP_DELAY, False, 0.0)
        _hold_pins_low()
        print("[motor_gate] Gate fully closed.")
        return True
    except Exception as e:
        _hold_pins_low()  # Hold low even on failure
        print(f"ERROR [motor_gate]: Failed to close gate - {e}")
        return False
