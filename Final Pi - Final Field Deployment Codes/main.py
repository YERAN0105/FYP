import RPi.GPIO as GPIO
import time

from sensor_rain      import setup_rain_sensor, is_raining
from sensor_field     import setup_field_sensor, get_water_depth_cm
from sensor_soil      import get_soil_moisture_pct
from sensor_tank      import setup_tank_sensor, get_tank_level_pct
from eto_calculator   import get_eto_mm_day
from rainfall_predict import get_rainfall_predicted_mm
from crop_stage       import get_crop_stage
from predict_cwr      import predict_cwr
from gate_control     import run_irrigation_cycle
from motor_gate       import setup_motor_pins, _hold_pins_low

# GPIO setup done once for all sensors
GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)

# Initialise motor pins LOW immediately at startup
setup_motor_pins()
_hold_pins_low()

try:
    print("Setting up sensors...")
    setup_rain_sensor()
    setup_field_sensor()
    setup_tank_sensor()
    print("Sensors ready.")
    print("-" * 45)

    # -------------------------------------------------------
    # STEP 1 - Crop stage resolution
    # -------------------------------------------------------
    print("Step 1: Resolving crop stage...")
    stage, is_active = get_crop_stage()

    if stage is None:
        print("  Crop cycle not started or already complete. System idle.")
        _hold_pins_low()
        exit()

    if not is_active:
        print("  Today is not an active irrigation day. System idle.")
        _hold_pins_low()
        exit()

    print(f"  Crop Stage      : {stage} (active irrigation day)")
    print("-" * 45)

    # -------------------------------------------------------
    # STEP 2 - Rain check with 1-hour retry loop
    # -------------------------------------------------------
    print("Step 2: Checking rain sensor...")
    while True:
        raining = is_raining()

        if raining is None:
            print("SYSTEM STOPPED: Rain sensor failed at Step 2")
            _hold_pins_low()
            exit()

        if not raining:
            print("  No rain detected - proceeding.")
            break

        print("  Rain detected - waiting 1 hour before retrying...")
        time.sleep(3600)  # Wait 1 hour then poll again
        print("  Retrying rain check...")

    print("-" * 45)

    # -------------------------------------------------------
    # STEP 3 - Read all sensors
    # -------------------------------------------------------
    print("Step 3: Reading field water depth...")
    water_depth = get_water_depth_cm()
    if water_depth is None:
        print("SYSTEM STOPPED: Field sensor failed at Step 3")
        _hold_pins_low()
        exit()
    print(f"  Water Depth     : {water_depth} cm")

    print("Step 4: Reading soil moisture...")
    soil_moisture = get_soil_moisture_pct()
    if soil_moisture is None:
        print("SYSTEM STOPPED: Soil sensor failed at Step 4")
        _hold_pins_low()
        exit()
    print(f"  Soil Moisture   : {soil_moisture}%")

    print("Step 5: Reading tank level...")
    tank_level = get_tank_level_pct()
    if tank_level is None:
        print("SYSTEM STOPPED: Tank sensor failed at Step 5")
        _hold_pins_low()
        exit()
    print(f"  Tank Level      : {tank_level}%")

    # -------------------------------------------------------
    # STEP 4 - Predictive inference pipeline
    # -------------------------------------------------------
    print("Step 6: Calculating ETo from Open-Meteo...")
    et_mm_day = get_eto_mm_day()
    if et_mm_day is None:
        print("SYSTEM STOPPED: ETo calculation failed at Step 6")
        _hold_pins_low()
        exit()
    print(f"  ETo             : {et_mm_day} mm/day")

    print("Step 7: Predicting 24h rainfall...")
    rainfall_mm = get_rainfall_predicted_mm()
    if rainfall_mm is None:
        print("SYSTEM STOPPED: Rainfall prediction failed at Step 7")
        _hold_pins_low()
        exit()
    print(f"  Rainfall (24h)  : {rainfall_mm} mm")

    print("-" * 45)
    print("Step 8: Predicting CWR...")
    cwr = predict_cwr(
        water_depth_cm        = water_depth,
        soil_moisture_pct     = soil_moisture,
        tank_level_pct        = tank_level,
        et_mm_day             = et_mm_day,
        rainfall_predicted_mm = rainfall_mm,
        crop_stage            = stage
    )

    if cwr is None:
        print("SYSTEM STOPPED: CWR prediction failed at Step 8")
        _hold_pins_low()
        exit()

    print(f"  Predicted CWR   : {cwr} mm")

    # -------------------------------------------------------
    # STEP 5 - Actuation decision
    # -------------------------------------------------------
    if cwr == 0.0:
        print("-" * 45)
        print("CWR is 0 mm - no irrigation needed. System idle.")
        _hold_pins_low()
        exit()

    print("-" * 45)
    print(f"Step 9: Running irrigation cycle with CWR = {cwr} mm...")
    run_irrigation_cycle(cwr_mm=cwr)

except KeyboardInterrupt:
    print("[main] Interrupted - holding motor pins low.")
    _hold_pins_low()
