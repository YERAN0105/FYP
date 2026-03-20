import RPi.GPIO as GPIO
import time
from sensor_tank import setup_tank_sensor, get_tank_distance_cm
from motor_gate  import open_gate, close_gate, _hold_pins_low

L_POT  = 23.0
W_POT  = 15.0
L_TANK = 20.0
W_TANK = 20.0

POLL_INTERVAL_SEC = 0.5
TIMEOUT_SEC       = 60.0

def _calculate_targets(cwr_mm):
    v_needed_cm3 = L_POT * W_POT * (cwr_mm / 10.0)
    delta_h_cm   = v_needed_cm3 / (L_TANK * W_TANK)
    return round(v_needed_cm3, 4), round(delta_h_cm, 4)

def run_irrigation_cycle(cwr_mm):
    # Main irrigation cycle - takes real CWR value from predict_cwr.py
    # Returns True if cycle completed successfully, False otherwise
    print("=" * 45)
    print("         IRRIGATION CYCLE STARTED")
    print("=" * 45)

    print("[gate_control] Initialising tank sensor...")
    setup_tank_sensor()

    v_needed, delta_h = _calculate_targets(cwr_mm)
    print(f"[gate_control] CWR            : {cwr_mm} mm")
    print(f"[gate_control] Volume needed   : {v_needed} cm3")
    print(f"[gate_control] Tank drop req   : {delta_h} cm")

    # Step 1 - Read initial tank distance
    d_initial = get_tank_distance_cm()
    if d_initial is None:
        print("ERROR [gate_control]: Cannot read initial tank distance. Aborting.")
        return False

    d_target = round(d_initial + delta_h, 2)
    print(f"[gate_control] d_initial       : {d_initial} cm")
    print(f"[gate_control] d_target        : {d_target} cm")

    # Step 2 - Open gate
    if not open_gate():
        print("ERROR [gate_control]: Gate failed to open. Aborting.")
        return False

    # Step 3 - Poll until target distance reached or timeout
    print("[gate_control] Monitoring tank level...")
    start_time     = time.time()
    target_reached = False

    try:
        while True:
            elapsed = time.time() - start_time

            if elapsed >= TIMEOUT_SEC:
                print(f"WARNING [gate_control]: Timeout ({TIMEOUT_SEC}s) - force closing gate.")
                break

            current_distance = get_tank_distance_cm()

            if current_distance is None:
                print("WARNING [gate_control]: Sensor read failed - retrying...")
                time.sleep(POLL_INTERVAL_SEC)
                continue

            print(f"  [{elapsed:>5.1f}s]  Distance: {current_distance} cm  Target: {d_target} cm")

            # Step 4 - Close when enough water delivered
            if current_distance >= d_target:
                print(f"[gate_control] Target reached: {current_distance} cm >= {d_target} cm")
                target_reached = True
                break

            time.sleep(POLL_INTERVAL_SEC)

    except KeyboardInterrupt:
        # Safety: hold pins low immediately on interrupt, then close gate
        _hold_pins_low()
        print("[gate_control] Interrupted - closing gate for safety...")
        close_gate()
        return False

    # Step 5 - Close gate
    if not close_gate():
        print("CRITICAL [gate_control]: Gate failed to close. Manual intervention required.")
        return False

    print("=" * 45)
    if target_reached:
        print(f"  CYCLE COMPLETE - {v_needed} cm3 delivered.")
    else:
        print("  CYCLE ENDED - Timeout. Check tank and gate.")
    print("=" * 45)

    return target_reached
