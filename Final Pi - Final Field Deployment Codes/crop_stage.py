from datetime import date

# Planting start date - change this to the actual Day 1 of your crop cycle
PLANTING_START_DATE = date(2026, 3, 15)

# Crop schedule: each stage defines total days and how many final days are active
# "active_last_days" = how many days at the END of the stage irrigation runs
# "active_last_days" = total days means the whole stage is active
CROP_SCHEDULE = [
    {"stage": 1, "name": "Land Preparation", "total_days": 30, "active_last_days": 30},
    {"stage": 2, "name": "Seedling",          "total_days": 17, "active_last_days": 3},
    {"stage": 3, "name": "Vegetative",        "total_days": 8,  "active_last_days": 3},
    {"stage": 4, "name": "Reproductive",      "total_days": 16, "active_last_days": 5},
    {"stage": 5, "name": "Ripening",          "total_days": 10, "active_last_days": 10},
]

def get_crop_stage():
    # Returns (stage_number, is_active_day) tuple
    # stage_number: 1-5 based on current day in crop cycle
    # is_active_day: True if irrigation should run today, False if system should idle
    # Returns (None, False) if crop cycle is complete or not yet started

    today = date.today()
    days_elapsed = (today - PLANTING_START_DATE).days + 1  # Day 1 = planting day

    if days_elapsed < 1:
        print("[crop_stage] Crop cycle has not started yet.")
        return None, False

    cumulative = 0
    for entry in CROP_SCHEDULE:
        cumulative += entry["total_days"]

        if days_elapsed <= cumulative:
            stage_number = entry["stage"]
            stage_name   = entry["name"]

            # Day within this stage (1-indexed)
            stage_start  = cumulative - entry["total_days"] + 1
            day_in_stage = days_elapsed - stage_start + 1

            # Active window is the last N days of the stage
            first_active_day = entry["total_days"] - entry["active_last_days"] + 1
            is_active = day_in_stage >= first_active_day

            print(f"[crop_stage] Day {days_elapsed} of crop cycle")
            print(f"[crop_stage] Stage {stage_number} ({stage_name}) - Day {day_in_stage} of {entry['total_days']}")
            print(f"[crop_stage] Active irrigation window starts day {first_active_day} of this stage")
            print(f"[crop_stage] Today is {'an ACTIVE' if is_active else 'NOT an active'} irrigation day")

            return stage_number, is_active

    print("[crop_stage] Crop cycle is complete (all 81 days passed).")
    return None, False
