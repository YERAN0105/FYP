# ============================================================
# RASPBERRY PI 4 - HYBRID STACK BENCHMARK SCRIPT
# ML-Based Predictive Irrigation System for Paddy Cultivation
# Benchmarks: Hybrid Stacking Ensemble (Teacher)
#
# Required files in same folder:
#   family1_champion_retrained.pkl
#   family2_champion_retrained.pkl
#   family3_champion_retrained.pkl
#   meta_learner_ridge.pkl
#
# Run: python pi_benchmark_stack.py
# ============================================================

import numpy as np
import psutil
import time
import os
import json
import gc
import pickle
import joblib
from datetime import datetime

# ============================================================
# CONFIGURATION
# ============================================================

N_WARMUP  = 10
N_REPEATS = 500

family1_path = './retrained_family1_catboost.pkl'
family2_path = './retrained_family2_svr.pkl'
family3_path = './retrained_family3_mlp.pkl'
meta_path    = './meta_learner_ridge.pkl'

FEATURE_COLUMNS = [
    'Water_Depth_cm', 'Soil_Moisture_%', 'Tank_Level_%',
    'ET_mm_day', 'Rainfall_Predicted_mm', 'Crop_Stage'
]

# ============================================================
# HARDCODED TEST SAMPLES
# ============================================================

test_samples = np.array([
    [3.0,  55.0, 80.0, 4.0,  5.0,  1],
    [5.0,  40.0, 65.0, 7.0,  1.0,  2],
    [8.0,  60.0, 50.0, 5.5,  0.0,  3],
    [10.0, 70.0, 45.0, 3.5,  8.0,  4],
    [12.0, 75.0, 30.0, 2.5, 12.0,  5],
    [4.0,  35.0, 70.0, 8.0,  0.5,  2],
    [6.0,  50.0, 60.0, 6.0,  3.0,  3],
    [2.0,  65.0, 90.0, 3.0, 10.0,  1],
], dtype=np.float64)   # sklearn models expect float64

single_sample = test_samples[[0]]   # shape (1, 6)

print("=" * 65)
print("RASPBERRY PI 4 -- HYBRID STACK BENCHMARK")
print("ML-Based Predictive Irrigation System for Paddy Cultivation")
print("=" * 65)
print(f"Timestamp      : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Platform       : {os.uname().machine}")
print(f"CPU cores      : {psutil.cpu_count(logical=False)} physical")
print(f"Total RAM      : {psutil.virtual_memory().total / 1024**2:.0f} MB")
print(f"Available RAM  : {psutil.virtual_memory().available / 1024**2:.0f} MB")
print("-" * 65)

# ============================================================
# LOAD MODELS
# ============================================================

print("\nLoading hybrid stack models...")

for path in [family1_path, family2_path, family3_path, meta_path]:
    assert os.path.exists(path), f"Missing: {path}"

family1 = joblib.load(family1_path)
family2 = joblib.load(family2_path)
family3 = joblib.load(family3_path)
meta    = joblib.load(meta_path)

f1_size_kb   = os.path.getsize(family1_path) / 1024
f2_size_kb   = os.path.getsize(family2_path) / 1024
f3_size_kb   = os.path.getsize(family3_path) / 1024
meta_size_kb = os.path.getsize(meta_path)    / 1024
total_size_kb = f1_size_kb + f2_size_kb + f3_size_kb + meta_size_kb

print(f"family1_champion_retrained : {f1_size_kb:.2f} KB")
print(f"family2_champion_retrained : {f2_size_kb:.2f} KB")
print(f"family3_champion_retrained : {f3_size_kb:.2f} KB")
print(f"meta_learner_ridge         : {meta_size_kb:.2f} KB")
print(f"Total stack size           : {total_size_kb:.2f} KB")

# ============================================================
# STACK INFERENCE HELPER
# ============================================================

import pandas as pd

def stack_predict(X):
    """Full hybrid stack inference: 3 champions → meta-learner → clipped output."""
    df = pd.DataFrame(X, columns=FEATURE_COLUMNS)
    p1 = family1.predict(df).reshape(-1, 1)
    p2 = family2.predict(df).reshape(-1, 1)
    p3 = family3.predict(df).reshape(-1, 1)
    Z  = np.hstack([p1, p2, p3])
    return np.clip(meta.predict(Z), 0, None)

# ============================================================
# PREDICTION VERIFICATION
# ============================================================

print("\n" + "=" * 65)
print("PREDICTION VERIFICATION (8 Test Samples)")
print("=" * 65)

stack_preds = stack_predict(test_samples)

print(f"\n{'Sample':<8} {'Stack Prediction (mm)':>22}")
print("-" * 32)
for i, pred in enumerate(stack_preds):
    print(f"{i+1:<8} {pred:>22.4f}")

# ============================================================
# LATENCY BENCHMARK
# ============================================================

print("\n" + "=" * 65)
print("LATENCY BENCHMARK (Single Sample, 500 runs)")
print("=" * 65)

# Warmup
for _ in range(N_WARMUP):
    stack_predict(single_sample)

times = []
for _ in range(N_REPEATS):
    t0 = time.perf_counter()
    stack_predict(single_sample)
    times.append((time.perf_counter() - t0) * 1000)

lat_stack = {
    'mean_ms': np.mean(times),
    'std_ms':  np.std(times),
    'min_ms':  np.min(times),
    'p50_ms':  np.percentile(times, 50),
    'p95_ms':  np.percentile(times, 95),
    'p99_ms':  np.percentile(times, 99),
    'max_ms':  np.max(times)
}

print(f"\n{'Metric':<10} {'Hybrid Stack (ms)':>18}")
print("-" * 30)
for m in ['mean_ms','std_ms','min_ms','p50_ms','p95_ms','p99_ms','max_ms']:
    label = m.replace('_ms','').upper()
    print(f"{label:<10} {lat_stack[m]:>18.4f}")

# ============================================================
# RAM BENCHMARK -- Model Memory Footprint
# ============================================================

print("\n" + "=" * 65)
print("RAM USAGE BENCHMARK (Model Load Footprint)")
print("=" * 65)

gc.collect()
proc   = psutil.Process(os.getpid())
before = proc.memory_info().rss / 1024

# Reload all four models fresh to measure combined footprint
_ = joblib.load(family1_path)
_ = joblib.load(family2_path)
_ = joblib.load(family3_path)
_ = joblib.load(meta_path)

ram_stack = max(proc.memory_info().rss / 1024 - before, 0)

print(f"Hybrid Stack RAM footprint : {ram_stack:.2f} KB")

# ============================================================
# SUMMARY TABLE
# ============================================================

print("\n" + "=" * 65)
print("SUMMARY")
print("=" * 65)
print(f"{'Metric':<25} {'Hybrid Stack':>18}")
print("-" * 45)
print(f"{'Total File Size (KB)':<25} {total_size_kb:>18.2f}")
print(f"{'Latency Mean (ms)':<25} {lat_stack['mean_ms']:>18.4f}")
print(f"{'Latency P95 (ms)':<25} {lat_stack['p95_ms']:>18.4f}")
print(f"{'Latency P99 (ms)':<25} {lat_stack['p99_ms']:>18.4f}")
print(f"{'RAM Usage (KB)':<25} {ram_stack:>18.2f}")
print("-" * 45)

# ============================================================
# SAVE JSON
# ============================================================

results = {
    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    'platform':  os.uname().machine,
    'cpu_cores': psutil.cpu_count(logical=False),
    'total_ram_mb': psutil.virtual_memory().total / 1024**2,
    'hybrid_stack': {
        'file_sizes_kb': {
            'family1': round(f1_size_kb, 2),
            'family2': round(f2_size_kb, 2),
            'family3': round(f3_size_kb, 2),
            'meta':    round(meta_size_kb, 2),
            'total':   round(total_size_kb, 2)
        },
        'latency':  {k: round(v, 4) for k, v in lat_stack.items()},
        'ram_kb':   round(ram_stack, 2),
        'sample_predictions': stack_preds.tolist()
    }
}

with open('./pi_results_stack.json', 'w') as f:
    json.dump(results, f, indent=4)

print("\nResults saved : pi_results_stack.json")
print("Benchmark complete.")
