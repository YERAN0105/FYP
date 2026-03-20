# ============================================================
# RASPBERRY PI 4 - ONNX + SKLEARN BENCHMARK SCRIPT
# ML-Based Predictive Irrigation System for Paddy Cultivation
# Benchmarks: Student MLP sklearn | ONNX FP32 | ONNX INT8
#
# Required files in same folder:
#   student_model.pkl
#   student_model_fp32.onnx
#   student_model_int8.onnx
#
# Run: python pi_benchmark_onnx.py
# ============================================================

import numpy as np
import onnxruntime as rt
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

N_WARMUP      = 10
N_REPEATS     = 500
TOLERANCE_MM  = 0.5

sklearn_path = './student_model_sklearn.pkl'
fp32_path    = './student_model_fp32.onnx'
int8_path    = './student_model_int8.onnx'

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
], dtype=np.float32)

single_sample     = test_samples[[0]]
single_sample_f64 = single_sample.astype(np.float64)

print("=" * 65)
print("RASPBERRY PI 4 -- ONNX + SKLEARN BENCHMARK")
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

print("\nLoading models...")

assert os.path.exists(sklearn_path), f"Missing: {sklearn_path}"
assert os.path.exists(fp32_path),    f"Missing: {fp32_path}"
assert os.path.exists(int8_path),    f"Missing: {int8_path}"

sklearn_model = joblib.load(sklearn_path)

sess_fp32 = rt.InferenceSession(fp32_path, providers=['CPUExecutionProvider'])
sess_int8 = rt.InferenceSession(int8_path, providers=['CPUExecutionProvider'])

input_name_fp32  = sess_fp32.get_inputs()[0].name
output_name_fp32 = sess_fp32.get_outputs()[0].name
input_name_int8  = sess_int8.get_inputs()[0].name
output_name_int8 = sess_int8.get_outputs()[0].name

sklearn_size_kb = os.path.getsize(sklearn_path) / 1024
fp32_size_kb    = os.path.getsize(fp32_path) / 1024
int8_size_kb    = os.path.getsize(int8_path) / 1024

print(f"sklearn loaded : {sklearn_size_kb:.2f} KB")
print(f"FP32 loaded    : {fp32_size_kb:.2f} KB")
print(f"INT8 loaded    : {int8_size_kb:.2f} KB")

# ============================================================
# PREDICTION VERIFICATION
# ============================================================

print("\n" + "=" * 65)
print("PREDICTION VERIFICATION (8 Test Samples)")
print("=" * 65)

sklearn_preds = np.clip(
    sklearn_model.predict(test_samples.astype(np.float64)).flatten(), 0, None)
fp32_preds = np.clip(
    sess_fp32.run([output_name_fp32], {input_name_fp32: test_samples})[0].flatten(), 0, None)
int8_preds = np.clip(
    sess_int8.run([output_name_int8], {input_name_int8: test_samples})[0].flatten(), 0, None)

print(f"\n{'Sample':<8} {'sklearn (mm)':>14} {'FP32 (mm)':>12} {'INT8 (mm)':>12} "
      f"{'sk-FP32':>10} {'FP32-INT8':>12}")
print("-" * 72)
for i, (sk_p, fp32_p, int8_p) in enumerate(zip(sklearn_preds, fp32_preds, int8_preds)):
    print(f"{i+1:<8} {sk_p:>14.4f} {fp32_p:>12.4f} {int8_p:>12.4f} "
          f"{abs(sk_p-fp32_p):>10.6f} {abs(fp32_p-int8_p):>12.6f}")

max_diff_sk_fp32   = np.abs(sklearn_preds - fp32_preds).max()
max_diff_fp32_int8 = np.abs(fp32_preds - int8_preds).max()
print("-" * 72)
print(f"Max sklearn vs FP32 diff : {max_diff_sk_fp32:.6f} mm  "
      + ("PASSED" if max_diff_sk_fp32 < TOLERANCE_MM else "WARNING"))
print(f"Max FP32 vs INT8 diff    : {max_diff_fp32_int8:.6f} mm  "
      + ("PASSED" if max_diff_fp32_int8 < TOLERANCE_MM else "WARNING"))

# ============================================================
# LATENCY BENCHMARK
# ============================================================

print("\n" + "=" * 65)
print("LATENCY BENCHMARK (Single Sample, 500 runs)")
print("=" * 65)

def benchmark_sklearn(model, sample):
    for _ in range(N_WARMUP):
        model.predict(sample)
    times = []
    for _ in range(N_REPEATS):
        t0 = time.perf_counter()
        model.predict(sample)
        times.append((time.perf_counter() - t0) * 1000)
    return {
        'mean_ms': np.mean(times),  'std_ms': np.std(times),
        'min_ms':  np.min(times),   'p50_ms': np.percentile(times, 50),
        'p95_ms':  np.percentile(times, 95),
        'p99_ms':  np.percentile(times, 99),
        'max_ms':  np.max(times)
    }

def benchmark_onnx(sess, input_name, output_name, sample):
    for _ in range(N_WARMUP):
        sess.run([output_name], {input_name: sample})
    times = []
    for _ in range(N_REPEATS):
        t0 = time.perf_counter()
        sess.run([output_name], {input_name: sample})
        times.append((time.perf_counter() - t0) * 1000)
    return {
        'mean_ms': np.mean(times),  'std_ms': np.std(times),
        'min_ms':  np.min(times),   'p50_ms': np.percentile(times, 50),
        'p95_ms':  np.percentile(times, 95),
        'p99_ms':  np.percentile(times, 99),
        'max_ms':  np.max(times)
    }

print("Benchmarking sklearn MLP...")
lat_sklearn = benchmark_sklearn(sklearn_model, single_sample_f64)
print("Benchmarking ONNX FP32...")
lat_fp32    = benchmark_onnx(sess_fp32, input_name_fp32, output_name_fp32, single_sample)
print("Benchmarking ONNX INT8...")
lat_int8    = benchmark_onnx(sess_int8, input_name_int8, output_name_int8, single_sample)

print(f"\n{'Metric':<10} {'sklearn (ms)':>14} {'FP32 (ms)':>12} {'INT8 (ms)':>12}")
print("-" * 52)
for m in ['mean_ms','std_ms','min_ms','p50_ms','p95_ms','p99_ms','max_ms']:
    label = m.replace('_ms','').upper()
    print(f"{label:<10} {lat_sklearn[m]:>14.4f} {lat_fp32[m]:>12.4f} {lat_int8[m]:>12.4f}")

# ============================================================
# RAM BENCHMARK -- Model Memory Footprint
# ============================================================

print("\n" + "=" * 65)
print("RAM USAGE BENCHMARK (Model Load Footprint)")
print("=" * 65)

import gc

def measure_load_ram_sklearn(path):
    gc.collect()
    proc   = psutil.Process(os.getpid())
    before = proc.memory_info().rss / 1024
    model  = joblib.load(path)
    after  = proc.memory_info().rss / 1024
    return max(after - before, 0), model

def measure_load_ram_onnx(path):
    gc.collect()
    proc   = psutil.Process(os.getpid())
    before = proc.memory_info().rss / 1024
    sess   = rt.InferenceSession(path, providers=['CPUExecutionProvider'])
    after  = proc.memory_info().rss / 1024
    return max(after - before, 0), sess

ram_sklearn, _ = measure_load_ram_sklearn(sklearn_path)
ram_fp32,    _ = measure_load_ram_onnx(fp32_path)
ram_int8,    _ = measure_load_ram_onnx(int8_path)

print(f"sklearn RAM footprint : {ram_sklearn:.2f} KB")
print(f"FP32 RAM footprint    : {ram_fp32:.2f} KB")
print(f"INT8 RAM footprint    : {ram_int8:.2f} KB")

# ============================================================
# SUMMARY TABLE
# ============================================================

print("\n" + "=" * 65)
print("SUMMARY")
print("=" * 65)
print(f"{'Metric':<25} {'sklearn':>14} {'FP32':>12} {'INT8':>12}")
print("-" * 65)
print(f"{'File Size (KB)':<25} {sklearn_size_kb:>14.2f} {fp32_size_kb:>12.2f} {int8_size_kb:>12.2f}")
print(f"{'Latency Mean (ms)':<25} {lat_sklearn['mean_ms']:>14.4f} {lat_fp32['mean_ms']:>12.4f} {lat_int8['mean_ms']:>12.4f}")
print(f"{'Latency P95 (ms)':<25} {lat_sklearn['p95_ms']:>14.4f} {lat_fp32['p95_ms']:>12.4f} {lat_int8['p95_ms']:>12.4f}")
print(f"{'Latency P99 (ms)':<25} {lat_sklearn['p99_ms']:>14.4f} {lat_fp32['p99_ms']:>12.4f} {lat_int8['p99_ms']:>12.4f}")
print(f"{'RAM Usage (KB)':<25} {ram_sklearn:>14.2f} {ram_fp32:>12.2f} {ram_int8:>12.2f}")
print("-" * 65)

# ============================================================
# SAVE JSON
# ============================================================

results = {
    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    'platform':  os.uname().machine,
    'cpu_cores': psutil.cpu_count(logical=False),
    'total_ram_mb': psutil.virtual_memory().total / 1024**2,
    'sklearn': {
        'size_kb': round(sklearn_size_kb, 2),
        'latency': {k: round(v, 4) for k, v in lat_sklearn.items()},
        'ram_kb':  round(ram_sklearn, 2),
        'sample_predictions': sklearn_preds.tolist()
    },
    'fp32': {
        'size_kb': round(fp32_size_kb, 2),
        'latency': {k: round(v, 4) for k, v in lat_fp32.items()},
        'ram_kb':  round(ram_fp32, 2),
        'sample_predictions': fp32_preds.tolist()
    },
    'int8': {
        'size_kb': round(int8_size_kb, 2),
        'latency': {k: round(v, 4) for k, v in lat_int8.items()},
        'ram_kb':  round(ram_int8, 2),
        'sample_predictions': int8_preds.tolist()
    },
    'verification': {
        'max_sklearn_fp32_diff_mm': round(float(max_diff_sk_fp32), 6),
        'max_fp32_int8_diff_mm':   round(float(max_diff_fp32_int8), 6),
        'sklearn_fp32_passed':     bool(max_diff_sk_fp32 < TOLERANCE_MM),
        'fp32_int8_passed':        bool(max_diff_fp32_int8 < TOLERANCE_MM)
    }
}

with open('./pi_results_onnx.json', 'w') as f:
    json.dump(results, f, indent=4)

print("\nResults saved : pi_results_onnx.json")
print("Benchmark complete.")
