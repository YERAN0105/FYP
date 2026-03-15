import numpy as np
import onnxruntime as rt

# Load INT8 ONNX model once at module level
try:
    sess = rt.InferenceSession('./student_model_int8.onnx',
                               providers=['CPUExecutionProvider'])
    input_name  = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name
except Exception as e:
    print(f"ERROR [predict_cwr]: Failed to load ONNX model - {e}")
    sess = None

def predict_cwr(water_depth_cm, soil_moisture_pct, tank_level_pct,
                et_mm_day, rainfall_predicted_mm, crop_stage):
    # Predicts Crop Water Requirement in mm using INT8 ONNX student model
    # Input order must match exactly: Water_Depth_cm, Soil_Moisture_%,
    # Tank_Level_%, ET_mm_day, Rainfall_Predicted_mm, Crop_Stage
    # Returns CWR in mm, or None if model not loaded

    if sess is None:
        print("ERROR [predict_cwr]: ONNX model not loaded")
        return None

    try:
        sample = np.array([[
            water_depth_cm,
            soil_moisture_pct,
            tank_level_pct,
            et_mm_day,
            rainfall_predicted_mm,
            crop_stage
        ]], dtype=np.float32)

        prediction = sess.run([output_name], {input_name: sample})[0]
        cwr_mm = max(0.0, float(prediction[0][0]))
        return round(cwr_mm, 2)

    except Exception as e:
        print(f"ERROR [predict_cwr]: Inference failed - {e}")
        return None
