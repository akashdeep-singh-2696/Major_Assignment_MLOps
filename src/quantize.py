import os
import joblib
import numpy as np
from utils import load_model, load_dataset, calculate_metrics

def zero_point_quantize(arr: np.ndarray):
    max_abs = float(np.max(np.abs(arr)))
    if max_abs == 0:
        q = np.full(arr.shape, 128, dtype=np.uint8)
        scale = 1.0
    else:
        scale = max_abs
        q = np.round((arr / scale) * 127 + 128).astype(np.uint8)
    return q, scale

def zero_point_dequantize(q: np.ndarray, scale: float):
    return ((q.astype(np.float32) - 128) / 127.0) * scale

def print_comparison_table(size_raw, size_quant, r2_orig, r2_dq, mse_orig, mse_dq, raw_path="models/unquant_params.joblib", quant_path="models/quant_params.joblib"):
    headers = ["Metric", "Raw Parameters", "Quantized Parameters"]
    col_widths = [24, 28, 29]

    def pad_str(s, width):
        return str(s).ljust(width)

    print("\n## Comparison: Raw vs. Quantized Parameters\n")
    header_row = "| " + " | ".join(pad_str(h, w) for h, w in zip(headers, col_widths)) + " |"
    separator = "|" + "|".join("-" * (w + 2) for w in col_widths) + "|"
    print(header_row)
    print(separator)

    rows = [
        ("File path", raw_path, quant_path),
        ("File size (KB)", f"{size_raw/1024:.2f}", f"{size_quant/1024:.2f}"),
        ("R² Score", f"{r2_orig:.4f}", f"{r2_dq:.4f}"),
        ("Mean Squared Error (MSE)", f"{mse_orig:.4f}", f"{mse_dq:.4f}")
    ]

    for metric, raw_val, quant_val in rows:
        print("| " + pad_str(metric, col_widths[0]) + " | " + pad_str(raw_val, col_widths[1]) + " | " + pad_str(quant_val, col_widths[2]) + " |")

def main():
    os.makedirs("models", exist_ok=True)
    model = load_model("models/linear_regression_model.joblib")

    coef = model.coef_.astype(np.float32)
    intercept = np.array([model.intercept_], dtype=np.float32)

    raw_params_path = "models/unquant_params.joblib"
    joblib.dump({"coef": coef, "intercept": intercept}, raw_params_path)

    q_coef, coef_scale = zero_point_quantize(coef)
    q_int, int_scale = zero_point_quantize(intercept)

    quant_params_path = "models/quant_params.joblib"
    quant_params = {
        "q_coef": q_coef,
        "coef_scale": float(coef_scale),
        "q_intercept": int(q_int[0]),
        "intercept_scale": float(int_scale)
    }
    joblib.dump(quant_params, quant_params_path)

    dq_coef = zero_point_dequantize(q_coef, coef_scale)
    dq_int = zero_point_dequantize(np.array([quant_params["q_intercept"]], dtype=np.uint8), int_scale)[0]

    _, X_test, _, y_test = load_dataset()
    y_pred_dq = X_test @ dq_coef + dq_int
    y_pred_orig = model.predict(X_test)
    r2_orig, mse_orig = calculate_metrics(y_test, y_pred_orig)
    r2_dq, mse_dq = calculate_metrics(y_test, y_pred_dq)

    size_raw = os.path.getsize(raw_params_path)
    size_quant = os.path.getsize(quant_params_path)

    print_comparison_table(
        size_raw, size_quant, r2_orig, r2_dq, mse_orig, mse_dq,
        raw_path=raw_params_path, quant_path=quant_params_path
    )

if __name__ == "__main__":
    main()