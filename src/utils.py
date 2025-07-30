import os
import json
import joblib
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split

def load_config(path="src/config.json"):
    with open(path, "r") as f:
        return json.load(f)

config = load_config()

def load_dataset():
    data = fetch_california_housing()
    ts = config["dataset"]["test_size"]
    rs = config["dataset"]["random_state"]
    return train_test_split(data.data, data.target, test_size=ts, random_state=rs)

def create_model():
    opts = config["model"]
    return LinearRegression(fit_intercept=opts["fit_intercept"])

def save_model(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)

def load_model(path):
    return joblib.load(path)

def calculate_metrics(y_true, y_pred):
    return r2_score(y_true, y_pred), mean_squared_error(y_true, y_pred)

def quantize_to_uint8(arr: np.ndarray):
    # Only 8-bit currently supported
    bit = config["quantization"]["bit_width"]
    if bit != 8:
        raise ValueError("Only 8-bit quantization is supported")
    w = arr.astype(np.float32)
    w_min, w_max = w.min(), w.max()
    if w_max == w_min:
        q = np.full_like(w, 127, dtype=np.uint8)
        return q, w_min, w_max
    q = np.round((w - w_min) / (w_max - w_min) * 255).astype(np.uint8)
    return q, w_min, w_max

def dequantize_from_uint8(q: np.ndarray, w_min, w_max):
    if w_max == w_min:
        return np.full(q.shape, w_min, dtype=np.float32)
    return (q.astype(np.float32) / 255) * (w_max - w_min) + w_min