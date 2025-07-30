import os
import numpy as np
import pytest
from sklearn.linear_model import LinearRegression
from src.utils import load_dataset, create_model, save_model, load_model, calculate_metrics

def test_dataset_loading():
    X_train, X_test, y_train, y_test = load_dataset()
    assert X_train.shape[1] == 8
    total = len(X_train) + len(X_test)
    ratio = len(X_train) / total
    assert 0.75 <= ratio <= 0.85

def test_model_creation_and_training():
    mdl = create_model()
    assert isinstance(mdl, LinearRegression)
    X_train, X_test, y_train, y_test = load_dataset()
    mdl.fit(X_train, y_train)
    assert hasattr(mdl, "coef_") and mdl.coef_.shape == (8,)

def test_metrics():
    X_train, X_test, y_train, y_test = load_dataset()
    mdl = create_model()
    mdl.fit(X_train, y_train)
    r2, mse = calculate_metrics(y_test, mdl.predict(X_test))
    assert r2 > 0.5
    assert mse > 0

def test_save_and_load(tmp_path):
    X_train, X_test, y_train, y_test = load_dataset()
    mdl = create_model()
    mdl.fit(X_train, y_train)
    path = tmp_path / "model.joblib"
    save_model(mdl, str(path))
    loaded = load_model(str(path))
    np.testing.assert_allclose(mdl.coef_, loaded.coef_)