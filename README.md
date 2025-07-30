# Major Assignment MLOps : California Housing Linear Regression MLOps Pipeline

## Overview

This project demonstrates a complete MLOps workflow for a **Linear Regression** model trained on the California Housing dataset. The pipeline includes:

- Modularized code using `utils.py`
- Model training, saving, and evaluation
- Manual quantization of model parameters to unsigned 8-bit integers
- Performance comparison between raw and quantized models
- Docker containerization
- Automated testing and CI/CD using GitHub Actions

All configurable parameters are centralized in `config.json` for flexibility and reproducibility.

---


## Project Structure

```
.
├── .github/
│   └── workflows/
|       └── ci.yml
├── src/
|   ├── config.json
│   ├── train.py
│   ├── quantize.py
│   ├── predict.py
│   └── utils.py
├── tests/
│   └── test_train.py
├── .gitignore
├── Dockerfile
├── README.md
└── requirements.txt
```

---

## Setup Instructions

### 1. Install Dependencies
```
python -m pip install --upgrade pip
pip install -r requirements.txt
```
### 2. Train the Model
```
python src/train.py
```
### 3. Quantize Model Parameters
```
python src/quantize.py
```
### 4. Make Predictions
```
python src/predict.py
```
### 5. Run Automated Tests
```
pytest
```
---

## CI/CD Pipeline Overview

The GitHub Actions workflow includes the following jobs:

| Job Name       | Description                                                   | Depends On        |
|:--------------|:-------------------------------------------------------------|:-----------------|
| **test_suite** | Runs pytest tests to validate the pipeline                    | None              |
| **train_and_quantize** | Trains the Linear Regression model, runs quantization, uploads artifacts | `test_suite`      |
| **build_and_test_container** | Builds Docker image, runs container with prediction script, pushes Docker image | `train_and_quantize` |

---

## Configuration (`config.json`)

All hyperparameters and pipeline settings are stored in `config.json`.

---

## Comparison Table

| Metric                        | Unquantized Model       | Quantized Model       |
|:------------------------------|:------------------------|:----------------------|
| **R² Score**                  | 0.5758                  | 0.5566                |
| **Mean Squared Error (Loss)** | 0.5559                  | 0.5810                |
| **File Size**                 | 0.33                    | 0.32                  |
| **File Name**                 | `unquant_params.joblib` | `quant_params.joblib` |

---