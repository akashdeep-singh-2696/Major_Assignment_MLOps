from utils import load_dataset, load_model, calculate_metrics

def main():
    model = load_model("models/linear_regression_model.joblib")
    X_train, X_test, y_train, y_test = load_dataset()
    y_pred = model.predict(X_test)
    r2, mse = calculate_metrics(y_test, y_pred)
    print(f"R² Score: {r2:.4f}")
    print(f"Mean Squared Error: {mse:.4f}\n")
    print("Sample Predictions (first 10):")
    for true, pred in zip(y_test[:10], y_pred[:10]):
        print(f"True: {true:.2f} | Pred: {pred:.2f} | Diff: {abs(true-pred):.2f}")
    return True

if __name__ == "__main__":
    main()