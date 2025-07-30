from utils import load_dataset, create_model, save_model, calculate_metrics

def main():
    X_train, X_test, y_train, y_test = load_dataset()
    model = create_model()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    r2, mse = calculate_metrics(y_test, y_pred)
    print(f"R² Score: {r2:.4f}")
    print(f"Loss/Mean Squared Error: {mse:.4f}")
    save_model(model, "models/linear_regression_model.joblib")
    print("Model saved to models/linear_regression_model.joblib")
    return r2, mse

if __name__ == "__main__":
    main()