import joblib
import pandas as pd


class FraudPredictor:
    def __init__(self,
                 model_path="artifacts/model.pkl",
                 scaler_path="artifacts/scaler.pkl"):
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)

    def predict(self, data):
        scaled_data = data.copy()

        scaled_data[["Amount", "Time"]] = self.scaler.transform(
            scaled_data[["Amount", "Time"]]
        )

        prediction = self.model.predict(scaled_data)
        probability = self.model.predict_proba(scaled_data)[:, 1]

        return {
            "prediction": prediction.tolist(),
            "probability": probability.tolist()
        }