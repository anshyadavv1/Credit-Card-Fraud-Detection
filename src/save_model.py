import os
import joblib


def save_artifacts(model,scaler,artifact_dir="artifacts"):
    os.makedirs(artifact_dir, exist_ok=True)

    model_path = os.path.join(artifact_dir, "model.pkl")
    scaler__path = os.path.join(artifact_dir, "scaler.pkl")

    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler__path)

    print(f"Saved model to: {model_path}")
    print(f"Saved scaler to: {scaler__path}")