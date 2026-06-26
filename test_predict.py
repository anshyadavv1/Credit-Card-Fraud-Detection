import pandas as pd

from src.predict import FraudPredictor

predictor = FraudPredictor()

sample = pd.DataFrame({
    "Time": [1000],
    "V1": [0.2],
    "V2": [-0.1],
    "V3": [1.3],
    "V4": [0.5],
    "V5": [-0.4],
    "V6": [0.7],
    "V7": [0.2],
    "V8": [0.1],
    "V9": [-0.3],
    "V10": [0.8],
    "V11": [0.4],
    "V12": [-0.5],
    "V13": [0.2],
    "V14": [0.6],
    "V15": [0.1],
    "V16": [-0.2],
    "V17": [0.3],
    "V18": [0.5],
    "V19": [-0.1],
    "V20": [0.2],
    "V21": [0.1],
    "V22": [0.3],
    "V23": [-0.2],
    "V24": [0.4],
    "V25": [0.2],
    "V26": [-0.1],
    "V27": [0.1],
    "V28": [0.05],
    "Amount": [150.75]
})

print(predictor.predict(sample))