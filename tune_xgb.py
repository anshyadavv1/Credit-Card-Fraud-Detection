import optuna
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold
import numpy as np
from xgboost import XGBClassifier

from src.preprocessing import*
from src.train import train_model
from sklearn.model_selection import StratifiedKFold
import numpy as np
df = load_data("data/raw/creditcard.csv")
df= remove_duplicates(df)
x_train, x_test, y_train, y_test = split_data(df)
x_train_scaled, x_test_scaled,scaler = feature_scaling(x_train, x_test)

#compute imbalance ratio

scale_pos_weight = y_train.value_counts()[0] / y_train.value_counts()[1]



def objective(trial):
    model = XGBClassifier(
        n_estimators=trial.suggest_int("n_estimators", 100, 500),
        max_depth=trial.suggest_int("max_depth", 3, 10),
        learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        subsample=trial.suggest_float("subsample", 0.6, 1.0),
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.6, 1.0),
        min_child_weight=trial.suggest_int("min_child_weight", 1, 10),
        gamma=trial.suggest_float("gamma", 0.0, 5.0),
        reg_alpha=trial.suggest_float("reg_alpha", 0.0, 5.0),
        reg_lambda=trial.suggest_float("reg_lambda", 0.1, 10.0),
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        eval_metric="logloss",
    )
    skf = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )

    scores = []

    for train_idx, val_idx in skf.split(x_train_scaled, y_train):
        X_train_fold = x_train_scaled.iloc[train_idx]
        X_val_fold = x_train_scaled.iloc[val_idx]

        y_train_fold = y_train.iloc[train_idx]
        y_val_fold = y_train.iloc[val_idx]

        trained_model = train_model(
            model,
            X_train_fold,
            y_train_fold,
        )

        y_prob = trained_model.predict_proba(X_val_fold)[:, 1]

        score = average_precision_score(
            y_val_fold,
            y_prob,
        )

        scores.append(score)

    return np.mean(scores)
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=100)
print("Best PR-AUC",study.best_value)
print("\nBest Params:")
print(study.best_params)