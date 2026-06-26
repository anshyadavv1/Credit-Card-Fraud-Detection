from sklearn.ensemble import RandomForestClassifier,StackingClassifier
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
def  build_random_forest():
    rf = RandomForestClassifier(class_weight="balanced",random_state=42)
    return rf
def build_xgboost(scale_pos_weight=1.0):
    return XGBClassifier(
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        eval_metric="logloss",
    )

def build_mlp():
    mlp = MLPClassifier(random_state=42)
    return mlp
def build_stacking(scale_pos_weight=1.0):
    estimators= [
        ("rf", build_random_forest()),
        ("xgb", build_xgboost(scale_pos_weight=scale_pos_weight))

    ]
    final_estimator= build_mlp()
    stacking= StackingClassifier(estimators=estimators,final_estimator=final_estimator,
                                 stack_method="predict_proba",
                                 cv=5,
                                 n_jobs=-1
                                 )
    return stacking

def get_model(model_name, **kwargs):
    models = {
        "rf": build_random_forest,
        "mlp": build_mlp,
        "xgb": build_xgboost,
        "stacking": build_stacking
    }
    if model_name not in models:
        raise ValueError(f"Unknown model: {model_name}")

    return models[model_name](**kwargs)