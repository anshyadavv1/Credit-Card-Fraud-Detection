from sklearn.ensemble import RandomForestClassifier,StackingClassifier
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier


def  build_random_forest():
    rf = RandomForestClassifier(random_state=42)
    return rf
def build_xgboost():
    xgb = XGBClassifier(random_state=42)
    return xgb
def build_mlp():
    mlp = MLPClassifier(random_state=42)
    return mlp
def build_stacking():
    pass