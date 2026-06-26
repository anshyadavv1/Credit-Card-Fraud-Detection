from sklearn.metrics import (accuracy_score,
precision_score, recall_score, f1_score, roc_auc_score,
average_precision_score,confusion_matrix,classification_report)



def evaluate(model, x_test, y_test, threshold=0.5):
    y_prob = model.predict_proba(x_test)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)



    metrics = {

        "Accuracy": accuracy_score(y_test, y_pred),

        "Precision": precision_score(y_test, y_pred),

        "Recall": recall_score(y_test, y_pred),

        "F1 Score": f1_score(y_test, y_pred),

        "ROC-AUC": roc_auc_score(y_test, y_prob),

        "PR-AUC": average_precision_score(y_test, y_prob),

    }
    print("=" * 50)

    print("Model Evaluation")

    print("=" * 50)

    for metric,value in metrics.items():
        print(f"{metric}: {value:.4f}")
    print("\nConfusion Matrix")

    print(confusion_matrix(y_test, y_pred))

    print("\nClassification Report")

    print(classification_report(y_test, y_pred))

    return metrics