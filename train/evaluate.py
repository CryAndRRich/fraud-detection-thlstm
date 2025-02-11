from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def evaluate(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    pre, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)

    return acc, pre, rec, f1


