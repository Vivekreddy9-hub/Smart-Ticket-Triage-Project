from sklearn.metrics import classification_report, accuracy_score

def print_metrics(y_true, y_pred):
    print('Accuracy:', accuracy_score(y_true, y_pred))
    print(classification_report(y_true, y_pred))
