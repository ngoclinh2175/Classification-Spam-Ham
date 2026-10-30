from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
from model import model                          # đổi "model" thành tên file model của bạn
from vectorization import X_test_tfidf, y_test

y_pred = model.predict(X_test_tfidf)

print("Accuracy :", round(accuracy_score(y_test, y_pred), 4))
print("Precision:", round(precision_score(y_test, y_pred), 4))
print("Recall   :", round(recall_score(y_test, y_pred), 4))
print("F1-score :", round(f1_score(y_test, y_pred), 4))

print("\nConfusion matrix (hàng: thực tế, cột: dự đoán) [ham, spam]:")
print(confusion_matrix(y_test, y_pred))

print("\n", classification_report(y_test, y_pred, target_names=["ham", "spam"], digits=4))