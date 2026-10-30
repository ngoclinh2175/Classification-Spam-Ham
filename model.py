from sklearn.svm import SVC
from vectorization import X_train_tfidf, X_test_tfidf, y_train, y_test, vectorizer


model = SVC(
    kernel="linear",    
    C=1.0,            
    random_state=42
)

model.fit(X_train_tfidf, y_train)
print("Huấn luyện xong!")

print("Số support vector của mỗi lớp [ham, spam]:", model.n_support_)
print("Tổng số support vector:", model.n_support_.sum(), "/", X_train_tfidf.shape[0], "mẫu train")

du_doan = model.predict(X_test_tfidf[:10])
print("Dự đoán:", du_doan)
print("Thực tế:", y_test.iloc[:10].tolist())
