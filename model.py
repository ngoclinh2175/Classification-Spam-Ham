from sklearn.svm import SVC
from vectorization import X_train_tfidf, X_test_tfidf, y_train, y_test, vectorizer

# ====================== 1. KHỞI TẠO MÔ HÌNH ======================
model = SVC(
    kernel="linear",    # ranh giới là đường thẳng / siêu phẳng
    C=1.0,              # mức độ "nghiêm khắc" khi gặp điểm bị phân loại sai
    random_state=42
)

# ====================== 2. HUẤN LUYỆN ======================
model.fit(X_train_tfidf, y_train)
print("Huấn luyện xong!")

# ====================== 3. XEM MÔ HÌNH HỌC ĐƯỢC GÌ ======================
print("Số support vector của mỗi lớp [ham, spam]:", model.n_support_)
print("Tổng số support vector:", model.n_support_.sum(), "/", X_train_tfidf.shape[0], "mẫu train")

# ====================== 4. THỬ DỰ ĐOÁN 5 BÌNH LUẬN TEST ĐẦU TIÊN ======================
du_doan = model.predict(X_test_tfidf[:10])
print("Dự đoán:", du_doan)
print("Thực tế:", y_test.iloc[:10].tolist())