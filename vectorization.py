from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from tokenization import df   # lấy df đã làm sạch, tách từ, bỏ stopword

# ====================== 1. CHỌN X VÀ y ======================
X = df["noi_dung_final"]     # văn bản (đổi thành "noi_dung_token" nếu muốn giữ stopword)
y = df["label"]          # nhãn: 0 = ham, 1 = spam

# ====================== 2. CHIA TRAIN / TEST ======================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,        # 20% làm tập test
    random_state=42,      # cố định kết quả chia, chạy lại vẫn ra như cũ
    stratify=y            # giữ tỉ lệ spam/ham giống nhau ở cả hai tập
)

# ====================== 3. VECTOR HÓA TF-IDF ======================
vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),   # dùng cả từ đơn và cặp 2 từ liền nhau
    min_df=2,             # bỏ từ chỉ xuất hiện trong < 2 bình luận
    max_df=0.95,          # bỏ từ xuất hiện trong > 95% bình luận
    sublinear_tf=True     # làm dịu ảnh hưởng của từ lặp quá nhiều lần
)

X_train_tfidf = vectorizer.fit_transform(X_train)   # học + biến đổi trên train
X_test_tfidf = vectorizer.transform(X_test)         # chỉ biến đổi trên test

# ====================== 4. KIỂM TRA ======================
# print("Số mẫu train:", X_train_tfidf.shape[0])
# print("Số mẫu test :", X_test_tfidf.shape[0])
# print("Số đặc trưng (từ vựng):", X_train_tfidf.shape[1])
# print("10 đặc trưng đầu:", vectorizer.get_feature_names_out()[:10])
