from underthesea import word_tokenize
from clean_data import df

def tokenize_vi(text):
    return word_tokenize(text, format="text")

df["noi_dung_token"] = df["noi_dung_sach"].apply(tokenize_vi)

stopwords = {
    "và", "là", "của", "cho", "với", "như", "rằng", "mà", "cũng",
    "đã", "đang", "sẽ", "vừa", "rất", "lắm", "quá", "các", "những",
    "một", "này", "đó", "ấy", "nên", "bị", "được", "ở", "tại",
    "trong", "ngoài", "khi", "lúc", "ạ", "ơi", "à", "ừ", "nha",
    "hả", "luôn", "thật", "hơi",
}

def remove_stopwords(text):
    return " ".join(w for w in text.split() if w not in stopwords)

df["noi_dung_final"] = df["noi_dung_token"].apply(remove_stopwords)

print(df.shape)
print(df[["noi_dung_token", "noi_dung_final"]].head(10))
