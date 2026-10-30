import pandas as pd
import re
# 1. Đọc file
df = pd.read_csv("E:\Machine Learning\Phan_loai_thu_rac/reviews_labeled.csv")

# 2. Chỉ giữ 2 cột cần thiết
df = df[["review", "label"]]

# 3. Kiểm tra nhanh
# df["label_num"] = df["label"]#.map({"ham": 0, "spam": 1})
# print(df.shape)
# print(df.head())

teencode = {
    "ko": "không", "k": "không", "kh": "không", "hok": "không", "hông": "không",
    "dc": "được", "đc": "được", "dk": "được",
    "ib": "inbox",
    "sp": "sản phẩm",
    "nv": "nhân viên",
    "dmx": "điện máy xanh",
    "xài": "dùng",
    "oke": "ok",
}

def remove_links(text):
    # Link dạng http://..., https://...
    text = re.sub(r"https?://\S+", " ", text)
    # Link dạng www.abc.com
    text = re.sub(r"www\.\S+", " ", text)
    # Link không có http/www, ví dụ: zalo.me/abc, bit.ly/xyz, shopee.vn
    text = re.sub(r"\b\S+\.(com|vn|me|net|org|info|ly|io|xyz)\b\S*", " ", text, flags=re.IGNORECASE)
    return text

def clean_text(text):
    text = text.lower()
    text = remove_links(text)
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"_", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    words = [teencode.get(w, w) for w in text.split()]
    return " ".join(words)

df["noi_dung_sach"] = df["review"].apply(clean_text)



