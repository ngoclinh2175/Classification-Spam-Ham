import pandas as pd
import re

df = pd.read_csv("E:\Machine Learning\Phan_loai_thu_rac/reviews_labeled.csv")


df = df[["review", "label"]]

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
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"www\.\S+", " ", text)
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

