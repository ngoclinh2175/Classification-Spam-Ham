from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from tokenization import df 


X = df["noi_dung_final"]    
y = df["label"]        

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,        
    random_state=42,      
    stratify=y           
)

vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),   
    min_df=2,             
    max_df=0.95,         
    sublinear_tf=True    
)

X_train_tfidf = vectorizer.fit_transform(X_train)   
X_test_tfidf = vectorizer.transform(X_test)         

