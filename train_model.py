import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import joblib
import json
import nltk
from nltk.tokenize import word_tokenize

# Load dataset
dataset = pd.read_csv('model/dataset_balance1130.csv', delimiter=';')

# Preprocessing function
def preprocessing(data):
    data['review'] = data['review'].str.lower()
    data['review'] = data['review'].str.replace(r'[^a-zA-Z\s]', '', regex=True)
    data['review'] = data['review'].apply(word_tokenize)
    with open('model/combined_slang_words.txt', 'r', encoding='utf-8') as file:
        kamus_normalisasi = json.load(file)
    data['review'] = data['review'].apply(lambda x: [kamus_normalisasi.get(kata, kata) for kata in x])
    with open('model/combined_stop_words.txt', 'r', encoding='utf-8') as file:
        stop_words = set(file.read().splitlines())
        stop_words.update(['sih', 'siih', 'si', 'sii', 'siii', 'iya', 'ya', 'yaa', 'nya', 'ku', 'yg', 'tp', 'gtu', 'deh', 'tuh', 'tuhh', 'tuuhh', 'ituuuuuu', 'iniii', 'hehe', 'he', 'e', 'eh', 'ehh', 'jg', 'ku', 'lah', 'laah', 'an', 'nge', 'kak', 'wkwk', 'wkwkwk', 'haha', 'hahaha', 'hahahaha', 'hhi', 'hihi', 'hihihi', 'hihihihi', 'hehe', 'hehehe', 'hehehehe', 'an', 'we', 'weh', 'yuk', 'yuuk', 'aaaaa', 'aaaaaaaaa', 'aaaaakkkkk', 'aaah', 'ahhh', 'aaaa', 'aaaaaa', 'aaaaaakkkkk'])
    data['review'] = data['review'].apply(lambda x: [word for word in x if word not in stop_words])
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()
    data['review'] = data['review'].apply(lambda x: [stemmer.stem(word) for word in x])
    return data

# Preprocessing
data_preprocessed = preprocessing(dataset)

# Compute TF-IDF
def compute_tfidf(data):
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(data['review'].apply(lambda x: ' '.join(x)))
    tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf_vectorizer.get_feature_names_out())
    return tfidf_vectorizer, tfidf_df

tfidf_vectorizer, data_tfidf = compute_tfidf(data_preprocessed)

# Apply KNN
def apply_knn(data_tfidf, dataset):
    model = KNeighborsClassifier(n_neighbors=7, metric='euclidean')
    X_train = data_tfidf
    y_train = dataset['klasifikasi']
    model.fit(X_train, y_train)
    y_pred = model.predict(X_train)
    dataset['Hasil Prediksi KNN'] = y_pred
    hasil_klasifikasi = pd.concat([dataset], axis=1)
    return X_train, y_train, model, hasil_klasifikasi

X_train, y_train, model, hasil_klasifikasi = apply_knn(data_tfidf, dataset)

# Save model and vectorizer
joblib.dump(model, 'model/knn_model.pkl')
joblib.dump(tfidf_vectorizer, 'model/tfidf_vectorizer.pkl')

print("Model and vectorizer saved successfully!")
