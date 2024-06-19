from flask import Flask, request, jsonify
import pandas as pd
import json
from nltk.tokenize import word_tokenize
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import joblib
import logging
import time
import mysql.connector
from datetime import datetime
from flask_cors import CORS

app = Flask(__name__)
CORS(app) 

# Database connection configuration
DB_CONFIG = {
    'user': 'root',
    'password': '',
    'host': '127.0.0.1',
    'database': 'sentiment',
    'raise_on_warnings': True
}

# Preprocessing function
def preprocessing(data):
    data['review_original'] = data['review']  # Simpan review asli sebelum preprocessing
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

# Load model dan vectorizer
model = joblib.load('model/knn_model.pkl')
tfidf_vectorizer = joblib.load('model/tfidf_vectorizer.pkl')

# Inisialisasi Logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger('flask_app')

@app.route('/predict', methods=['POST'])
def predict():
    start_time = time.time()
    data_review_baru = request.json
    logger.debug(f"Received request: {data_review_baru}")
    
    # Convert the single JSON object to DataFrame
    data_review_baru = pd.DataFrame([data_review_baru])
    
    # Ambil rating dari request JSON
    rating = data_review_baru['rating']  # Pastikan ini sesuai dengan key yang Anda gunakan di frontend

    # Preprocess review
    data_review_baru_preprocessed = preprocessing(data_review_baru)
    
    # Transformasi menggunakan TF-IDF Vectorizer
    data_review_baru_tfidf = tfidf_vectorizer.transform(data_review_baru_preprocessed['review'].apply(lambda x: ' '.join(x)))
    
    # Lakukan prediksi menggunakan model
    hasil_prediksi = model.predict(data_review_baru_tfidf)
    
    # Tambahkan hasil prediksi ke DataFrame
    data_review_baru['klasifikasi'] = hasil_prediksi

    # Tambahkan tanggal saat review dimasukkan
    current_date = datetime.now().strftime('%Y-%m-%d')
    data_review_baru['date'] = current_date
    
    # Tambahkan rating ke DataFrame
    data_review_baru['rating'] = rating

    # Gabungkan review asli dan hasil klasifikasi dalam response
    response = data_review_baru[['id_product', 'user', 'review_original', 'date', 'klasifikasi', 'rating']].copy()
    response.rename(columns={'review_original': 'review'}, inplace=True)
    
    # Atur urutan kolom sesuai yang diminta
    response = response[['id_product', 'user', 'review', 'date', 'klasifikasi', 'rating']]

    # Simpan ke database
    conn = None
    cursor = None
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        cursor = conn.cursor()
        for index, row in response.iterrows():
            cursor.execute(
                "INSERT INTO module_review (id_product, user, review, date, klasifikasi, rating) VALUES (%s, %s, %s, %s, %s, %s)",
                (row['id_product'], row['user'], row['review'], row['date'], row['klasifikasi'], row['rating'])
            )
        conn.commit()
        logger.debug("Data inserted into database successfully.")
    except mysql.connector.Error as err:
        logger.error(f"Error: {err}")
        return jsonify({"error": str(err)}), 500
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()

    logger.debug(f"Processed request in {time.time() - start_time} seconds")
    response_data = response.to_dict(orient='records')[0]
    return jsonify({"data": response_data, "message": "Data inserted into database successfully."})


if __name__ == '__main__':
    app.run(debug=True, port=5000)
