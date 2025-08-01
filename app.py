from flask import Flask, render_template, request
import pickle
import numpy as np
import pymorphy3
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.tokenize import TreebankWordTokenizer

tokenizer_nltk = TreebankWordTokenizer()

import nltk
# nltk.download('punkt')
nltk.download('stopwords')

app = Flask(__name__)

MAX_LEN = 1000

# Загрузка модели и вспомогательных объектов
with open("model/model.pkl", "rb") as f:
    model = pickle.load(f)
with open("model/tokenizer.pickle", "rb") as f:
    tokenizer = pickle.load(f)
with open("model/label_encoder.pickle", "rb") as f:
    label_encoder = pickle.load(f)

# Предобработка текста
punctuation_marks = ['!', ',', '(', ')', ':', '-', '?', '.', '..', '...', '/', '@', '1', '2', '3', '4', '5', '6', '7', '8', '9', '0', '"']
stop_words = stopwords.words("russian")
morph = pymorphy3.MorphAnalyzer()

def preprocess(text):
    tokens = tokenizer_nltk.tokenize(text.lower())
    cleaned = []
    for token in tokens:
        if token not in punctuation_marks:
            lemma = morph.parse(token)[0].normal_form
            if lemma not in stop_words:
                cleaned.append(lemma)
    return " ".join(cleaned)

# Основной маршрут
@app.route("/", methods=["GET", "POST"])
def classify():
    result = None
    probability = None

    if request.method == "POST":
        text = request.form["text"]

        # Предобработка текста
        cleaned_text = preprocess(text)

        # Токенизация
        seq = tokenizer.texts_to_sequences([cleaned_text])
        padded = pad_sequences(seq, maxlen=MAX_LEN)

        # Предсказание
        prediction = model.predict(padded)[0]
        class_index = np.argmax(prediction)
        result = label_encoder.inverse_transform([class_index])[0]
        probability = round(float(prediction[class_index]) * 100, 2)

    return render_template("index.html", result=result, probability=probability)

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8080, debug=True)
