import unittest
import pickle
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences

MAX_LEN = 1000

# Загружаем модель, токенизатор и энкодер
with open("model/model.pkl", "rb") as f:
    model = pickle.load(f)

with open("model/tokenizer.pickle", "rb") as f:
    tokenizer = pickle.load(f)

with open("model/label_encoder.pickle", "rb") as f:
    label_encoder = pickle.load(f)


def classify_text(text):
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=MAX_LEN)
    prediction = model.predict(padded)[0]
    class_index = np.argmax(prediction)
    result = label_encoder.inverse_transform([class_index])[0]
    return result


class MinistryClassificationTest(unittest.TestCase):

    def test_health_ministry(self):
        text = "Обеспечение санитарно-эпидемиологического благополучия и охрана здоровья населения"
        result = classify_text(text)
        self.assertIn("здравоохранения", result.lower())

    def test_internal_affairs_ministry(self):
        text = "Охрана общественного порядка, обеспечение безопасности и борьба с преступностью"
        result = classify_text(text)
        self.assertIn("внутренних дел", result.lower())

    def test_water_resources_ministry(self):
        text = "Рациональное использование водных ресурсов и развитие ирригационной системы"
        result = classify_text(text)
        self.assertIn("водных ресурсов", result.lower())


if __name__ == '__main__':
    unittest.main()
