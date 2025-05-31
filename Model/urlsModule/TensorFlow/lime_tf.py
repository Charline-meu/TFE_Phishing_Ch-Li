import tensorflow as tf
import numpy as np
import pandas as pd
from lime.lime_text import LimeTextExplainer
from collections import Counter
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import re

# ===============================
# 1. Chargement du modèle TensorFlow et des données de test
# ===============================

# Charger le modèle TensorFlow
model = tf.keras.models.load_model('urlsModule/TensorFlow/cnn_lstm_tf_model_ebu.keras')

# Charger les données de test (X_test) enregistrées dans un fichier .npy
X_test = np.load('urlsModule/TensorFlow/X_test_1.npy')

# Charger le CSV pour avoir les URL en clair
csv_path = 'urlsModule/Tensorflow/balanced_train_data.csv'
data = pd.read_csv(csv_path)
urls = data['url'].values

# ===============================
# 2. Définir le prédicteur pour LIME
# ===============================

def encode_url(url, max_len=200):
    encoded = [ord(char) for char in url if ord(char) < 128]
    return np.pad(encoded[:max_len], (0, max(0, max_len - len(encoded))), 'constant')


def predict_fn(text_list):
    encoded_urls = np.array([encode_url(text) for text in text_list])
    predictions = model.predict(encoded_urls)
    return np.vstack([(1 - predictions), predictions]).T

# ===============================
# 3. Initialisation de LIME
# ===============================

explainer = LimeTextExplainer(class_names=['Légitime', 'Phishing'])

# ===============================
# 4. Filtrer les URL classées comme phishing et Extraire les mots
# ===============================

phishing_words = []

for url in urls:
    prediction = predict_fn([url])
    predicted_label = np.argmax(prediction)
    if predicted_label == 1:
        words = re.findall(r'\b\w+\b', url)
        phishing_words.extend(words)

print(f'🔎 Nombre d\'URL classées comme phishing : {len(phishing_words)}')

# ===============================
# 5. Compter la fréquence des mots
# ===============================

word_counter = Counter(phishing_words)
most_common_words = word_counter.most_common(20)

# ===============================
# 6. Visualisation des résultats globaux
# ===============================

# Bar Plot des mots les plus fréquents
words, counts = zip(*most_common_words)
plt.figure(figsize=(12, 8))
plt.barh(words, counts, color='skyblue')
plt.xlabel('Fréquence d\'apparition')
plt.title('🔎 Mots les plus fréquents dans les URL Phishing')
plt.gca().invert_yaxis()
plt.show()

