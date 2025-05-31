import tensorflow as tf
from keras import layers, models, optimizers
import numpy as np
import pandas as pd
import shap
import joblib
import matplotlib.pyplot as plt
import re

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# **1. Charger les datasets**
train_data_40_path = 'enron_spamassassin_nazario_nigerian_best_compo/extracted_urls/urls_train_set_40.csv'
train_ebu_csv_path = 'datasetURLs/ebubekirbbr.csv'
test_csv_path_1 = 'enron_spamassassin_nazario_nigerian_best_compo/extracted_urls/urls_test_set.csv'
test_csv_path_2 = 'enron_spamassassin_nazario_nigerian_best_compo/extracted_urls/urls_train_set_20.csv'

# Charger le dataset
train_data_40 = pd.read_csv(train_data_40_path)
test_data_1 = pd.read_csv(test_csv_path_1)
test_data_2 = pd.read_csv(test_csv_path_2)

# Fusionner les datasets
train_data = pd.concat([train_data_40, pd.read_csv(train_ebu_csv_path)], ignore_index=True)

# **2. Tokenisation des URLs**
def tokenize_url(url):
    """ Divise une URL en tokens basés sur '/', '.', et '-' """
    return re.split(r'[/\.-]', str(url).lower())

# Appliquer la tokenisation
train_tokens = train_data['url'].apply(tokenize_url)
test_tokens = test_data_1['url'].apply(tokenize_url)

# **3. Création d'un vocabulaire unique de tokens**
unique_tokens = list(set(token for tokens in train_tokens for token in tokens))

# Mapping des tokens en indices numériques
token_to_idx = {token: idx + 1 for idx, token in enumerate(unique_tokens)}

# Fonction pour encoder une URL sous forme d'indices de tokens
def encode_tokens(tokens, max_len=200):
    encoded = [token_to_idx.get(token, 0) for token in tokens]
    return np.pad(encoded[:max_len], (0, max(0, max_len - len(encoded))), 'constant')

# Encodage des données
X_train_tokens = np.array([encode_tokens(tokens) for tokens in train_tokens])
X_test_tokens = np.array([encode_tokens(tokens) for tokens in test_tokens])

# Extraction des labels
y_train = train_data['label'].values
y_test = test_data_1['label'].values

input_layer = tf.keras.Input(shape=(200,), dtype=tf.int32)
x = layers.Embedding(input_dim=len(unique_tokens)+1, output_dim=128)(input_layer)
x = layers.Conv1D(256, 4, activation='relu', padding='same')(x)
x = layers.MaxPooling1D(pool_size=4)(x)
x = layers.LSTM(256, return_sequences=False)(x)
x = layers.Dense(128, activation='sigmoid')(x)
x = layers.Dropout(0.1)(x)
output = layers.Dense(1, activation='sigmoid')(x)

model = tf.keras.Model(inputs=input_layer, outputs=output)

# **5. Compilation du modèle**
model.compile(
    loss=tf.keras.losses.BinaryCrossentropy(),
    optimizer=optimizers.Adam(learning_rate=1e-4),
    metrics=['accuracy']
)

# **6. Entraînement du modèle**
history = model.fit(
    X_train_tokens, y_train,
    epochs=5,
    batch_size=32,
    validation_split=0.2
)

# **7. Sauvegarde du modèle**
model.save('urlsModule/TensorFlow/cnn_lstm_tf_model_token.keras')
print('✅ Modèle CNN-LSTM sauvegardé !')

