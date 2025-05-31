import shap
import joblib
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras import layers, models, optimizers
import pandas as pd
import re

# **1. Charger le modèle Keras**
full_model_path = 'urlsModule/TensorFlow/cnn_lstm_tf_model_token.keras'
full_model = tf.keras.models.load_model(full_model_path)
print(f"✅ Modèle chargé depuis {full_model_path}")


# **2. Charger les données de test**
test_csv_path = 'enron_spamassassin_nazario_nigerian_best_compo/extracted_urls/urls_test_set.csv'
test_data = pd.read_csv(test_csv_path)

# **3. Tokenisation des URLs**
def tokenize_url(url):
    """ Divise une URL en tokens basés sur '/', '.', et '-' """
    return re.split(r'[/\.-]', str(url).lower())

# Appliquer la tokenisation sur les données de test
test_tokens = test_data['url'].apply(tokenize_url)

# **4. Création d'un vocabulaire unique de tokens**
unique_tokens = list(set(token for tokens in test_tokens for token in tokens))

# **5. Mapping des tokens en indices numériques**
token_to_idx = {token: idx + 1 for idx, token in enumerate(unique_tokens)}

# **6. Fonction pour encoder une URL sous forme d'indices de tokens**
def encode_tokens(tokens, max_len=200):
    encoded = [token_to_idx.get(token, 0) for token in tokens]  # Convertir les tokens en indices
    return np.pad(encoded[:max_len], (0, max(0, max_len - len(encoded))), 'constant')

# **7. Encodage des données de test**
X_test_tokens = np.array([encode_tokens(tokens) for tokens in test_tokens])

# **8. Sélectionner un sous-ensemble pour SHAP**
background_size = 1000  # Échantillons de fond pour SHAP
test_size = 100  # Nombre d'exemples de test à expliquer

background = X_test_tokens[:background_size]  # SHAP utilise ces exemples pour la moyenne
test_samples = X_test_tokens[:test_size]  # Exemples de test à expliquer
background = background[:50].astype(np.int32)
test_samples = test_samples[:10].astype(np.int32)

# **9. Vérifier la forme des entrées du modèle**
expected_input_shape = full_model.input_shape  # Shape attendue par le modèle
"""
if len(full_model.input_shape) == 3:  # (None, 20, 1) par ex.
    test_samples = test_samples.reshape((test_samples.shape[0], test_samples.shape[1], 1))
    background = background.reshape((background.shape[0], background.shape[1], 1))

"""

print(f"✅ Forme attendue par le modèle: {full_model.input_shape}")
print(f"🔹 Taille de `background` (pour SHAP) : {background.shape}")
print(f"🔹 Taille de `test_samples` (données à expliquer) : {test_samples.shape}")
sample_prediction = full_model.predict(test_samples)
print(f"🔹 Taille de la sortie du modèle : {sample_prediction.shape}")

embedding_layer = full_model.layers[0]

background_embed = embedding_layer(background).numpy()
test_embed = embedding_layer(test_samples).numpy()

# 4. Créer un sous-modèle (de l'embedding à la sortie)
input_embed = tf.keras.Input(shape=background_embed.shape[1:])
x = full_model.layers[1](input_embed)  # Conv1D
x = full_model.layers[2](x)            # MaxPooling1D
x = full_model.layers[3](x)            # LSTM
x = full_model.layers[4](x)            # Dense
x = full_model.layers[5](x)            # Dropout
output = full_model.layers[6](x)       # Dernière Dense
sub_full_model = tf.keras.Model(inputs=input_embed, outputs=output)

# 5. Maintenant que tout est en float -> SHAP fonctionne 🎉
explainer = shap.GradientExplainer(sub_full_model, background_embed)
shap_values = explainer.shap_values(test_embed)



# **11. Affichage des résultats SHAP**
shap.summary_plot(shap_values, test_samples, feature_names=unique_tokens)

# **12. Bar Plot des tokens les plus influents**
shap_values_mean = np.abs(shap_values).mean(axis=0)

plt.figure(figsize=(10, 6))
plt.barh([unique_tokens[i] for i in np.argsort(shap_values_mean)[-10:]], sorted(shap_values_mean)[-10:])
plt.xlabel("Impact SHAP moyen")
plt.ylabel("Mots-clés dans l'URL")
plt.title("Influence des mots-clés des URLs sur la classification")
plt.show()



"""
# **13. Analyse détaillée d'une URL spécifique avec Waterfall Plot**
example_index = 5  # Choisir une URL spécifique à expliquer
shap.waterfall_plot(shap.Explanation(values=shap_values[0][example_index], 
                                     base_values=explainer.expected_value[0], 
                                     feature_names=tokenize_url(test_data['url'].iloc[example_index])))
"""