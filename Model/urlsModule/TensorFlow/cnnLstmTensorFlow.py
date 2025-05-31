import tensorflow as tf
from keras import layers, models, optimizers
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# **2. Fonction d'encodage des URLs**
def encode_url(url, max_len=200):
    """ Convertit l'URL en séquence de caractères ASCII avec padding à max_len """
    encoded = [ord(char) for char in str(url) if ord(char) < 128]  # Convertir en ASCII
    return np.pad(encoded[:max_len], (0, max(0, max_len - len(encoded))), 'constant')

# **3. Prétraitement des données**
def preprocess_data(df):
    df['url'] = df['url'].astype(str).str.lower()  # Conversion en minuscules
    X = np.array([encode_url(url) for url in df['url']])  # Encodage des URLs
    y = df['label'].values  # Labels (0 = légitime, 1 = phishing)
    
    # On conserve `index` et `email_id` mais on les exclut des prédictions
    metadata = df[['index']].copy()
    
    return X, y, metadata

# **7. Fonction pour évaluer et sauvegarder les résultats**
def evaluate_and_save_results(X_test, y_test, metadata, test_name, original_df):
    # ➕ Reconstituer le DataFrame complet (avec URL)
    df_full = original_df.copy()
    
    # ⚠️ Séparer les URLs valides et -1
    df_valid = df_full[df_full['url'] != "-1"].copy()
    df_invalid = df_full[df_full['url'] == "-1"].copy()

    # ✅ Prédiction sur les valides
    X_valid, y_valid, meta_valid = preprocess_data(df_valid)
    y_pred_proba = model.predict(X_valid).flatten()

    # Résultats valides
    df_valid_output = meta_valid.copy()
    df_valid_output["phishing_probability"] = y_pred_proba
    df_valid_output["label"] = y_valid
    df_valid_output["url"] = df_valid["url"]

    # 🚫 Résultats pour -1 (proba = -1)
    df_invalid_output = df_invalid[["index", "label"]].copy()
    df_invalid_output["phishing_probability"] = -1
    df_invalid_output["url"] = -1

    # 🧪 Fusionner
    full_output_df = pd.concat([df_valid_output, df_invalid_output], ignore_index=True)

    # Prendre l'URL avec la plus haute proba pour chaque index
    final_df = full_output_df.sort_values(by=["index", "phishing_probability"], ascending=[True, False]) \
                            .groupby("index", as_index=False).first()

    # Réorganiser les colonnes (optionnel)
    final_df = final_df[["index", "url", "label", "phishing_probability"]]

    # 📁 Sauvegarde
    output_csv_path = f"enron_spamassassin_nazario_nigerian/url_proba_{test_name}.csv"
    final_df.to_csv(output_csv_path, index=False)
    print(f"✅ Fichier CSV généré : {output_csv_path}")

    # 📊 Évaluation uniquement sur les valides
    y_pred_binary = (y_pred_proba > 0.5).astype(int)
    print(f"\n📌 **Rapport CNN-LSTM (valid URLs only) — {test_name}** 📌\n")
    print(classification_report(y_valid, y_pred_binary, target_names=['Légitime', 'Phishing']))

if __name__ == "__main__":
    # **1. Charger les datasets**
    train_data_40_path = 'enron_spamassassin_nazario_nigerian_best_compo/extracted_urls/urls_train_set_40.csv'  # Fichier d'entraînement
    train_ebu_csv_path = 'datasetURLs/ebubekirbbr.csv'
    test_csv_path_1 = 'enron_spamassassin_nazario_nigerian_best_compo/extracted_urls/urls_test_set.csv'  # Premier fichier de test
    test_csv_path_2 = 'enron_spamassassin_nazario_nigerian_best_compo/extracted_urls/urls_train_set_20.csv'  # Deuxième fichier de test

    # Charger le dataset
    train_data_40 = pd.read_csv(train_data_40_path)

    train_data_40.drop(columns=['index'])
    train_data_1 = train_data_40
    train_data_2 = pd.read_csv(train_ebu_csv_path)
    train_data = pd.concat([train_data_1, train_data_2], ignore_index=True)

    # Vérifier les valeurs uniques dans la colonne `label`
    print(train_data_40['label'].value_counts())

    # Séparer les phishing et les légitimes
    phishing_df = train_data[train_data['label'] == 1]
    legit_df = train_data[train_data['label'] == 0]

    # Trouver le plus petit nombre d'échantillons entre les deux classes
    min_count = min(len(phishing_df), len(legit_df))

    # Échantillonner aléatoirement un nombre égal d'URLs pour chaque classe
    phishing_sample = phishing_df.sample(n=min_count, random_state=42)
    legit_sample = legit_df.sample(n=min_count, random_state=42)

    # Fusionner les deux échantillons équilibrés
    balanced_train_data = pd.concat([phishing_sample, legit_sample], ignore_index=True)
    balanced_train_data.to_csv("urlsModule/Tensorflow/balanced_train_data.csv", index=False)

    # Vérifier l'équilibre
    print("balanced_set")
    print(balanced_train_data['label'].value_counts())

    # Sauvegarder le dataset équilibré (optionnel)
    balanced_train_data.to_csv("enron_spamassassin_nazario_nigerian_best_compo/extracted_urls/train_set_ebu_balanced.csv", index=False)

    print("📥 Chargement des datasets...")
    test_data_1 = pd.read_csv(test_csv_path_1)
    test_data_2 = pd.read_csv(test_csv_path_2)

    # Encodage du training set
    X_train, y_train, _ = preprocess_data(balanced_train_data)

    # Encodage des test sets
    X_test_1, y_test_1, meta_test_1 = preprocess_data(test_data_1)
    X_test_2, y_test_2, meta_test_2 = preprocess_data(test_data_2)

    # **4. Définition du modèle CNN-LSTM**
    model = models.Sequential([
        layers.Embedding(input_dim=128, output_dim=128, input_length=200),  # Embedding Layer
        layers.Conv1D(filters=256, kernel_size=4, activation='relu', padding='same'),
        layers.MaxPooling1D(pool_size=4),
        layers.LSTM(units=512, return_sequences=False),
        layers.Dropout(0.1),
        layers.Dense(units=256, activation='relu'),
        layers.Dense(units=1, activation='sigmoid')  # Output en probabilité
    ])

    # **5. Compilation du modèle**
    model.compile(
        loss=tf.keras.losses.BinaryCrossentropy(),
        optimizer=optimizers.Adam(learning_rate=1e-4),
        metrics=['accuracy',tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall'),
            tf.keras.metrics.AUC(name='AUC', curve='PR')]
    )

    # **6. Entraînement du modèle**
    history = model.fit(
        X_train, y_train,
        epochs=4,
        batch_size=32
    )

    # **8. Évaluer les deux test sets**
    evaluate_and_save_results(X_test_1, y_test_1, meta_test_1, "test_set", test_data_1)
    evaluate_and_save_results(X_test_2, y_test_2, meta_test_2, "train_set_20", test_data_2)


    model.save('urlsModule/TensorFlow/cnn_lstm_tf_model_ebu.keras')
    np.save('urlsModule/TensorFlow/X_test_1.npy', X_test_1)
    np.save('urlsModule/TensorFlow/y_test_1.npy', y_test_1)
    np.save('urlsModule/TensorFlow/X_test_1.npy', X_train)
    np.save('urlsModule/TensorFlow/y_test_1.npy', y_train)
    print('✅ Modèle CNN-LSTM TensorFlow/Keras sauvegardé avec succès !')
