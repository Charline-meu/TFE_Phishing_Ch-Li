import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report

# **1. Charger le modèle sauvegardé**
model_path = "urlsModule/TensorFlow/cnn_lstm_tf_model_test.keras"
model = tf.keras.models.load_model(model_path)
print("✅ Modèle chargé avec succès !")

# **2. Définir le chemin des fichiers de test**
test_csv_path_1 = "enron_spamassassin_nazario_nigerian_best_compo/extracted_urls/urls_test_set.csv"
test_csv_path_2 = "enron_spamassassin_nazario_nigerian_best_compo/extracted_urls/urls_train_set_20.csv"

# **3. Fonction d'encodage des URLs**
def encode_url(url, max_len=200):
    """ Convertit l'URL en séquence de caractères ASCII avec padding à max_len """
    if url == "-1":  # Si l'email n'a pas d'URL
        return None  # Exclure cet email des prédictions

    encoded = [ord(char) for char in str(url) if ord(char) < 128]  # Convertir en ASCII
    return np.pad(encoded[:max_len], (0, max(0, max_len - len(encoded))), 'constant')

# **4. Fonction de test du modèle sur un dataset**
def test_model_on_dataset(csv_path, test_name):
    print(f"\n📥 Chargement du dataset : {csv_path}")

    # Charger les données
    df = pd.read_csv(csv_path)

    # Vérifier si les colonnes nécessaires existent
    required_columns = {'url', 'label', 'index'}
    if not required_columns.issubset(df.columns):
        raise ValueError(f"Le fichier {csv_path} doit contenir les colonnes : {required_columns}")

    # **5. Prétraitement des URLs**
    df['url'] = df['url'].astype(str).str.lower()  # Conversion en minuscules
    df['encoded_url'] = df['url'].apply(encode_url)  # Encodage des URLs

    # Séparer les emails avec et sans URL
    df_with_urls = df[df['encoded_url'].notna()].copy()  # Emails avec URL
    df_without_urls = df[df['encoded_url'].isna()].copy()  # Emails sans URL

    # **6. Préparer les données pour la prédiction**
    if not df_with_urls.empty:
        X_test = np.array(df_with_urls['encoded_url'].tolist())  # Exclure les emails sans URL
        y_test = df_with_urls['label'].values  # Labels
        metadata = df_with_urls[['index']].copy()
        
        # **7. Prédiction des probabilités de phishing**
        y_pred_proba = model.predict(X_test).flatten()

        # **8. Créer un DataFrame de sortie**
        result_df = metadata.copy()  # Récupérer `index`
        result_df["phishing_probability"] = y_pred_proba
        result_df["label"] = y_test  # Label réel
    else:
        result_df = pd.DataFrame(columns=['index', 'phishing_probability', 'label'])

    # **9. Ajouter les emails sans URL avec `phishing_probability = -1`**
    if not df_without_urls.empty:
        no_url_results = df_without_urls[['index', 'label']].copy()
        no_url_results['phishing_probability'] = -1  # Indiquer qu'il n'y a pas d'URL
        result_df = pd.concat([result_df, no_url_results], ignore_index=True)

    # **10. Sélectionner la probabilité MAX par email**
    final_df = result_df.groupby(['index'], as_index=False).agg({
        'phishing_probability': 'max',
        'label': 'first'
    })

    # **11. Sauvegarde du CSV**
    output_csv_path = f"enron_spamassassin_nazario_nigerian_best_compo/extracted_urls/urls_proba_ebu_{test_name}.csv"
    final_df.to_csv(output_csv_path, index=False)
    print(f"✅ Fichier CSV généré : {output_csv_path}")

    # **12. Rapport de Classification (exclure les emails sans URL)**
    valid_predictions = final_df[final_df['phishing_probability'] != -1]
    
    if not valid_predictions.empty:
        final_y_pred_classes = (valid_predictions['phishing_probability'] > 0.5).astype(int)
        print(f"\n📌 **Rapport de Classification pour {test_name}** 📌\n")
        print(classification_report(valid_predictions['label'], final_y_pred_classes, target_names=['Légitime', 'Phishing']))
    else:
        print(f"\n⚠️ Aucun email valide avec URL pour la classification dans {test_name}\n")

# **13. Tester le modèle sur les 2 datasets de test**
test_model_on_dataset(test_csv_path_1, "test_set")
test_model_on_dataset(test_csv_path_2, "train_set_20")

print("✅ Tests terminés avec succès !")
