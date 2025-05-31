import pandas as pd
import xgboost as xgb
import json
from structureModule.extraction_features import load_email_from_string, extract_structure_features
from pathlib import Path
import tensorflow as tf
from keras import layers, models, optimizers
import numpy as np
from urlsModule.TensorFlow.cnnLstmTensorFlow import encode_url
import shap
import matplotlib.pyplot as plt


# --------------------- CONFIG -----------------------
INPUT_CSV = "emails_eml/Perso_Liza/pub_english_liza/pub_english_liza.csv"
MODEL_PATH = "structureModule/xgboost_email_model.json"
FEATURES_PATH = "structureModule/feature_names.json"
OUTPUT_CSV = "predictions_emails.csv"

CNN_MODEL_PATH = "urlsModule/TensorFlow/cnn_lstm_tf_model_ebu.keras"
# ----------------------------------------------------

# 📥 Chargement du CSV
df_input = pd.read_csv(INPUT_CSV)
df_input = df_input.head(150)  # 🔥 Garde uniquement les 10 premiers emails

if 'email' not in df_input.columns:
    raise ValueError("❌ La colonne 'email' est manquante dans le fichier CSV.")

# 📦 Chargement du modèle
model = xgb.XGBClassifier()
model.load_model(MODEL_PATH)

# 📑 Colonnes attendues
with open(FEATURES_PATH, "r") as f:
    expected_columns = json.load(f)

results = []

for idx, raw_email in enumerate(df_input['email']):
    try:
        msg = load_email_from_string(raw_email)
        features,clickable_urls, text_content = extract_structure_features(msg, index=idx)

        df = pd.DataFrame([features])

        # 🔧 Fixer les colonnes
        for col in expected_columns:
            if col not in df.columns:
                df[col] = 0  # ajouter colonne manquante
        df = df[expected_columns]  # réordonner
        df = df.fillna(0)

        structure_proba = model.predict_proba(df)[0][1]

        """

        X_background = pd.read_csv("enron_spamassassin_nazario_nigerian_best_compo/extracted_features/features_train_set_40.csv").drop(columns=['label', 'index', 'email_id'], errors='ignore')

        explainer = shap.Explainer(model, X_background)
        shap_values = explainer(df)

        shap_df = pd.DataFrame({
            "feature": df.columns,
            "value": df.iloc[0].values,
            "shap_value": shap_values.values[0]
        })

        # Trier par importance absolue
        shap_df["abs_shap"] = np.abs(shap_df["shap_value"])
        shap_df = shap_df.sort_values(by="abs_shap", ascending=False)
        

        # 📦 Afficher les 10 features les plus influentes
        print("\n🔍 Top 5 structural features impacting prediction:")
        print(shap_df[["feature", "value", "shap_value"]].head(5))
        """
        # TEXT AUTOMATISATION
        # 🔽 Sauvegarde du contenu texte dans un fichier
        text_filename = f"text_outputs/text_output_{idx}.txt"
        with open(text_filename, "w", encoding="utf-8") as txt_file:
            txt_file.write(text_content)
        print(f"📝 text_content saved to {text_filename}")

        #Url automatisation : clickable_urls
        cnn_model = tf.keras.models.load_model(CNN_MODEL_PATH)
                # 🔍 Analyse des URLs avec le modèle CNN
        url_probas = []
        for url in clickable_urls:
            encoded = encode_url(url)
            prediction = cnn_model.predict(np.array([encoded]), verbose=0)[0][0]
            url_probas.append(prediction)

        # Prendre la proba max des URLs ou -1 s’il n’y a pas d’URL
        max_url_proba = max(url_probas) if url_probas else -1


        results.append({
            "id": idx,
            "filename": "pub_english_liza",
            "structure_proba": structure_proba,
            "text_proba": 0,
            "url_proba": max_url_proba
        })

        print(f"✅ {idx} structure_proba: {structure_proba:.4f} |text_proba: {0:.4f} | url_proba = {max_url_proba:.4f}")

    except Exception as e:
        print(f"⚠️ Error processing {idx}: {e}")

# Sauvegarde des résultats
df_results = pd.DataFrame(results)
df_results.to_csv(OUTPUT_CSV, index=False)
print(f"\n📁 Résultats sauvegardés dans: {OUTPUT_CSV}")



