import pandas as pd
import xgboost as xgb
import json
from structureModule.extraction_features import load_email_from_string, extract_structure_features
from pathlib import Path
import numpy as np
import shap
import matplotlib.pyplot as plt
import subprocess
import sys
import os


# --------------------- CONFIG -----------------------
INPUT_CSV = "email_test_redirect.csv"
MODEL_PATH = "structureModule/xgboost_model_esnnpperso.json"
FEATURES_PATH = "structureModule/feature_names.json"
OUTPUT_CSV = "predictions_emails.csv"

URLS_XGBOOST_MODEL ="urlsModule/nlp_features/xgboost_model_nlp_features.json"
# ----------------------------------------------------

# 📥 Chargement du CSV
# Chargement des colonnes attendues
with open(FEATURES_PATH, "r") as f:
    expected_columns = json.load(f)

# 📦 Chargement du modèle
model = xgb.XGBClassifier()
model.load_model(MODEL_PATH)

urls_model = xgb.XGBClassifier()
urls_model.load_model(URLS_XGBOOST_MODEL)

input_path = Path(sys.argv[1])
eml_files = []

if input_path.is_dir():
    eml_files = list(input_path.glob("*.eml"))
elif input_path.is_file() and input_path.suffix == ".eml":
    eml_files = [input_path]
else:
    print("❌ Please provide a .eml file or folder containing .eml files.")
    sys.exit(1)

def run_nlp_pipeline(csv_path):
    print("🚀 Lancement de la pipeline NLP...")
    csv_path = Path(csv_path)
    if csv_path.suffix == '.csv':
        csv_path = csv_path.with_suffix('')
    csv_base = str(csv_path)
    base_name = os.path.splitext(os.path.basename(csv_path))[0]
    subprocess.run(f"py urlsModule/nlp_features/nlp_features_extraction/input/csv_to_json.py {csv_base}.csv", check=True, shell=True)
    subprocess.run(f"py urlsModule/nlp_features/nlp_features_extraction/src/train.py {base_name}_phishing.json phish {base_name}_legitimate.json legitimate", check=True, shell=True)
    subprocess.run(f"py urlsModule/nlp_features/nlp_features_extraction/output/features/txt_to_csv.py {base_name}_features.csv", check=True, shell=True)
    subprocess.run(f"py numerique.py {csv_base}_features.csv", check=True, shell=True)
    print("✅ Pipeline terminée.")

# 📑 Colonnes attendues
with open(FEATURES_PATH, "r") as f:
    expected_columns = json.load(f)

results = []

X_background = pd.read_csv("urlsModule/nlp_features/nlp_features_ebu_numerique.csv").drop(columns=['class', 'id', 'url'], errors='ignore')

explainer = shap.Explainer(urls_model, X_background)

for idx, eml_file in enumerate(eml_files):
    try:
        with open(eml_file, "r", encoding="utf-8", errors="ignore") as f:
            raw_email = f.read()
        msg = load_email_from_string(raw_email)
        features,clickable_urls, text_content = extract_structure_features(msg, index=idx)

        urls_max = ""
        df = pd.DataFrame([features])  # correct
        print(clickable_urls)


        # 🔧 Fixer les colonnes
        for col in expected_columns:
            if col not in df.columns:
                df[col] = 0  # ajouter colonne manquante
        df = df[expected_columns]  # réordonner
        df = df.fillna(0)

        structure_proba = model.predict_proba(df)[0][1]

        # TEXT AUTOMATISATION
        # 🔽 Sauvegarde du contenu texte dans un fichier
        text_filename = f"text_outputs/text_output_{idx}.txt"
        with open(text_filename, "w", encoding="utf-8") as txt_file:
            txt_file.write(text_content)
        print(f"📝 text_content saved to {text_filename}")

        #Url automatisation : clickable_urls

        url_probas = []
        df_urls = pd.DataFrame(clickable_urls, columns=["url"])
        df_urls.to_csv("urls_clickable.csv", index=False)
        run_nlp_pipeline("urls_clickable.csv")
        features_df = pd.read_csv("urls_clickable_features_filtered.csv")
        features_df = features_df.drop(columns=["url","id","label"])
        # 🔮 Prédire la probabilité pour chaque URL (classe phishing = 1)
        url_probas = urls_model.predict_proba(features_df)[:, 1]

                # 🌟 Prendre la probabilité maximale
        max_url_proba = url_probas.max() if len(url_probas) > 0 else -1

        if len(url_probas) > 0:
            max_index = np.argmax(url_probas)
            urls_max = features_df.iloc[[max_index]]
            max_url = df_urls.iloc[max_index]["url"]
        else:
            urls_max = features_df.iloc[[0]]
            max_url = "N/A"

        shap_values = explainer(urls_max)

        shap_df = pd.DataFrame({
            "feature": urls_max.columns,
            "value": urls_max.iloc[0].values,
            "shap_value": shap_values.values[0]
        })

        shap_df["abs_shap"] = np.abs(shap_df["shap_value"])
        shap_df = shap_df.sort_values(by="abs_shap", ascending=False)

        print("\n🔍 Top 5 structural features impacting prediction:")
        print(shap_df[["feature", "value", "shap_value"]].head(5))

        results.append({
            "id": idx,
            "filename": "pub_english_liza",
            "structure_proba": structure_proba,
            "text_proba": 0,
            "url_proba": max_url_proba
        })

        print(f"✅ {idx} structure_proba: {structure_proba:.4f} | text_proba: {0:.4f} | url_proba = {max_url_proba:.4f}")
        print(f"🔗 URL with highest phishing probability: {max_url}")

    except Exception as e:
        print(f"⚠️ Error processing {idx}: {e}")

# Sauvegarde des résultats
df_results = pd.DataFrame(results)
df_results.to_csv(OUTPUT_CSV, index=False)
print(f"\n📁 Résultats sauvegardés dans: {OUTPUT_CSV}")



