
import pandas as pd
import random
import requests
import xgboost as xgb
import os
from pathlib import Path
import subprocess
import shap
import numpy as np
import matplotlib.pyplot as plt

#📊 Sur 100 URLs encodées, 0 sont détectées comme légitimes (0.00%)


XGB_MODEL_PATH = "urlsModule/nlp_features/xgboost_model_nlp_features.json"
model = xgb.XGBClassifier()
model.load_model(XGB_MODEL_PATH)

# === NLP Pipeline Runner ===
def run_nlp_pipeline(csv_path):
    print("🚀 Lancement de la pipeline NLP...")
    csv_path = Path(csv_path)
    if csv_path.suffix == '.csv':
        csv_path = csv_path.with_suffix('')
    csv_base = str(csv_path)
    base_name = os.path.splitext(os.path.basename(csv_path))[0]
    subprocess.run(f"py urlsModule/nlp_features/nlp_features_extraction/input/csv_to_json.py {csv_base}.csv", check=True, shell=True)
    subprocess.run(f"py urlsModule/nlp_features/nlp_features_extraction/src/train.py {base_name}_phishing.json phish {base_name}_legitimate.json legitimate", check=True, shell=True)
    subprocess.run(f"py urlsModule/nlp_features/nlp_features_extraction/output/features/txt_to_csv.py {csv_base}_features.csv", check=True, shell=True)
    subprocess.run(f"py numerique.py {csv_base}_features.csv", check=True, shell=True)
    print("✅ Pipeline terminée.")



import pandas as pd

# Charger le CSV
df = pd.read_csv('urls_attacks/selected_phishing_urls_nlp_train_set.csv')

# Renommer la colonne 'url' -> 'original_url'
df = df.rename(columns={'url': 'original_url'})


# Fonction pour encoder uniquement la partie après http:// ou https://
def partial_encode_url(url):
    if url.startswith('http://'):
        prefix = 'http://'
        rest = url[len('http://'):]
    elif url.startswith('https://'):
        prefix = 'https://'
        rest = url[len('https://'):]
    else:
        prefix = ''
        rest = url
    encoded_rest = ''.join(f'%{ord(c):02X}' for c in rest)
    return prefix + encoded_rest

# Créer la nouvelle colonne 'url' avec l'url encodée
df['url'] = df['original_url'].apply(partial_encode_url)

# Sauvegarder dans un nouveau CSV
df.to_csv('urls_attacks/experience_2_hexadecimal/hexadecimal_urls.csv', index=False)

print("✅ Nouveau CSV généré avec succès !")


run_nlp_pipeline("urls_attacks/experience_2_hexadecimal/hexadecimal_urls.csv")

features = pd.read_csv("urls_attacks/experience_2_hexadecimal/hexadecimal_urls_features_filtered.csv").drop(columns=["url", "id", "label"], errors="ignore")

# === Prédictions
proba_hexa = model.predict_proba(features)[:, 1]

# Charger de nouveau le CSV avec les URLs encodées
df_results = pd.read_csv('urls_attacks/experience_2_hexadecimal/hexadecimal_urls.csv')

# Ajouter la colonne avec les probabilités
df_results['hexa_proba'] = proba_hexa

# Sauvegarder les résultats si besoin
df_results.to_csv('urls_attacks/experience_2_hexadecimal/hexadecimal_urls_with_proba.csv', index=False)
print("✅ Fichier sauvegardé avec les probabilités !")

# Calculer le % d'URLs détectées comme légitimes
num_legit = (df_results['hexa_proba'] < 0.5).sum()
total = len(df_results)
percent_legit = (num_legit / total) * 100

print(f"\n📊 Sur {total} URLs encodées, {num_legit} sont détectées comme légitimes ({percent_legit:.2f}%)")

X_background = pd.read_csv("enron_spamassassin_nazario_nigerian_best_compo/extracted_urls/combined_train_set_features_filtered.csv").drop(columns=['label', 'id', 'url'], errors='ignore')

explainer = shap.Explainer(model, X_background)

# Calculer les valeurs SHAP
shap_values = explainer(features)

# Afficher le summary plot
shap.summary_plot(shap_values, features)
plt.show()

# Afficher le bar plot des SHAP moyens
shap.plots.bar(shap_values.mean(0), max_display=30)
plt.show()
