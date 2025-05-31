import pandas as pd
import random
import xgboost as xgb
import os
from pathlib import Path
import subprocess
import shap
import numpy as np
import matplotlib.pyplot as plt

# === CONFIG ===
XGB_MODEL_PATH = "urlsModule/nlp_features/xgboost_model_nlp_features.json"
OUTPUT_CSV = "redirect_attack_results.csv"

# === Charger le modèle ===
model = xgb.XGBClassifier()
model.load_model(XGB_MODEL_PATH)
X_background = pd.read_csv("enron_spamassassin_nazario_nigerian_best_compo/extracted_urls/combined_train_set_features_filtered.csv").drop(columns=['label', 'id', 'url'], errors='ignore')

explainer = shap.Explainer(model, X_background)

# === Charger les données ===
df_legit = pd.read_csv("url_legitimate_redirect.csv")
df_phish = pd.read_csv("selected_phishing_urls_nlp_train_set.csv")

# === Pas de raccourcissement : on utilise directement les URLs phishing
df_phish["short_url"] = df_phish["url"]
df_phish["shortener"] = "original"
df_phish_success = df_phish.copy()
num_total = len(df_phish_success)
print(f"✅ Toutes les {num_total} URLs phishing ont été conservées pour l’analyse (pas de raccourcissement).")

# === Liste de services de redirection légitimes
services = [
    "https://www.google.com/url?q=",
    "https://slack-redir.net/link?url=",
    "https://www.facebook.com/l.php?u=",
    "https://www.youtube.com/redirect?q=",
    "https://www.linkedin.com/redir/redirect?url="
]

# === Générer les URLs redirect
redirect_urls = [
    random.choice(services) + phishing_url
    for phishing_url in df_phish_success["short_url"]
]
df_phish_success["redirect_url"] = redirect_urls

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
    subprocess.run(f"py urlsModule/nlp_features/nlp_features_extraction/output/features/txt_to_csv.py {base_name}_features.csv", check=True, shell=True)
    subprocess.run(f"py numerique.py {csv_base}_features.csv", check=True, shell=True)
    print("✅ Pipeline terminée.")

# === Pipeline NLP pour original & redirect URLs
df_original = df_phish_success[["short_url"]].copy()
df_original.columns = ["url"]
df_original["label"] = 1
df_original["index"] = range(len(df_original))
df_original.to_csv("temp_phishing_urls.csv", index=False)

df_modified = df_phish_success[["redirect_url"]].copy()
df_modified.columns = ["url"]
df_modified["label"] = 1
df_modified["index"] = range(len(df_modified))
df_modified.to_csv("temp_redirect_urls.csv", index=False)

run_nlp_pipeline("temp_phishing_urls.csv")
run_nlp_pipeline("temp_redirect_urls.csv")

# === Charger les features
features_orig = pd.read_csv("temp_phishing_urls_features_filtered.csv").drop(columns=["url", "id", "label"], errors="ignore")
features_mod  = pd.read_csv("temp_redirect_urls_features_filtered.csv").drop(columns=["url", "id", "label"], errors="ignore")

# === Prédictions
proba_orig = model.predict_proba(features_orig)[:, 1]
proba_mod = model.predict_proba(features_mod)[:, 1]
df_phish_success["original_proba"] = proba_orig
df_phish_success["redirect_proba"] = proba_mod

# === Résultat final
df_phish_success[["url", "short_url", "shortener", "redirect_url", "original_proba", "redirect_proba"]].to_csv(OUTPUT_CSV, index=False)
print(f"✅ Résultats sauvegardés dans {OUTPUT_CSV}")

# === Statistiques
num_legit = sum(df_phish_success["redirect_proba"] < 0.5)
total = len(df_phish_success)
percent_legit = (num_legit / total) * 100
print(f"\n📉 {num_legit}/{total} URLs modifiées ({percent_legit:.2f}%) sont maintenant considérées comme légitimes par le modèle.")

print("\n🔍 Legitimate base URLs that successfully 'masked' the phishing attempt:")
successful = df_phish_success[df_phish_success["redirect_proba"] < 0.5]
if not successful.empty:
    for i, row in successful.iterrows():
        print(f"→ {row['redirect_url']} — new proba: {row['redirect_proba']:.4f}")
else:
    print("❌ Aucune URL légitime n'a permis de faire passer l'URL phishing comme légitime.")

# === Drop moyen (pour les vrais phishing restants)
still_phishing = df_phish_success[df_phish_success["redirect_proba"] >= 0.5]
mean_drop_phishing = (still_phishing["original_proba"] - still_phishing["redirect_proba"]).mean()
print(f"\n📌 Moyenne du drop (URLs toujours classées comme phishing) : {mean_drop_phishing:.4f} ({len(still_phishing)} cas)")

# === Stat globale
total_all = len(df_phish)
total_detected_phishing = len(df_phish_success[df_phish_success["redirect_proba"] >= 0.5])
percent_detected_as_phishing = (total_detected_phishing / total_all) * 100
print(f"\n🔐 Overall phishing detection: {total_detected_phishing}/{total_all} "
      f"({percent_detected_as_phishing:.2f}%) are still classified as phishing.")

# === SHAP explicabilité
X = features_mod

idx_max = df_phish_success["redirect_proba"].idxmax()
idx_min = df_phish_success["redirect_proba"].idxmin()

print("\n🔍 URL avec la PLUS HAUTE probabilité de phishing (redirect):")
print(f"→ {df_phish_success.loc[idx_max, 'redirect_url']} — proba: {df_phish_success.loc[idx_max, 'redirect_proba']:.4f}")
shap_values_max = explainer(X.iloc[[idx_max]])
df_shap_max = pd.DataFrame({
    "feature": X.columns,
    "value": X.iloc[idx_max].values,
    "shap_value": shap_values_max.values[0]
}).sort_values(by="shap_value", key=abs, ascending=False)
print(df_shap_max.head(5))

print("\n🔍 URL avec la PLUS BASSE probabilité de phishing (redirect):")
print(f"→ {df_phish_success.loc[idx_min, 'redirect_url']} — proba: {df_phish_success.loc[idx_min, 'redirect_proba']:.4f}")
shap_values_min = explainer(X.iloc[[idx_min]])
df_shap_min = pd.DataFrame({
    "feature": X.columns,
    "value": X.iloc[idx_min].values,
    "shap_value": shap_values_min.values[0]
}).sort_values(by="shap_value", key=abs, ascending=False)
print(df_shap_min.head(5))


# === SHAP + valeur moyenne pour les URLs détectées comme phishing

# Calcul des valeurs SHAP
shap_values = explainer(X)

shap.summary_plot(shap_values, X)
plt.show()

shap.plots.bar(shap_values.mean(0), max_display=30)
plt.show()