import pandas as pd
import random
import xgboost as xgb
import os
from pathlib import Path
import subprocess
import shap
import numpy as np
import string
import matplotlib.pyplot as plt
import requests

# === CONFIGURATION ===
XGB_MODEL_PATH = "urlsModule/nlp_features/xgboost_model_nlp_features.json"
OUTPUT_CSV = "urls_attacks/experience_4_redirect/redirect_attack_results.csv"

# === Charger le modèle ===
model = xgb.XGBClassifier()
model.load_model(XGB_MODEL_PATH)

X_background = pd.read_csv("enron_spamassassin_nazario_nigerian_best_compo/extracted_urls/combined_train_set_features_filtered.csv").drop(columns=['label', 'id', 'url'], errors='ignore')

explainer = shap.Explainer(model, X_background)

# === Charger les données ===
df_legit = pd.read_csv("url_legitimate_redirect.csv")
df_phish = pd.read_csv("selected_phishing_urls_nlp_train_set.csv")
"""
def shorten_url(url, prefix="http://bit.ly/"):
    code = ''.join(random.choices(string.ascii_letters + string.digits, k=6))
    digit_count = sum(1 for c in code if c.isdigit())
    return prefix + code, "bitly"
""" 

def ensure_url_format(url):
    if not url.startswith("http://") and not url.startswith("https://"):
        return "http://" + url
    return url

def simulate_tinyurl(url, prefix="https://tinyurl.com/"):
    # Simule un code typique TinyURL (7 caractères alphanumériques)
    code = ''.join(random.choices(string.ascii_letters + string.digits, k=7))
    return prefix + code

# 🔹 Raccourcissement via API TinyURL directement (plus fiable que pyshorteners)
def shorten_url(url, timeout=10):
    try:
        url = ensure_url_format(url)
        api_url = f"https://tinyurl.com/api-create.php?url={url}"
        response = requests.get(api_url, timeout=timeout)
        if response.status_code == 200:
            return response.text.strip(), "tinyurl"
        else:
            print(f"❌ TinyURL API error {response.status_code} for {url} — fallback simulated.")
            return simulate_tinyurl(url), "tinyurl_simulated"
    except Exception as e:
        print(f"❌ Exception TinyURL: {url} — {e} — fallback simulated.")
        return simulate_tinyurl(url), "tinyurl_simulated"

    
    
def shorten_url_bitly(url, prefix="http://bit.ly/"):
    while True:
        code = ''.join(random.choices(string.ascii_letters + string.digits, k=6))
        digit_count = sum(1 for c in code if c.isdigit())
        if digit_count >= 3:
            return prefix + code, "bitly"
""" 
BITLY_TOKEN = "0483d308d683994d4317678881731e22fa5a0b81"

def shorten_url_bitly(long_url, group_guid=None, timeout=10):
    api_url = "https://api-ssl.bitly.com/v4/shorten"
    headers = {
        "Authorization": f"Bearer {BITLY_TOKEN}",
        "Content-Type": "application/json"
    }
    data = {
        "long_url": long_url
    }
    if group_guid:
        data["group_guid"] = group_guid  # si besoin
    try:
        response = requests.post(api_url, json=data, headers=headers, timeout=timeout)
        if response.status_code == 200:
            short_url = response.json()["link"]
            return short_url, "bitly"
        else:
            print(f"❌ Bitly API error {response.status_code}: {response.text}")
            return long_url, "none"
    except Exception as e:
        print(f"❌ Exception Bitly API: {long_url} — {e}")
        return long_url, "none"

"""
short_urls, services = [], []
for url in df_phish["url"]:
    short, used = shorten_url(url)
    short_urls.append(short)
    services.append(used)
df_phish["short_url"] = short_urls
df_phish["shortener"] = services

# 🔎 Ne garder que celles raccourcies avec succès
df_phish_success = df_phish
print(f"✅ {len(df_phish_success)} URLs phishing ont été raccourcies avec succès via TinyURL.")


# === Étape 2 : encapsuler avec des redirections de services légitimes
redirect_services = [
    #"https://www.google.com/url?q=",
    #"https://www.facebook.com/l.php?u=",
    #"https://slack-redir.net/link?url=",
    #"https://www.youtube.com/redirect?q=",
    #"https://www.linkedin.com/redir/redirect?url="

    #"https://secure.log.com/url?q=",
    #"https://check.amz.com/url?q=",
    #"https://www.ucl.edu/url?q=",
    #"https://auth.pro.net/url?q="

    "https://secure.log.com/redirect?q=",
    "https://check.amz.com/redirect?q=",
    "https://www.ucl.edu/redirect?q=",
    "https://auth.pro.net/redirect?q="

]

df_phish_success["redirect_url"] = [
    random.choice(redirect_services) + short_url
    for short_url in df_phish_success["short_url"]
]

# === NLP Pipeline Runner
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

# === Analyse de performance
num_legit = sum(df_phish_success["redirect_proba"] < 0.5)
total = len(df_phish_success)
percent_legit = (num_legit / total) * 100
print(f"\n📉 {num_legit}/{total} URLs modifiées ({percent_legit:.2f}%) sont considérées comme légitimes.")

print("\n🔍 Legitimate base URLs that masked the phishing attempt:")
successful = df_phish_success[df_phish_success["redirect_proba"] < 0.5]
if not successful.empty:
    for i, row in successful.iterrows():
        print(f"→ {row['redirect_url']} — new proba: {row['redirect_proba']:.4f}")
else:
    print("❌ Aucune URL n'a contourné le modèle.")

# === Moyenne du drop
still_phishing = df_phish_success[df_phish_success["redirect_proba"] >= 0.5]
mean_drop_phishing = (still_phishing["original_proba"] - still_phishing["redirect_proba"]).mean()
print(f"\n📌 Moyenne du drop (toujours classées comme phishing) : {mean_drop_phishing:.4f} ({len(still_phishing)} cas)")

# === Stat globale
total_all = len(df_phish)
total_detected_phishing = len(df_phish_success[df_phish_success["redirect_proba"] >= 0.5])
percent_detected_as_phishing = (total_detected_phishing / total_all) * 100
print(f"\n🔐 Global phishing detection: {total_detected_phishing}/{total_all} "
      f"({percent_detected_as_phishing:.2f}%) classées comme phishing.")

# === SHAP analysis
X = features_mod

idx_max = df_phish_success["redirect_proba"].idxmax()
idx_min = df_phish_success["redirect_proba"].idxmin()

print("\n🔍 URL avec la PLUS HAUTE proba phishing (redirect):")
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

# Calcul des valeurs SHAP
shap_values = explainer(X)

shap.summary_plot(shap_values, X)
plt.show()

shap.plots.bar(shap_values.mean(0), max_display=30)
plt.show()