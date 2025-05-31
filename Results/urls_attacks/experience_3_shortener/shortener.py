import pandas as pd
import numpy as np
import requests
import time
import subprocess
from pathlib import Path
import xgboost as xgb
import pyshorteners
import os
import shap
import matplotlib.pyplot as plt
from urllib.parse import urlparse

# ---------------- CONFIG ----------------
XGB_MODEL_PATH = "urlsModule/nlp_features/xgboost_model_nlp_features.json"
TEMP_URLS_FILE = "urls_attacks/experience_3_shortener/temp_urls_shortened_input.csv"
TEMP_FEATURES_FILE = "urls_attacks/experience_3_shortener/temp_urls_shortened_input_features_filtered.csv"
# ----------------------------------------

# 🔹 Charger le modèle XGBoost
xgb_model = xgb.XGBClassifier()
xgb_model.load_model(XGB_MODEL_PATH)

X_background = pd.read_csv("enron_spamassassin_nazario_nigerian_best_compo/extracted_urls/combined_train_set_features_filtered.csv").drop(columns=['label', 'id', 'url'], errors='ignore')

explainer = shap.Explainer(xgb_model, X_background)

# 🔹 Assurer un format d'URL correct
def ensure_url_format(url):
    if not url.startswith("http://") and not url.startswith("https://"):
        return "http://" + url
    return url

# 🔹 Raccourcissement via API TinyURL directement (plus fiable que pyshorteners)
def shorten_tinyurl_direct(url, timeout=10):
    try:
        url = ensure_url_format(url)
        api_url = f"https://tinyurl.com/api-create.php?url={url}"
        response = requests.get(api_url, timeout=timeout)
        if response.status_code == 200:
            return response.text.strip(), "tinyurl"
        else:
            print(f"❌ TinyURL API error {response.status_code} for {url}")
            return url, "none"
    except Exception as e:
        print(f"❌ Exception TinyURL: {url} — {e}")
        return url, "none"

# 🔹 Raccourcisseur clck.ru (facultatif)
def shorten_clck(url):
    try:
        response = requests.get(f"https://clck.ru/--?url={url}", timeout=5)
        return response.text.strip(), "clck"
    except:
        return url, "none"
    

def replace_domain(url, new_domain="safe.tinyURLs.com"):
    try:
        parsed = urlparse(url)
        # Recompose l'URL avec le nouveau domaine
        new_url = f"{parsed.scheme}://{new_domain}{parsed.path}"
        return new_url
    except Exception as e:
        print(f"Erreur remplacement de domaine pour {url}: {e}")
        return url


def shorten_url(url, service):
    url = ensure_url_format(url)

    try:
        if service == "tinyurl":
            shortened, _ = shorten_tinyurl_direct(url)
            #if shortened != url and shortened != "none":
                #shortened = replace_domain(shortened, new_domain="safe.tinyURLs.com")
        elif service == "clck":
            shortened, _ = shorten_clck(url)
        elif service == "isgd":
            s = pyshorteners.Shortener()
            shortened = s.isgd.short(url)
        elif service == "dagd":
            s = pyshorteners.Shortener()
            shortened = s.dagd.short(url)
        else:
            print(f"❌ Service inconnu ou non pris en charge : {service}")
            return url, "none"


        return shortened, service

    except Exception as e:
        print(f"❌ Erreur raccourcissement avec {service} pour {url}: {e}")
        return url, "none"



# 🔹 NLP pipeline
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

# 🔹 Traitement complet
def process_and_predict(input_csv, output_csv, services=["tinyurl"]):
    df = pd.read_csv(input_csv)
    if "url" not in df.columns or "label" not in df.columns:
        raise ValueError("❌ Le fichier doit contenir les colonnes 'url' et 'label'.")

    for service in services:
        urls_shortened = []
        success_flags = []

        print("ici 1")
        for url in df["url"]:
            print("ici 2")
            short, used = shorten_url(url, service)
            if short != url and short != "none":
                success_flags.append(True)
            else:
                success_flags.append(False)

            urls_shortened.append(short)
            time.sleep(0.5)

        # Export temporaire
        temp_df = pd.DataFrame({"url": urls_shortened})
        temp_df["index"] = df.index
        temp_df["label"] = df.label
        temp_df.to_csv(TEMP_URLS_FILE, index=False)

        # NLP pipeline
        run_nlp_pipeline(TEMP_URLS_FILE)

        # Prédictions
        features_df = pd.read_csv(TEMP_FEATURES_FILE)
        features_df = features_df.drop(columns=["url", "id", "label"], errors="ignore")
        url_probas = xgb_model.predict_proba(features_df)[:, 1]

        # === Masque de succès ===
        mask_success = np.array(success_flags)

        # Sélectionner seulement les URLs raccourcies
        urls_shortened_success = np.array(urls_shortened)[mask_success]
        features_success = features_df.loc[mask_success].reset_index(drop=True)
        probas_orig_success = df["label"][mask_success].values
        probas_mod_success = url_probas[mask_success]

        if len(features_success) > 0:
            # === SHAP sur URLs raccourcies ===
            shap_values_success = explainer(features_success)

            print("\n📊 Affichage des SHAP plots uniquement pour les URLs raccourcies correctement :")
            shap.summary_plot(shap_values_success, features_success)
            plt.show()

            shap.plots.bar(shap_values_success.mean(0), max_display=30)
            plt.show()

            # Identifier les extrêmes
            idx_highest = np.argmax(probas_mod_success)
            idx_lowest = np.argmin(probas_mod_success)

            print(f"\n📈 URL avec la plus HAUTE proba de phishing : {urls_shortened_success[idx_highest]} — proba: {probas_mod_success[idx_highest]:.4f}")
            print(f"\n📉 URL avec la plus BASSE proba de phishing : {urls_shortened_success[idx_lowest]} — proba: {probas_mod_success[idx_lowest]:.4f}")

            # Détail SHAP pour highest
            shap_values_high = explainer(features_success.iloc[[idx_highest]])
            df_shap_high = pd.DataFrame({
                "feature": features_success.columns,
                "value": features_success.iloc[idx_highest].values,
                "shap_value": shap_values_high.values[0]
            }).sort_values(by="shap_value", key=abs, ascending=False)

            print("\n🔥 Top 5 SHAP features pour l'URL la plus phishing :")
            print(df_shap_high.head(5))

            # Détail SHAP pour lowest
            shap_values_low = explainer(features_success.iloc[[idx_lowest]])
            df_shap_low = pd.DataFrame({
                "feature": features_success.columns,
                "value": features_success.iloc[idx_lowest].values,
                "shap_value": shap_values_low.values[0]
            }).sort_values(by="shap_value", key=abs, ascending=False)

            print("\n📦 Top 5 SHAP features pour l'URL la plus légitime :")
            print(df_shap_low.head(5))

            # Drop phishing score
            still_phishing_mask = probas_mod_success > 0.5
            if still_phishing_mask.any():
                mean_drop = (probas_orig_success[still_phishing_mask] - probas_mod_success[still_phishing_mask]).mean()
                print(f"\n📉 Mean drop pour les URLs raccourcies et encore détectées phishing : {mean_drop:.4f}")
            else:
                print("\n📉 Aucun phishing détecté parmi les URLs raccourcies.")
        else:
            print("\n❌ Aucune URL n'a été raccourcie correctement, pas de SHAP plot possible.")

        # 🔹 Ajout dans ton DataFrame final
        df[f"url_shortened_{service}"] = urls_shortened
        df[f"proba_{service}"] = url_probas

        nb_phish = sum(url_probas > 0.5)
        print(f"\n📊 Résumé pour {service}:")
        print(f"   → Détectés phishing : {nb_phish}/{len(df)} ({(nb_phish / len(df)) * 100:.2f}%)")
        print("-" * 50)

    # 🔚 Sauvegarde finale
    df.to_csv(output_csv, index=False)
    print(f"\n✅ Résultats enregistrés dans {output_csv}")


    


# 🔹 Lancement
if __name__ == "__main__":
    INPUT = "urls_attacks/selected_phishing_urls_nlp_train_set.csv"
    OUTPUT = "urls_attacks/experience_3_shortener/shortened_urls_features.csv"
    process_and_predict(INPUT, OUTPUT)