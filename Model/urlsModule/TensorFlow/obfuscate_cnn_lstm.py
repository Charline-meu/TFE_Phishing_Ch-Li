import pandas as pd
import numpy as np
import tensorflow as tf
import pyshorteners
import random
import requests
import time
from urlsModule.TensorFlow.cnnLstmTensorFlow import encode_url

# 🔧 CONFIG
CNN_MODEL_PATH = "urlsModule/TensorFlow/cnn_lstm_tf_model_ebu.keras"
cnn_model = tf.keras.models.load_model(CNN_MODEL_PATH)
AVAILABLE_SHORTENERS = ["tinyurl", "isgd", "dagd", "clck"]

# 🔹 Raccourcisseur personnalisé clck.ru
def shorten_clck(url):
    try:
        response = requests.get(f"https://clck.ru/--?url={url}", timeout=5)
        return response.text.strip()
    except:
        return url

# 🔹 Shorten a URL with the given service
def shorten_url(url, method):
    try:
        if method == "clck":
            return shorten_clck(url)
        else:
            s = pyshorteners.Shortener()
            return getattr(s, method).short(url)
    except:
        return url

# 🔹 Predict with CNN
def predict_url_proba(url):
    encoded = encode_url(url)
    return cnn_model.predict(np.array([encoded]), verbose=0)[0][0]

# 🔹 Main processing
def process_urls(input_csv, output_csv, shortener="all"):
    df = pd.read_csv(input_csv)

    if "url" not in df.columns or "label" not in df.columns:
        raise ValueError("Le fichier doit contenir les colonnes 'url' et 'label'.")

    print(f"🔍 Traitement des URLs avec shortener = '{shortener}'...")

    # Proba de l’URL originale
    df["original_proba"] = df["url"].apply(predict_url_proba)

    # Appliquer tous les shorteners
    used_shorteners = AVAILABLE_SHORTENERS if shortener == "all" else [shortener]

    for method in used_shorteners:
        url_col = f"url_{method}"
        proba_col = f"proba_{method}"

        urls_short = []
        probas_short = []

        for i, row in df.iterrows():
            short_url = shorten_url(row["url"], method)
            proba = predict_url_proba(short_url)
            urls_short.append(short_url)
            probas_short.append(proba)
            print(f"[{i+1}/{len(df)}] {method} | {short_url} | proba: {proba:.4f}")
            time.sleep(0.5)

        df[url_col] = urls_short
        df[proba_col] = probas_short

        # 🔍 Statistiques pour ce shortener
        predicted_as_phishing = np.sum(np.array(probas_short) > 0.5)
        total = len(probas_short)
        percent = (predicted_as_phishing / total) * 100

        print(f"\n📊 Résumé pour {method}:")
        print(f"   → Détectés phishing : {predicted_as_phishing}/{total} ({percent:.2f}%)")
        print("-" * 50)


    df.to_csv(output_csv, index=False)
    print(f"\n✅ Résultats sauvegardés dans : {output_csv}")

# 🔹 Exécution
if __name__ == "__main__":
    INPUT_CSV = "selected_phishing_urls.csv"
    OUTPUT_CSV = "phishing_urls_all_shortened.csv"

    # 🧩 Tu peux changer ici
    # shortener = "all" → tous les shorteners
    # shortener = "tinyurl", "clck", etc. → juste un
    process_urls(INPUT_CSV, OUTPUT_CSV, shortener="all")
