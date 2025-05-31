import pandas as pd
import numpy as np
import re
from langdetect import detect, LangDetectException

# --------- Prétraitement --------- #
def preprocess_text(text):
    if text == "-1":
        return np.nan

    try:
        if detect(text) != 'en':
            return np.nan
    except LangDetectException:
        return np.nan

    text = re.sub(r"http\S+|www\S+|https\S+", "", text)  # Supprimer les URLs
    text = re.sub(r"\d+", "", text)  # Supprimer les nombres
    text = re.sub(r"\S+@\S+\.\S+", "", text)  # Supprimer les adresses e-mail
    text = re.sub(r"\s+", " ", text).strip()  # Normaliser les espaces
    return text

# --------- Analyse --------- #
def analyze_email_csv(file_path):
    df = pd.read_csv(file_path)

    # Statistiques avant
    total_before = len(df)
    phishing_before = df['label'].sum()
    legit_before = (df['label'] == 0).sum()

    print("------ AVANT PREPROCESSING ------")
    print(f"Nombre total d'emails     : {total_before}")
    print(f"Phishing                  : {phishing_before}")
    print(f"Légitimes                 : {legit_before}")

    # Appliquer le prétraitement
    df['text_clean'] = df['text'].apply(preprocess_text)

    # Supprimer les NaN
    df_cleaned = df.dropna(subset=['text_clean'])

    # Statistiques après
    total_after = len(df_cleaned)
    phishing_after = df_cleaned['label'].sum()
    legit_after = (df_cleaned['label'] == 0).sum()

    print("\n------ APRÈS PREPROCESSING ------")
    print(f"Nombre total d'emails     : {total_after}")
    print(f"Phishing                  : {phishing_after}")
    print(f"Légitimes                 : {legit_after}")

    # Différences
    print("\n------ DIFFÉRENCE ------")
    print(f"Emails supprimés          : {total_before - total_after}")
    print(f"Phishing supprimés        : {phishing_before - phishing_after}")
    print(f"Légitimes supprimés       : {legit_before - legit_after}")

# --------- Exemple d'exécution --------- #
if __name__ == "__main__":
    # Remplace ce chemin par celui de ton fichier CSV
    fichier_csv = "text_train_set_augmented.csv"
    analyze_email_csv(fichier_csv)