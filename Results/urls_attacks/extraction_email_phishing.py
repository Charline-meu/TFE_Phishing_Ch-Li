import pandas as pd


# Charger ton fichier de prédictions
df = pd.read_csv("enron_spamassassin_nazario_nigerian_best_compo/extracted_urls/nlp_proba_test_all.csv")  # ← adapte le nom si besoin

# Ajouter la colonne 'predicted_label' si elle n'existe pas
if "predicted_label" not in df.columns:
    df["predicted_label"] = (df["phishing_probability"] > 0.5).astype(int)

# Filtrer : bien phishing et bien détecté
phishing_correct = df[(df["label"] == 1) & (df["predicted_label"] == 1)]

# En prendre 15 aléatoirement
selected = phishing_correct.sample(n=100, random_state=42)

# Sauvegarder dans un nouveau fichier
selected.to_csv("urls_attacks/selected_phishing_urls_nlp_train_set.csv", index=False)
print("✅ 100 URLs phishing correctement détectées enregistrées dans selected_phishing_urls.csv")
