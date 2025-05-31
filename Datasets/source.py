import pandas as pd

# Charger les datasets d'origine
enron = pd.read_csv("dataset/emailsEnron.csv")
spamassassin = pd.read_csv("dataset/spam_assassin_original.csv")
nazario = pd.read_csv("emails_nazario_original.csv")
nigerian = pd.read_csv("nigerian_emails.csv")

# Charger le dataset principal
features = pd.read_csv("enron_spamassassin_nazario_nigerian_best_compo/original_csv/train_set_60.csv")

# Vérifie que la colonne contenant le contenu de l'email s'appelle bien 'email' dans chaque fichier
# Si c’est différent (ex: 'content', 'text', etc), adapte ici :
def normalize_text(series):
    return set(series.astype(str).str.strip().str.lower())

# Normalisation des contenus
enron_emails = normalize_text(enron['message'])
spamassassin_emails = normalize_text(spamassassin['email'])
nazario_emails = normalize_text(nazario['email'])
nigerian_emails = normalize_text(nigerian['email'])

# Même chose pour le fichier principal
features['email_normalized'] = features['email'].astype(str).str.strip().str.lower()

# Identifier la source
def find_source(email):
    if email in enron_emails:
        return "enron"
    elif email in spamassassin_emails:
        return "spamassassin"
    elif email in nazario_emails:
        return "nazario"
    elif email in nigerian_emails:
        return "nigerian"
    else:
        return "unknown"

features["source"] = features["email_normalized"].apply(find_source)

# Supprimer la colonne temporaire si besoin
features.drop(columns=["email_normalized"], inplace=True)

# Sauvegarde
features.to_csv("features_with_source.csv", index=False)



# Charger le fichier de base
features_base = pd.read_csv("enron_spamassassin_nazario_nigerian_best_compo/extracted_features/features_train_set_60.csv")

# Charger celui qui contient la colonne 'source'
features_with_source = pd.read_csv("features_with_source.csv")

# Ajouter la colonne 'source' directement
features_base["source"] = features_with_source["source"]

# Réécrire le fichier original avec la colonne ajoutée
features_base.to_csv("enron_spamassassin_nazario_nigerian_best_compo/extracted_features/features_train_set_60.csv", index=False)