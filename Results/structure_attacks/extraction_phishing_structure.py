import pandas as pd
from email import message_from_string

# Fonction pour parser l'e-mail brut
def load_email_from_string(email_raw):
    return message_from_string(email_raw)

# Charger les fichiers
df_proba = pd.read_csv("enron_spamassassin_nazario_nigerian_best_compo/extracted_features/features_proba_test_set.csv")
df_full = pd.read_csv("enron_spamassassin_nazario_nigerian_best_compo/original_csv/test_set.csv")



# Nettoyer les message IDs dans proba
df_proba["message_id"] = (
    df_proba["email_id"]
    .astype(str)
    .str.strip()
    .str.replace("<", "", regex=False)
    .str.replace(">", "", regex=False)
)

# Supprimer les lignes avec message_id vide dans proba
df_proba = df_proba[df_proba["message_id"].notna() & (df_proba["message_id"] != "")]

# Extraire les Message-ID depuis les e-mails du test_set
message_ids = []
for email in df_full["email"]:
    try:
        msg = load_email_from_string(email)
        msg_id = msg.get("Message-ID")
        if msg_id:
            msg_id = msg_id.strip().lstrip("<").rstrip(">")
        message_ids.append(msg_id)
    except:
        message_ids.append(None)

df_full["message_id"] = message_ids

# Supprimer les e-mails sans message_id dans test_set
df_full = df_full[df_full["message_id"].notna() & (df_full["message_id"] != "")]

# Filtrer les phishing bien détectés avec message_id
phishing_correct = df_proba[
    (df_proba["label"] == 1)
    & (df_proba["phishing_probability"] > 0.5)
]

# Garde uniquement ceux dont le message_id est aussi présent dans df_full
valid_ids = set(df_full["message_id"])
phishing_correct = phishing_correct[phishing_correct["message_id"].isin(valid_ids)]

# Sélectionner 100 aléatoirement
selected = phishing_correct.sample(n=100, random_state=42)

# Fusion avec les e-mails complets
merged = pd.merge(selected, df_full[['email', 'message_id', 'label']], on="message_id", how="inner")
merged = pd.merge(selected, df_full[['email', 'message_id', 'label']], on="message_id", how="inner")

# Nettoyer les colonnes : on garde un seul label
merged = merged.rename(columns={"label_x": "label"})
merged = merged.drop(columns=["label_y"])

# Exporter
merged.to_csv("structure_attacks/selected_phishing_full_structure.csv", index=False)

print("✅ 100 e-mails phishing avec Message-ID valides et label enregistrés.")
