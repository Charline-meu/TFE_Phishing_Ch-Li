import pandas as pd

# Charger les fichiers CSV
nazario_df = pd.read_csv("emails_nazario_original.csv")  # Dataset Nazario
spam_assassin_df = pd.read_csv("dataset/spam_assassin_original.csv")  # Dataset SpamAssassin
enron_df = pd.read_csv("dataset/emailsEnron.csv")  # Dataset Enron

# Suppression de la colonne "index" dans SpamAssassin si elle existe
if "index" in spam_assassin_df.columns:
    spam_assassin_df = spam_assassin_df.drop(columns=["index"])

# Séparer phishing et légitimes dans SpamAssassin
spam_phishing = spam_assassin_df[spam_assassin_df["label"] == 1]  # 1899 phishing
spam_legit = spam_assassin_df[spam_assassin_df["label"] == 0]  # 4153 légitimes

# Modifier Enron : garder seulement "message" et le renommer en "email"
enron_df = enron_df.rename(columns={"message": "email"})[["email"]]

# Ajouter la colonne "label" à Enron (qui est toujours légitime)
enron_df["label"] = 0

# Prendre tous les emails de Nazario (phishing)
all_phishing = pd.concat([nazario_df, spam_phishing], ignore_index=True)  # 2985 + 1899 = 4884 phishing

# Prendre tous les légitimes de SpamAssassin et 7000 de Enron
all_legit = pd.concat([spam_legit, enron_df.sample(n=7000, random_state=42)], ignore_index=True)  # 4153 + 7000 = 11153 légitimes

# Taille cible pour Training Set 40% et Test Set
target_size = 3908  # Même taille pour train_set_40 et test_set

# ----------- 1. Création du Training Set 40% (Équilibré) -----------
train_40_phishing = all_phishing.sample(n=target_size // 2, random_state=42)  # 50% phishing
train_40_legit = all_legit.sample(n=target_size // 2, random_state=42)  # 50% légitimes

train_set_40 = pd.concat([train_40_phishing, train_40_legit], ignore_index=True)

# ----------- 2. Création du Training Set 20% (Équilibré) -----------
train_20_phishing = all_phishing.drop(train_40_phishing.index).sample(n=target_size // 2 // 2, random_state=42)  
train_20_legit = all_legit.drop(train_40_legit.index).sample(n=target_size // 2 // 2, random_state=42)  

train_set_20 = pd.concat([train_20_phishing, train_20_legit], ignore_index=True)

# ----------- 3. Création du Test Set 40% (Déséquilibré) -----------
test_phishing = all_phishing.drop(train_40_phishing.index).drop(train_20_phishing.index).sample(n=target_size // 4, random_state=42)  
test_legit = all_legit.drop(train_40_legit.index).drop(train_20_legit.index).sample(n=target_size - len(test_phishing), random_state=42)  

test_set = pd.concat([test_phishing, test_legit], ignore_index=True)

# ----------- 4. Sauvegarde des CSV -----------
train_set_40.to_csv("enron_spamassassin_nazario/train_set_40.csv", index=False)
train_set_20.to_csv("enron_spamassassin_nazario/train_set_20.csv", index=False)
test_set.to_csv("enron_spamassassin_nazario/test_set.csv", index=False)


# Vérifier la composition des datasets
def check_dataset_distribution(df, dataset_name):
    # Phishing counts
    nazario_count = df[df["email"].isin(nazario_df["email"])].shape[0]
    spam_phishing_count = df[df["email"].isin(spam_phishing["email"])].shape[0]
    
    # Legitimate counts
    spam_legit_count = df[df["email"].isin(spam_legit["email"])].shape[0]
    enron_count = df[df["email"].isin(enron_df["email"])].shape[0]

    print(f"\n📊 {dataset_name} - Composition:")
    print(f"   - 📌 Phishing Emails: {nazario_count + spam_phishing_count}")
    print(f"     - Nazario: {nazario_count} ({nazario_count / max(1, (nazario_count + spam_phishing_count)):.2%})")
    print(f"     - SpamAssassin: {spam_phishing_count} ({spam_phishing_count / max(1, (nazario_count + spam_phishing_count)):.2%})")
    
    print(f"   - ✅ Legitimate Emails: {spam_legit_count + enron_count}")
    print(f"     - SpamAssassin: {spam_legit_count} ({spam_legit_count / max(1, (spam_legit_count + enron_count)):.2%})")
    print(f"     - Enron: {enron_count} ({enron_count / max(1, (spam_legit_count + enron_count)):.2%})")
    print("-" * 50)

# Vérification des ratios dans chaque dataset
check_dataset_distribution(train_set_40, "Training Set 40%")
check_dataset_distribution(train_set_20, "Training Set 20%")
check_dataset_distribution(test_set, "Test Set")


print("✅ CSV générés : train_set_40.csv, train_set_20.csv, test_set.csv")