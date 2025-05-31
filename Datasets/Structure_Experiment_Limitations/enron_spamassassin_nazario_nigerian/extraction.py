import pandas as pd
from sklearn.utils import shuffle

# 📥 Chargement des datasets
print("📥 Chargement des datasets...")
spamassassin = pd.read_csv("dataset/spam_assassin_original.csv")
nigerian = pd.read_csv("nigerian_emails.csv")
nazario = pd.read_csv("emails_nazario_original.csv")
enron = pd.read_csv("dataset/emailsEnron.csv").rename(columns={"message": "email"})

# 🔖 Ajout des colonnes manquantes
nigerian["label"] = 1
nazario["label"] = 1
enron["label"] = 0

spamassassin["source"] = "SpamAssassin"
nigerian["source"] = "Nigerian"
nazario["source"] = "Nazario"
enron["source"] = "Enron"

# ✅ Séparation légitimes / phishing
spam_legit = spamassassin[spamassassin.label == 0]
spam_phish = spamassassin[spamassassin.label == 1]

# 📦 Phishing complet
phishing_df = pd.concat([spam_phish, nigerian, nazario], ignore_index=True)
phishing_df = shuffle(phishing_df, random_state=42)

# 📦 Legitimate : combiner spam_legit + enron pour avoir assez
total_legit_needed = 7878 + 2952  # train + test = 10_830
spam_legit = shuffle(spam_legit, random_state=42)
enron = shuffle(enron, random_state=42)

legit_df = pd.concat([
    spam_legit,
    enron.iloc[:(total_legit_needed - len(spam_legit))]
], ignore_index=True)

# ✅ Répartition des ensembles
train_phish = 7878
train_legit = 7878
test_phish = 984
test_legit = 2952

phish_train_df = phishing_df.iloc[:train_phish]
phish_test_df = phishing_df.iloc[train_phish:train_phish + test_phish]

legit_train_df = legit_df.iloc[:train_legit]
legit_test_df = legit_df.iloc[train_legit:train_legit + test_legit]

# 📦 Construction finale
train_set = shuffle(pd.concat([phish_train_df, legit_train_df], ignore_index=True), random_state=42)
test_set = shuffle(pd.concat([phish_test_df, legit_test_df], ignore_index=True), random_state=42)

# 💾 Sauvegarde
train_set[['email', 'label']].to_csv("enron_spamassassin_nazario_nigerian/train_set_80.csv", index=False)
test_set[['email', 'label']].to_csv("enron_spamassassin_nazario_nigerian/test_set_20.csv", index=False)

# 📊 Stats
def print_stats(df, name):
    total = len(df)
    legit = (df.label == 0).sum()
    phish = (df.label == 1).sum()
    print(f"\n📊 {name}")
    print(f"➡️ Total: {total}")
    print(f"✔️ Legitimate: {legit} ({legit / total:.1%})")
    print(f"❗ Phishing:    {phish} ({phish / total:.1%})")

print_stats(train_set, "Train Set (80%)")
print_stats(test_set, "Test Set (20%)")

print("\n✅ Fichiers générés : train_set_balanced_80.csv et test_set_unbalanced_20.csv")
