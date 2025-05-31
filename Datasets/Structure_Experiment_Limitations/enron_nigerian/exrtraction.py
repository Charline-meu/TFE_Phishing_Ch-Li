import pandas as pd
from sklearn.utils import shuffle

# 📥 Chargement des données
print("📥 Chargement des données...")
enron = pd.read_csv("dataset/emailsEnron.csv").rename(columns={"message": "email"})
nigerian = pd.read_csv("nigerian_emails.csv")

# 🏷️ Ajout des colonnes
enron["label"] = 0
enron["source"] = "Enron"
nigerian["label"] = 1
nigerian["source"] = "Nigerian"

# 📊 Statistiques initiales
print(f"📦 Enron (legit): {len(enron)} emails")
print(f"📦 Nigerian (phish): {len(nigerian)} emails")

# ✂️ Tailles cibles
train_phish = 3536
train_legit = 3536
test_phish = 442
test_legit = 1326

# 🧪 Construction des sets
phish_train_df = nigerian.iloc[:train_phish]
phish_test_df = nigerian.iloc[train_phish:train_phish + test_phish]

legit_train_df = enron.iloc[:train_legit]
legit_test_df = enron.iloc[train_legit:train_legit + test_legit]

# 📦 Concaténation
train_set = shuffle(pd.concat([phish_train_df, legit_train_df], ignore_index=True), random_state=42)
test_set = shuffle(pd.concat([phish_test_df, legit_test_df], ignore_index=True), random_state=42)

# 📊 Fonction stats
def print_stats(df, name):
    total = len(df)
    legit = (df.label == 0).sum()
    phish = (df.label == 1).sum()
    print(f"\n📊 {name}")
    print(f"➡️ Total: {total}")
    print(f"✔️ Legitimate: {legit} ({legit/total:.1%})")
    print(f"❗ Phishing:    {phish} ({phish/total:.1%})")
    print("📦 Source breakdown:")
    print(df['source'].value_counts())

# 🔍 Stats
print_stats(train_set, "Train Set")
print_stats(test_set, "Test Set")

# 💾 Sauvegarde
train_set[['email', 'label']].to_csv("enron_nigerian/train_set_80.csv", index=False)
test_set[['email', 'label']].to_csv("enron_nigerian/test_set_20.csv", index=False)

print("\n✅ Fichiers sauvegardés : train_set_balanced_80.csv, test_set_unbalanced_20.csv")
