import pandas as pd
from sklearn.utils import shuffle

print("📥 Chargement des datasets...")
enron = pd.read_csv("dataset/emailsEnron.csv")  # Legitimate
enron = enron.rename(columns={"message": "email"})
spamassassin = pd.read_csv("dataset/spam_assassin_original.csv")  # Mix
nazario = pd.read_csv("emails_nazario_original.csv")  # Phishing
nigerian = pd.read_csv("nigerian_emails.csv")  # Phishing

# 🏷️ Ajout des colonnes
enron["label"] = 0
enron["source"] = "Enron"
spamassassin["source"] = "SpamAssassin"
nazario["source"] = "Nazario"
nigerian["source"] = "Nigerian"

# 🧪 Séparer SpamAssassin
spam_legit = spamassassin[spamassassin.label == 0]
spam_phish = spamassassin[spamassassin.label == 1]

# 📦 Phishing complet
phishing_df = pd.concat([spam_phish, nazario, nigerian], ignore_index=True)
phishing_df = shuffle(phishing_df, random_state=42)

# ✅ Legitimate : utiliser SpamAssassin en priorité
required_legit_total = 4431 + 2215 + 6645  # Total needed: train_40 + train_20 + test
if len(spam_legit) >= required_legit_total:
    legit_df = spam_legit.sample(n=required_legit_total, random_state=42)
else:
    needed_from_enron = required_legit_total - len(spam_legit)
    legit_df = pd.concat([
        spam_legit,
        enron.sample(n=needed_from_enron, random_state=42)
    ], ignore_index=True)

legit_df = shuffle(legit_df, random_state=42)

# ✅ Découpage
# Phishing
phish_40 = phishing_df.iloc[:4431]
phish_20 = phishing_df.iloc[4431:4431+2215]
phish_test = phishing_df.iloc[4431+2215:]

# Legitimate
legit_40 = legit_df.iloc[:4431]
legit_20 = legit_df.iloc[4431:4431+2215]
legit_test = legit_df.iloc[4431+2215:4431+2215+6645]

# 📦 Construction finale
train_40 = shuffle(pd.concat([phish_40, legit_40], ignore_index=True), random_state=42)
train_20 = shuffle(pd.concat([phish_20, legit_20], ignore_index=True), random_state=42)
test_set = shuffle(pd.concat([phish_test, legit_test], ignore_index=True), random_state=42)

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
    sources = df['source'].value_counts()
    sources_pct = df['source'].value_counts(normalize=True) * 100
    for src in sources.index:
        print(f"  - {src:<12}: {sources[src]:>5} emails ({sources_pct[src]:.1f}%)")

    # Bonus : détails SpamAssassin
    if "SpamAssassin" in df['source'].values:
        sa_legit = df[(df['source'] == "SpamAssassin") & (df['label'] == 0)].shape[0]
        sa_phish = df[(df['source'] == "SpamAssassin") & (df['label'] == 1)].shape[0]
        print(f"🔍 SpamAssassin - Legitimate: {sa_legit}, Phishing: {sa_phish}")

# 📊 Stats
print_stats(train_40, "Train Set (40%)")
print_stats(train_20, "Train Set (20%)")
print_stats(test_set, "Test Set (40%)")

# 🧼 Garder uniquement email et label dans les sorties
train_40 = train_40[['email', 'label']]
train_20 = train_20[['email', 'label']]
test_set = test_set[['email', 'label']]

# 💾 Sauvegarde
train_40.to_csv("enron_spamassassin_nazario_nigerian_best_compo/original_csv/train_set_40_without_p.csv", index=False)
train_20.to_csv("enron_spamassassin_nazario_nigerian_best_compo/original_csv/train_set_20_without_p.csv", index=False)
test_set.to_csv("enron_spamassassin_nazario_nigerian_best_compo/original_csv/test_set_without_p.csv", index=False)

print("\n✅ Fichiers sauvegardés avec uniquement 'email' et 'label' : train_set_40.csv, train_set_20.csv, test_set.csv")

