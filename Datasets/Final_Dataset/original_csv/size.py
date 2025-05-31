import pandas as pd

try:
    spamassassin = pd.read_csv("dataset/spam_assassin_original.csv")
except Exception as e:
    print("❌ Erreur chargement SpamAssassin:", e)
    spamassassin = pd.DataFrame()

try:
    nazario = pd.read_csv("emails_nazario_original.csv")
except Exception as e:
    print("❌ Erreur chargement Nazario:", e)
    nazario = pd.DataFrame()

try:
    nigerian = pd.read_csv("nigerian_emails.csv")
except Exception as e:
    print("❌ Erreur chargement Nigerian:", e)
    nigerian = pd.DataFrame()

try:
    english_pub_liza = pd.read_csv("emails_eml/Perso_Liza/pub_english_liza/pub_english_liza.csv")
except Exception as e:
    print("❌ Erreur chargement EnglishPubLiza:", e)
    english_pub_liza = pd.DataFrame()

enron_nazario = pd.read_csv("enron_nazario/merged_enron_nazario.csv")

print("\n📊 Statistiques par fichier :\n")

if not enron_nazario.empty:
    count_legit = (enron_nazario['label'] == 0).sum()
    count_phish = (enron_nazario['label'] == 1).sum()
    print(f"✅ enron_nazario    : {len(enron_nazario)} emails")
    print(f"   └─ Legitimate : {count_legit}")
    print(f"   └─ Phishing   : {count_phish}")

if not spamassassin.empty:
    count_legit = (spamassassin['label'] == 0).sum()
    count_phish = (spamassassin['label'] == 1).sum()
    print(f"✅ SpamAssassin    : {len(spamassassin)} emails")
    print(f"   └─ Legitimate : {count_legit}")
    print(f"   └─ Phishing   : {count_phish}")

if not nazario.empty:
    print(f"✅ Nazario         : {len(nazario)} emails (Phishing)")

if not nigerian.empty:
    print(f"✅ Nigerian        : {len(nigerian)} emails (Phishing)")

if not english_pub_liza.empty:
    print(f"✅ EnglishPubLiza  : {len(english_pub_liza)} emails (Légitimes)")
