#pip install pandas beautifulsoup4 lxml
import pandas as pd
from bs4 import BeautifulSoup
import re
import csv

# 📂 Charger les e-mails depuis un fichier CSV (doit contenir des colonnes 'body' et 'label')
data = pd.read_csv('dataset/spam_assassin_original.csv')  # 'body' contient le contenu de l'e-mail, 'label' contient le label de l'e-mail

# 🧹 Fonction pour extraire les URLs du corps des e-mails avec leur label
def extract_urls_with_labels(email_body: str, label: str) -> list:
    url_label_pairs = []

    if isinstance(email_body, str):  # ✅ Éviter les erreurs sur les valeurs NaN
        soup = BeautifulSoup(email_body, 'lxml')

        # 🎯 Extraire les balises <a> avec href tout en filtrant les "mailto:" même obfusqués
        for link in soup.find_all('a', href=True):
            url = link['href'].strip()

            # 💡 Nettoyer les URLs obfusquées avec "3D"
            url = re.sub(r'^3D"', '', url)  # Supprime le préfixe "3D"
            url = url.strip('"')# Supprime les guillemets restants

            # ❌ Filtrer les "mailto:" même après nettoyage
            if not url.lower().startswith('mailto:'):
                url_label_pairs.append((url, label))

        # 🎯 Utiliser des expressions régulières pour trouver uniquement les URLs HTTP/HTTPS
        urls = re.findall(r'(https?://[^\s"\'>]+)', email_body)
        url_label_pairs.extend([(url, label) for url in urls])

    return url_label_pairs

# 🚦 Extraire les URLs et les labels pour chaque e-mail
all_url_label_pairs = []

for _, row in data.iterrows():
    email_body = row['text']
    email_label = row['target']
    url_label_pairs = extract_urls_with_labels(email_body, email_label)
    all_url_label_pairs.extend(url_label_pairs)

# 🧾 Créer un DataFrame avec les URLs et leurs labels associés
urls_data = pd.DataFrame(all_url_label_pairs, columns=['url', 'label'])
# Supprimer les lignes avec des URLs vides (juste par précaution)
urls_data.dropna(subset=['url'], inplace=True)

def clean_url(url: str) -> str:
    if isinstance(url, str):
        # 🧽 Supprimer les guillemets et espaces superflus
        url = url.strip().strip('"').strip('>,').strip('"')

        # 🎯 Utiliser une expression régulière pour extraire uniquement l'URL correcte
        url_match = re.search(r'(https?://[^\s"\'>]+)', url)
        if url_match:
            url = url_match.group(0)
        url = re.sub(r'["<>]', '', url)
        # Supprimer les guillemets doubles autour des URL si présents

        # Supprimer les préfixes indésirables
        url = re.sub(r'^"+|"+$', '', url)
        url = re.sub(r'^\'|\'$', '', url)
    return url

# Appliquer le nettoyage à toutes les URLs
urls_data['url'] = urls_data['url'].apply(clean_url)


# 📊 Afficher un aperçu des données extraites
print(urls_data.head())

# 💾 Sauvegarder les données dans un fichier CSV prêt pour l'entraînement du modèle CNN-LSTM
urls_data.to_csv('extracted_urls_spam_assassin.csv', index=False)
print("✅ Fichier 'extracted_urls_spam_assassin.csv' sauvegardé avec succès !")
