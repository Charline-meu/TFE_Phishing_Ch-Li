import json
import pandas as pd
import sys

# Vérifie que l'utilisateur a donné un argument
if len(sys.argv) < 2:
    print("❌ Usage: python script.py output_filename.csv")
    sys.exit(1)

# Nom du fichier de sortie fourni en argument
output_csv = sys.argv[1]

# Charger les données JSON
with open("urlsModule/nlp_features/nlp_features_extraction/output/features/extraction_features.txt", "r", encoding="utf-8") as f:
    data = json.load(f)

# Tout aplatir : info + url_features + nlp_info + target_words + compoun_words
flattened = []
for item in data:
    info = item["info"]
    url_features = item["url_features"]

    # Extraire et retirer nlp_info du bloc "info"
    nlp_info = info.pop("nlp_info", {})

    # Extraire target_words et compoun_words
    target_words = nlp_info.pop("target_words", {})
    compound_words = nlp_info.pop("compoun_words", {})

    # Aplatir les champs en listes (format string)
    nlp_flat = {k: ", ".join(v) if isinstance(v, list) else v for k, v in nlp_info.items()}
    target_flat = {f"target_{k}": ", ".join(v) if isinstance(v, list) else v for k, v in target_words.items()}
    compound_flat = {f"compound_{k}": ", ".join(v) if isinstance(v, list) else v for k, v in compound_words.items()}

    # Fusionner tous les dictionnaires
    merged = {**info, **url_features, **nlp_flat, **target_flat, **compound_flat}
    flattened.append(merged)

# Création du DataFrame
df = pd.DataFrame(flattened)

# Sauvegarde CSV
df.to_csv(output_csv, index=False)
print(f"✅ CSV généré : {output_csv}")
