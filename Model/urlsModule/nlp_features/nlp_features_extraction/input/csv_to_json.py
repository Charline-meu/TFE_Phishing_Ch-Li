import pandas as pd
import json
import sys
import os

# Vérifie que l'utilisateur a donné un argument
if len(sys.argv) < 2:
    print("❌ Usage: python csv_to_json.py path_to_csv_file")
    sys.exit(1)

# Chemin du fichier CSV fourni en argument
csv_path = sys.argv[1]

# Charger le CSV
df = pd.read_csv(csv_path)

# Supprimer les lignes avec URL manquante ou -1
df = df[df["url"].notna() & (df["url"] != "-1")]

# Ajouter un index si manquant
if "index" not in df.columns:
    df = df.reset_index().rename(columns={"index": "index"})

# Créer le nom de base du fichier
base_name = os.path.splitext(os.path.basename(csv_path))[0]

# Cas 1 : fichier avec labels (phishing vs legitimate)
if "label" in df.columns:
    df_filtered = df[["index", "url", "label"]]

    phishing = df_filtered[df_filtered["label"] == 1][["index", "url"]].to_dict(orient="records")
    legitimate = df_filtered[df_filtered["label"] == 0][["index", "url"]].to_dict(orient="records")

    # Sauvegarde
    with open(f"urlsModule/nlp_features/nlp_features_extraction/input/{base_name}_phishing.json", "w") as f:
        json.dump(phishing, f, indent=2)

    with open(f"urlsModule/nlp_features/nlp_features_extraction/input/{base_name}_legitimate.json", "w") as f:
        json.dump(legitimate, f, indent=2)

    print(f"✅ JSONs generated from {csv_path} → {base_name}_phishing.json / {base_name}_legitimate.json")

# Cas 2 : fichier sans label, juste des URLs
else:
    urls = df[["index", "url"]].to_dict(orient="records")
    with open(f"urlsModule/nlp_features/nlp_features_extraction/input/{base_name}_phishing.json", "w") as f:
        json.dump(urls, f, indent=2)

    print(f"✅ JSON generated from {csv_path} → {base_name}_urls.json (no labels)")
