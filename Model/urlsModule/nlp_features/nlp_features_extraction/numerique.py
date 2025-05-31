import pandas as pd

# Charger le fichier
df = pd.read_csv("nlp_features_ebu.csv")

# Transformer la colonne 'class' en numérique : phish = 1, legitimate = 0
df["class"] = df["class"].map({"phish": 1, "legitimate": 0})

# Sélectionner les colonnes numériques + la colonne 'url'
df_numeric = df.select_dtypes(include=['number'])
df_numeric["url"] = df["url"]

# Réorganiser les colonnes (mettre 'url' en premier si tu veux)
cols = ['url'] + [col for col in df_numeric.columns if col != 'url']
df_numeric = df_numeric[cols]

# Sauvegarder dans un nouveau fichier CSV
df_numeric.to_csv("nlp_features_ebu_numerique.csv", index=False)

print("✅ CSV généré avec colonnes numériques + url + class.")
