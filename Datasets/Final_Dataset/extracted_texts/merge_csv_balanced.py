import pandas as pd
import random

# Charger les fichiers
main_df = pd.read_csv("text_train_set_40.csv")
ling_df = pd.read_csv("ling.csv")

# Filtrer les lignes label=1 et label=0 dans ling
ling_pos = ling_df[ling_df['label'] == 1]
ling_neg = ling_df[ling_df['label'] == 0]

# Prendre 40% des positifs et 40% des négatifs
ling_pos_sample = ling_pos.sample(frac=0.4, random_state=42)
ling_neg_sample = ling_neg.sample(n=len(ling_pos_sample), random_state=42)

# Fusionner les deux sous-ensembles
ling_combined = pd.concat([ling_pos_sample, ling_neg_sample], ignore_index=True)

# Ajouter les colonnes manquantes pour correspondre à main_df
ling_combined = ling_combined[['text', 'label']]
ling_combined.insert(0, 'email_id', 0)  # placeholder
ling_combined.insert(0, 'index', 0)     # placeholder

# Réindexer les nouvelles lignes
start_index = main_df['index'].max() + 1
ling_combined['index'] = range(start_index, start_index + len(ling_combined))
ling_combined['email_id'] = ling_combined['index']

# Fusionner avec le dataset principal
final_df = pd.concat([main_df, ling_combined], ignore_index=True)

# Exporter le résultat
final_df.to_csv("text_train_set_augmented.csv", index=False)

print("Fichier augmenté sauvegardé sous text_train_set_augmented.csv")
