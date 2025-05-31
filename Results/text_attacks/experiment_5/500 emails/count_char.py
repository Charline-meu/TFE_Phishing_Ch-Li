import pandas as pd

# Charger le fichier CSV
df = pd.read_csv("experiment_5_500emails.csv")  # Assure-toi que le fichier est dans le même dossier que ton script

# Calculer le nombre de caractères pour chaque ligne
df['original_char_count'] = df['original_text'].fillna('').apply(len)
df['modified_char_count'] = df['new_text'].fillna('').apply(len)

# Calculer la différence
df['difference'] = df['modified_char_count'] - df['original_char_count']

# Afficher les résultats
print(df[['original_char_count', 'modified_char_count', 'difference']])