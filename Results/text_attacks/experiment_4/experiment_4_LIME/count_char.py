import pandas as pd

# Charger le fichier CSV
df = pd.read_csv("experiment_4_LIME.csv")

# Remplir les valeurs manquantes avec une chaîne vide
df['original_text'] = df['original_text'].fillna('')
df['new_text_2'] = df['new_text_2'].fillna('')

# Calculer le nombre de caractères
df['original_char_count'] = df['original_text'].apply(len)
df['modified_char_count'] = df['new_text_2'].apply(len)
df['char_difference'] = df['modified_char_count'] - df['original_char_count']

# Calculer le nombre de mots
df['original_word_count'] = df['original_text'].apply(lambda x: len(x.split()))
df['modified_word_count'] = df['new_text_2'].apply(lambda x: len(x.split()))
df['word_difference'] = df['modified_word_count'] - df['original_word_count']

# Afficher les résultats ligne par ligne
print(df[['original_char_count', 'modified_char_count', 'char_difference',
          'original_word_count', 'modified_word_count', 'word_difference']])

# Calculer et afficher les moyennes
print("\nAverage values:")
print(df[['original_char_count', 'modified_char_count', 'char_difference',
          'original_word_count', 'modified_word_count', 'word_difference']].mean())