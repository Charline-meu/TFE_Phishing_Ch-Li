import os

# Source des domaines populaires
source_file = "data/allbrand.txt"
output_dir = "data/alexa-tld"

# Lettres + chiffres autorisés pour fichier .txt
valid_start_chars = list("abcdefghijklmnopqrstuvwxyz0123456789") 

# Crée le dossier s'il n'existe pas
os.makedirs(output_dir, exist_ok=True)

# Trie les domaines par première lettre
buckets = {c: [] for c in valid_start_chars}

with open(source_file, "r", encoding="utf-8") as f:
    for line in f:
        domain = line.strip().lower()
        if domain:
            first_char = domain[0]
            if first_char in buckets:
                buckets[first_char].append(domain)

# Écriture dans chaque fichier, même s'il est vide
for key in valid_start_chars:
    with open(os.path.join(output_dir, f"{key}.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(buckets[key]))

print("✅ Tous les fichiers (lettres et chiffres) sont créés dans", output_dir)
