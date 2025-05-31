import pandas as pd
from email import message_from_string
from email.generator import Generator
from io import StringIO

# Charger le CSV
df = pd.read_csv("structure_attacks/selected_phishing_full_structure.csv")
print(f"📊 Nombre de structures analysées : {len(df)}")

# Liste pour stocker les emails modifiés
modified_emails = []

# Nouveau domaine "plus clean"
new_domain = "test-friendly"

for raw_email in df["email"]:
    msg = message_from_string(raw_email)

    # Extraire et modifier le Message-ID
    msg_id = msg.get("Message-ID")
    if msg_id:
        msg_id_clean = msg_id.strip().lstrip("<").rstrip(">")
        if "@" in msg_id_clean:
            lhs, _ = msg_id_clean.split("@", 1)
            new_msg_id = f"<{lhs}@{new_domain}>"
            msg.replace_header("Message-ID", new_msg_id)
        else:
            # fallback : recréer un ID si mal formé
            new_msg_id = f"<randomid@{new_domain}>"
            msg["Message-ID"] = new_msg_id
    else:
        # fallback si pas de Message-ID
        msg["Message-ID"] = f"<generatedid@{new_domain}>"

    # Convertir le message en texte brut (string)
    with StringIO() as output:
        gen = Generator(output, mangle_from_=False)
        gen.flatten(msg)
        modified_email = output.getvalue()
        modified_emails.append(modified_email)

# Créer un nouveau DataFrame avec les e-mails modifiés
df_modified = df.copy()
df_modified["email"] = modified_emails

# Sauvegarder pour réutilisation
df_modified.to_csv("structure_attacks/experiment_1_message_id/selected_phishing_modified_domain.csv", index=False)
print(f"📊 Nombre de structures analysées : {len(df_modified)}")
print("✅ E-mails modifiés avec nouveau domaine dans le Message-ID enregistrés.")
