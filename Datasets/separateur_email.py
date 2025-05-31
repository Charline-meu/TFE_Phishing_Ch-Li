import mailbox
import csv

def extract_emails_from_mbox(mbox_path, output_csv):
    """ Extrait tous les emails d'un fichier .mbox et les enregistre dans un CSV """
    emails_list = []

    try:
        # 📩 Ouvrir le fichier .mbox
        with open(mbox_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()

        # 📌 Séparer les emails avec "From "
        email_blocks = content.split("\nFrom ")

        for i, email in enumerate(email_blocks):
            email = email.strip()
            if email:
                if i > 0:  # Ajouter "From " devant tous sauf le premier bloc
                    email = "From " + email
                emails_list.append(email)  # Ajouter l'e-mail à la liste

    except Exception as e:
        print(f"❌ Impossible d'ouvrir {mbox_path} : {e}")
        return

    # 📥 Sauvegarder dans un fichier CSV
    with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["email", "label"])  # En-tête CSV
        for email in emails_list:
            writer.writerow([email, 0])  # Stocker chaque email avec label=0

    print(f"✅ Extraction terminée ! {len(emails_list)} e-mails enregistrés dans {output_csv}")

# ------------------------------#
# 📌 Utilisation du script #
# ------------------------------#
mbox_file = "email_perso_liza/Favoris.mbox"  # 📂 Ton fichier `.mbox`
output_csv = "email_perso_liza/emails_perso_liza.csv"  # 📥 Fichier de sortie

extract_emails_from_mbox(mbox_file, output_csv)
