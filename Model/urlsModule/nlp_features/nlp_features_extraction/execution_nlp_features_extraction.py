import subprocess
import sys
import os

def run_step(command, step_name):
    print(f"🔹 {step_name}")
    try:
        subprocess.run(command, check=True, shell=True)
        print(f"✅ {step_name} réussi.\n")
    except subprocess.CalledProcessError:
        print(f"❌ Échec de : {step_name}\n")
        sys.exit(1)

if __name__ == "__main__":
    print("🚀 Lancement de la pipeline NLP...")

    # Étape 1 : CSV to JSON
    run_step("py urlsModule/nlp_features/nlp_features_extraction/input/csv_to_json.py urlsModule/nlp_features/nlp_features_extraction/input/urls_train_set_20.csv", "Étape 1: Conversion CSV → JSON")

    # Étape 2 : Feature extraction
    run_step("py urlsModule/nlp_features/nlp_features_extraction/src/train.py urls_train_set_20_phishing.json phish urls_train_set_20_legitimate.json legitimate", "Étape 2: Extraction des features")

    # Étape 3 : TXT to CSV
    run_step("py urlsModule/nlp_features/nlp_features_extraction/output/features/txt_to_csv.py", "Étape 3: Conversion TXT → CSV")

    print("🏁 Pipeline terminée avec succès !")
