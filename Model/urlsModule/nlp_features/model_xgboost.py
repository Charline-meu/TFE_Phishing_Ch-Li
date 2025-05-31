import pandas as pd
from sklearn.metrics import classification_report
import shap
import xgboost as xgb
import matplotlib.pyplot as plt
from pathlib import Path
import subprocess
from sklearn.metrics import average_precision_score
import os
from sklearn.utils import resample


# 🔹 NLP pipeline
def run_nlp_pipeline(csv_path):
    print("🚀 Lancement de la pipeline NLP...")
    csv_path = Path(csv_path)
    if csv_path.suffix == '.csv':
        csv_path = csv_path.with_suffix('')
    csv_base = str(csv_path)
    base_name = os.path.splitext(os.path.basename(csv_path))[0]
    subprocess.run(f"py urlsModule/nlp_features/nlp_features_extraction/input/csv_to_json.py {csv_base}.csv", check=True, shell=True)
    subprocess.run(f"py urlsModule/nlp_features/nlp_features_extraction/src/train.py {base_name}_phishing.json phish {base_name}_legitimate.json legitimate", check=True, shell=True)
    subprocess.run(f"py urlsModule/nlp_features/nlp_features_extraction/output/features/txt_to_csv.py {csv_base}_features.csv", check=True, shell=True)
    subprocess.run(f"py numerique.py {csv_base}_features.csv", check=True, shell=True)
    print("✅ Pipeline terminée.")

def remove_no_url_entries(input_path):
    df = pd.read_csv(input_path)
    df_with_url = df[df['url'] != '-1'].copy()
    df_no_url = df[df['url'] == '-1'].copy()

    # Sauvegarde fichiers
    cleaned_path = Path(input_path)
    no_url_path = cleaned_path.with_name(f"no_url_{cleaned_path.name}")

    df_with_url.to_csv(cleaned_path, index=False)
    df_no_url.to_csv(no_url_path, index=False)

    print(f"✅ {len(df_no_url)} emails sans URL exclus de : {input_path}")
    return cleaned_path, no_url_path



train_path = "enron_spamassassin_nazario_nigerian_best_compo/extracted_urls/urls_train_set_60.csv"
test1_path = "enron_spamassassin_nazario_nigerian_best_compo/extracted_urls/urls_train_set_20.csv"
test2_path = "enron_spamassassin_nazario_nigerian_best_compo/extracted_urls/urls_test_set.csv"
train_path_ebu = "datasetURLs/ebubekirbbr.csv"

# Charger les deux CSV
df_main = pd.read_csv(train_path)
df_ebu = pd.read_csv(train_path_ebu)

# Concaténer les deux datasets
df_combined = pd.concat([df_main, df_ebu], ignore_index=True)
initial_len = len(df_combined)

# 🔹 Composition initiale
print("🔍 Composition initiale :")
print(df_combined['label'].value_counts())
print(f"Total emails before deduplication: {len(df_combined)}")

if 'url' in df_combined.columns:
    #enleve les emails sans urls aussi
    df_combined = df_combined.drop_duplicates(subset='url')
deduplicated_len = len(df_combined)
print(f"🧹 {initial_len - deduplicated_len} duplicates removed.")
print(f"Total emails after deduplication: {deduplicated_len}")

# 🔹 Équilibrage des classes
if 'label' in df_combined.columns:
    phishing = df_combined[df_combined['label'] == 1]
    legitimate = df_combined[df_combined['label'] == 0]

    min_size = min(len(phishing), len(legitimate))

    phishing_balanced = resample(phishing, replace=False, n_samples=min_size, random_state=42)
    legitimate_balanced = resample(legitimate, replace=False, n_samples=min_size, random_state=42)

    df_combined = pd.concat([phishing_balanced, legitimate_balanced]).sample(frac=1, random_state=42).reset_index(drop=True)

    print("🎯 Composition après équilibrage :")
    print(df_combined['label'].value_counts())
    print(f"Total emails after balancing: {len(df_combined)}")

# Définir le dossier de sortie à partir de train_path
output_dir = Path(train_path).parent
output_path = output_dir / "combined_train_set.csv"

# Sauvegarder dans le dossier souhaité
df_combined.to_csv(output_path, index=False)

train_path_combined = "enron_spamassassin_nazario_nigerian_best_compo/extracted_urls/combined_train_set.csv"
test1_path_cleaned, test1_no_url_path = remove_no_url_entries(test1_path)
test2_path_cleaned, test2_no_url_path = remove_no_url_entries(test2_path)
""" 
run_nlp_pipeline(train_path_combined)
run_nlp_pipeline(test1_path_cleaned)
run_nlp_pipeline(test2_path_cleaned)
"""

train_df = pd.read_csv("enron_spamassassin_nazario_nigerian_best_compo/extracted_urls/combined_train_set_features_filtered.csv")
test_df_1 = pd.read_csv("enron_spamassassin_nazario_nigerian_best_compo/extracted_urls/urls_train_set_20_features_filtered.csv")
test_df_2 = pd.read_csv("enron_spamassassin_nazario_nigerian_best_compo/extracted_urls/urls_test_set_features_filtered.csv")


X_train = train_df.drop(columns=['label', 'id','url'], errors='ignore')
y_train = train_df['label']

# **3. Entraîner XGBoost**
xgb_model = xgb.XGBClassifier(
    n_estimators=300, max_depth=7, learning_rate=0.2, random_state=42, colsample_bytree = 0.8, subsample = 1.0,
    eval_metric="logloss"
)
xgb_model.fit(X_train, y_train)
xgb_model.save_model("urlsModule/nlp_features/xgboost_model_nlp_features.json")
print("✅ Modèle sauvegardé dans 'urlsModule/nlp_features/xgboost_model_nlp_features.json'")

def process_test_set(test_df, test_name, no_url_path=None):
    """
    Prédit sur les emails avec URL (déjà filtrés en amont)
    Réintègre les emails sans URL à partir du fichier no_url_path
    """
    # 🔹 Prédiction sur les emails AVEC URL
    X_test = test_df.drop(columns=['label', 'id', 'url'], errors='ignore')
    #indices_test = test_df['id'] if 'id' in test_df.columns else pd.Series(range(len(test_df)))
    y_pred = xgb_model.predict(X_test)
    y_pred_proba = xgb_model.predict_proba(X_test)[:, 1]

    auprc = average_precision_score(test_df['label'], y_pred_proba)
    print(f"AUPRC for {test_name}: {auprc:.4f}")

    test_df["url_proba"] = y_pred_proba

    # 🔹 Réintégrer les emails SANS URL
    if no_url_path and Path(no_url_path).exists():
        no_url_df = pd.read_csv(no_url_path)
        no_url_df["url_proba"] = -1
        if 'id' not in no_url_df.columns:
            no_url_df["id"] = range(len(no_url_df))
        if 'label' not in no_url_df.columns:
            no_url_df["label"] = -1
        no_url_df = no_url_df[['id', 'url_proba', 'label', 'url']]
    else:
        no_url_df = pd.DataFrame(columns=['id', 'url_proba', 'label', 'url'])

    # 🔹 Fusion des prédictions et des emails sans URL
    full_df = pd.concat([test_df, no_url_df], ignore_index=True)

    # 🔹 Sélection de la proba max par email
    full_df_sorted = full_df.sort_values(by='url_proba', ascending=False)
    output_df = full_df_sorted.groupby('id', as_index=False).first()[['id', 'url_proba', 'label', 'url']]
    output_df = output_df.sort_values(by="id")

    # 🔹 Sauvegarde
    output_filename = f"enron_spamassassin_nazario_nigerian_best_compo/extracted_urls/nlp_proba_{test_name}.csv"
    output_df.to_csv(output_filename, index=False)
    print(f"✅ Fichier CSV généré : {output_filename}")

    # 🔹 Évaluation uniquement sur emails avec URL
    print(f"📊 Rapport de classification pour {test_name} (emails avec URL uniquement) :")
    print(classification_report(test_df['label'], y_pred))

    return X_test



# **5. Appliquer aux deux jeux de test**
X_test_1 = process_test_set(test_df_1, "train_set_20", no_url_path=test1_no_url_path)
X_test_2 = process_test_set(test_df_2, "test_set", no_url_path=test2_no_url_path)
X_test_all = process_test_set(pd.concat([test_df_1.drop(columns=['url_proba']), test_df_2.drop(columns=['url_proba'])], ignore_index=True), "test_all")
# Sauvegarder le modèle
xgb_model.save_model("urlsModule/nlp_features/xgb_model_nlp_features.json")


# **6. Analyse avec SHAP (sur le premier jeu de test uniquement)**
explainer = shap.Explainer(xgb_model, X_train.drop(columns=['id','url'], errors='ignore'))
shap_values = explainer(X_test_all.drop(columns=['id','url'], errors='ignore'))

# **7. Graphiques SHAP**
shap.summary_plot(shap_values, X_test_all.drop(columns=['id','url'], errors='ignore'))
shap.plots.bar(shap_values.mean(0), max_display=30)
