import pandas as pd
import numpy as np
import os
from pathlib import Path
import subprocess
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, average_precision_score
from sklearn.utils import resample
import shap
import matplotlib.pyplot as plt

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

    cleaned_path = Path(input_path)
    no_url_path = cleaned_path.with_name(f"no_url_{cleaned_path.name}")

    df_with_url.to_csv(cleaned_path, index=False)
    df_no_url.to_csv(no_url_path, index=False)

    print(f"✅ {len(df_no_url)} emails sans URL exclus de : {input_path}")
    return cleaned_path, no_url_path

# 🔹 Paths
train_path = "enron_spamassassin_nazario_nigerian_best_compo/extracted_urls/urls_train_set_60.csv"
test1_path = "enron_spamassassin_nazario_nigerian_best_compo/extracted_urls/urls_train_set_20.csv"
test2_path = "enron_spamassassin_nazario_nigerian_best_compo/extracted_urls/urls_test_set.csv"
train_path_ebu = "datasetURLs/ebubekirbbr.csv"

# 🔹 Fusion et équilibrage
df_main = pd.read_csv(train_path)
df_ebu = pd.read_csv(train_path_ebu)
df_combined = pd.concat([df_main, df_ebu], ignore_index=True)
print("🔍 Composition initiale :")
print(df_combined['label'].value_counts())
print(f"Total emails before deduplication: {len(df_combined)}")

if 'url' in df_combined.columns:
    df_combined = df_combined.drop_duplicates(subset='url')
print(f"Total emails after deduplication: {len(df_combined)}")

# 🔹 Équilibrage
phishing = df_combined[df_combined['label'] == 1]
legitimate = df_combined[df_combined['label'] == 0]
min_size = min(len(phishing), len(legitimate))
phishing_bal = resample(phishing, replace=False, n_samples=min_size, random_state=42)
legitimate_bal = resample(legitimate, replace=False, n_samples=min_size, random_state=42)
df_combined = pd.concat([phishing_bal, legitimate_bal]).sample(frac=1, random_state=42).reset_index(drop=True)
print("🎯 Composition après équilibrage :")
print(df_combined['label'].value_counts())

# 🔹 Sauvegarde jeu combiné
output_dir = Path(train_path).parent
output_path = output_dir / "combined_train_set.csv"
df_combined.to_csv(output_path, index=False)

# 🔹 Nettoyage test sets
train_path_combined = str(output_path)
test1_path_cleaned, test1_no_url_path = remove_no_url_entries(test1_path)
test2_path_cleaned, test2_no_url_path = remove_no_url_entries(test2_path)

# 🔹 Charger les features extraits
train_df = pd.read_csv("enron_spamassassin_nazario_nigerian_best_compo/extracted_urls/combined_train_set_features_filtered.csv")
test_df_1 = pd.read_csv("enron_spamassassin_nazario_nigerian_best_compo/extracted_urls/urls_train_set_20_features_filtered.csv")
test_df_2 = pd.read_csv("enron_spamassassin_nazario_nigerian_best_compo/extracted_urls/urls_test_set_features_filtered.csv")

X_train = train_df.drop(columns=['label', 'id','url'], errors='ignore')
y_train = train_df['label']

# 🔹 Entraîner Random Forest
rf_model = RandomForestClassifier(
    n_estimators=300,
    max_depth=15,
    max_features='sqrt',
    min_samples_leaf=1,
    min_samples_split=2,
    random_state=42,
    n_jobs=-1
)
rf_model.fit(X_train, y_train)
print("✅ Random Forest model trained.")

# 🔹 Fonction d'évaluation
def process_test_set(test_df, test_name, no_url_path=None):
    X_test = test_df.drop(columns=['label', 'id', 'url'], errors='ignore')
    y_pred = rf_model.predict(X_test)
    y_proba = rf_model.predict_proba(X_test)[:, 1]

    auprc = average_precision_score(test_df['label'], y_proba)
    print(f"AUPRC for {test_name}: {auprc:.4f}")

    test_df["phishing_probability"] = y_proba

    if no_url_path and Path(no_url_path).exists():
        no_url_df = pd.read_csv(no_url_path)
        no_url_df["phishing_probability"] = -1
        if 'id' not in no_url_df.columns:
            no_url_df["id"] = range(len(no_url_df))
        if 'label' not in no_url_df.columns:
            no_url_df["label"] = -1
        no_url_df = no_url_df[['id', 'phishing_probability', 'label', 'url']]
    else:
        no_url_df = pd.DataFrame(columns=['id', 'phishing_probability', 'label', 'url'])

    full_df = pd.concat([test_df, no_url_df], ignore_index=True)
    full_df_sorted = full_df.sort_values(by='phishing_probability', ascending=False)
    output_df = full_df_sorted.groupby('id', as_index=False).first()[['id', 'phishing_probability', 'label', 'url']]
    output_df = output_df.sort_values(by="id")

    output_filename = f"enron_spamassassin_nazario_nigerian_best_compo/extracted_urls/nlp_proba_{test_name}.csv"
    output_df.to_csv(output_filename, index=False)
    print(f"✅ Fichier CSV généré : {output_filename}")

    print(f"📊 Rapport de classification pour {test_name} (emails avec URL uniquement) :")
    print(classification_report(test_df['label'], y_pred))

    return X_test

# 🔹 Appliquer aux jeux de test
X_test_1 = process_test_set(test_df_1, "train_set_20", no_url_path=test1_no_url_path)
X_test_2 = process_test_set(test_df_2, "test_set", no_url_path=test2_no_url_path)
X_test_all = process_test_set(
    pd.concat([
        test_df_1.drop(columns=['phishing_probability'], errors='ignore'),
        test_df_2.drop(columns=['phishing_probability'], errors='ignore')
    ], ignore_index=True),
    "test_all"
)

# 🔍 SHAP analysis
print("ici")
explainer = shap.Explainer(rf_model, X_train)
X_shap = X_test_all.drop(columns=['id', 'url'], errors='ignore').sample(n=100, random_state=42)
shap_values = explainer(X_shap)

shap_vals_rf = shap_values.values[:, :, 1]
mean_shap_rf = shap_vals_rf.mean(axis=0)

# ✅ Moyenne signée des SHAP values (classe phishing)
mean_signed_shap = pd.DataFrame({
    'feature': X_shap.columns,
    'mean_shap': shap_vals_rf.mean(axis=0)
})

# 🔁 Tri par valeur absolue décroissante (mais on garde le signe)
mean_signed_shap['abs_val'] = mean_signed_shap['mean_shap'].abs()
mean_signed_shap = mean_signed_shap.sort_values(by='abs_val', ascending=False).drop(columns='abs_val')

# 🎨 Couleurs selon le signe
colors = ['#1E88E5' if val < 0 else '#D81B60' for val in mean_signed_shap['mean_shap']]

# 📊 Plot stylé trié par importance absolue
plt.figure(figsize=(12, 8))
bars = plt.barh(mean_signed_shap['feature'], mean_signed_shap['mean_shap'], color=colors)
plt.axvline(0, color='black', linestyle='--')
plt.xlabel("Mean SHAP Value (signed)")
plt.title("Top Features Impacting Phishing Prediction (RF, sorted by abs importance)")

# 💬 Valeurs au bout des barres
for bar, val in zip(bars, mean_signed_shap['mean_shap']):
    plt.text(val + 0.002 if val > 0 else val - 0.005,
             bar.get_y() + bar.get_height() / 2,
             f"{val:+.2f}",
             va='center',
             ha='left' if val > 0 else 'right',
             fontsize=9)

plt.tight_layout()
plt.show()
