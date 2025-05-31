import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, accuracy_score, precision_score,
    recall_score, f1_score, average_precision_score
)
from sklearn.model_selection import StratifiedKFold

# 🔄 1. Charger & fusionner les jeux de données
df1 = pd.read_csv("enron_spamassassin_nazario_nigerian_best_compo/extracted_urls/combined_train_set_features_filtered.csv")
df2 = pd.read_csv("enron_spamassassin_nazario_nigerian_best_compo/extracted_urls/urls_train_set_20_features_filtered.csv")
df3 = pd.read_csv("enron_spamassassin_nazario_nigerian_best_compo/extracted_urls/urls_test_set_features_filtered.csv")

full_df = pd.concat([df1, df2, df3], ignore_index=True)

# 🔍 2. Nettoyage
full_df = full_df[full_df['url'] != '-1'].copy()
full_df.drop_duplicates(subset='url', inplace=True)

# 🎯 3. Préparation des features et labels
X = full_df.drop(columns=['label', 'id', 'url'], errors='ignore')
y = full_df['label'].astype(int)

# 🔁 4. Cross-validation
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

# Stockage global
all_preds = []
all_probas = []
all_true = []
all_ids = []

# Moyennes
accuracy_scores = []
precision_scores = []
recall_scores = []
f1_scores = []
auprc_scores = []

for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
    print(f"\n🔄 Fold {fold}/10")

    X_train_fold, X_test_fold = X.iloc[train_idx], X.iloc[test_idx]
    y_train_fold, y_test_fold = y.iloc[train_idx], y.iloc[test_idx]
    ids_test_fold = full_df.iloc[test_idx]['id'].values

    # 🌲 Random Forest
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train_fold, y_train_fold)

    # Prédiction
    y_pred = model.predict(X_test_fold)
    y_proba = model.predict_proba(X_test_fold)[:, 1]

    # Évaluation
    acc = accuracy_score(y_test_fold, y_pred)
    prec = precision_score(y_test_fold, y_pred)
    rec = recall_score(y_test_fold, y_pred)
    f1 = f1_score(y_test_fold, y_pred)
    auprc = average_precision_score(y_test_fold, y_proba)

    print(f"✅ Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}, AUPRC: {auprc:.4f}")

    # Stockage
    accuracy_scores.append(acc)
    precision_scores.append(prec)
    recall_scores.append(rec)
    f1_scores.append(f1)
    auprc_scores.append(auprc)

    all_preds.extend(y_pred)
    all_probas.extend(y_proba)
    all_true.extend(y_test_fold)
    all_ids.extend(ids_test_fold)

# 💾 5. Sauvegarde des résultats
results_df = pd.DataFrame({
    "id": all_ids,
    "phishing_probability": all_probas,
    "predicted_label": all_preds,
    "true_label": all_true
})
results_df.to_csv("urlsModule/nlp_features/rf_cv_results.csv", index=False)

# 📊 6. Résumé global
print("\n📊 Résumé global des scores (moyenne ± std sur 10 folds):")
print(f"✅ Accuracy       : {np.mean(accuracy_scores):.4f} ± {np.std(accuracy_scores):.4f}")
print(f"✅ Precision      : {np.mean(precision_scores):.4f} ± {np.std(precision_scores):.4f}")
print(f"✅ Recall         : {np.mean(recall_scores):.4f} ± {np.std(recall_scores):.4f}")
print(f"✅ F1-score       : {np.mean(f1_scores):.4f} ± {np.std(f1_scores):.4f}")
print(f"🎯 AUPRC          : {np.mean(auprc_scores):.4f} ± {np.std(auprc_scores):.4f}")
print("✅ Résultats sauvegardés dans rf_cv_results.csv")

# 🔍 7. SHAP Analysis
print("\n🔎 SHAP analysis (last model):")
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)

# Résumé global (classe 1 = phishing)
shap.summary_plot(shap_values[1], X)
shap.plots.bar(shap.Explanation(values=np.abs(shap_values[1]).mean(0), feature_names=X.columns), max_display=30)
