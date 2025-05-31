import pandas as pd
from sklearn.metrics import (
    classification_report, accuracy_score, precision_score,
    recall_score, f1_score, average_precision_score
)
from sklearn.model_selection import StratifiedKFold
import shap
import xgboost as xgb
import matplotlib.pyplot as plt
import numpy as np

# 🔄 1. Charger & fusionner les jeux de données
df1 = pd.read_csv("enron_spamassassin_nazario_nigerian_best_compo/extracted_urls/combined_train_set_features_filtered.csv")
df2 = pd.read_csv("enron_spamassassin_nazario_nigerian_best_compo/extracted_urls/urls_train_set_20_features_filtered.csv")
df3 = pd.read_csv("enron_spamassassin_nazario_nigerian_best_compo/extracted_urls/urls_test_set_features_filtered.csv")
#df4 = pd.read_csv("urlsModule/nlp_features/nlp_features_ebu_numerique.csv")

full_df = pd.concat([df1, df2, df3], ignore_index=True)
#full_df = df4

# 3. Supprimer les doublons d'URL
full_df = full_df[full_df['url'] != '-1'].copy()
full_df.drop_duplicates(subset='url', inplace=True)

X = full_df.drop(columns=['label', 'id', 'url'], errors='ignore')
y = full_df['label'].astype(int)

# 🔁 2. Cross-validation
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

# Stockage global
all_preds = []
all_probas = []
all_true = []
all_ids = []

# Pour les moyennes
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

    model = xgb.XGBClassifier(
        n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42,
        eval_metric="logloss"
    )
    model.fit(X_train_fold, y_train_fold)

    y_pred = model.predict(X_test_fold)
    y_proba = model.predict_proba(X_test_fold)[:, 1]

    acc = accuracy_score(y_test_fold, y_pred)
    prec = precision_score(y_test_fold, y_pred)
    rec = recall_score(y_test_fold, y_pred)
    f1 = f1_score(y_test_fold, y_pred)
    auprc = average_precision_score(y_test_fold, y_proba)

    print(f"✅ Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}, AUPRC: {auprc:.4f}")

    # Stocker
    accuracy_scores.append(acc)
    precision_scores.append(prec)
    recall_scores.append(rec)
    f1_scores.append(f1)
    auprc_scores.append(auprc)

    all_preds.extend(y_pred)
    all_probas.extend(y_proba)
    all_true.extend(y_test_fold)
    all_ids.extend(ids_test_fold)

# 🧾 3. Créer un fichier avec résultats finaux
results_df = pd.DataFrame({
    "id": all_ids,
    "phishing_probability": all_probas,
    "predicted_label": all_preds,
    "true_label": all_true
})
results_df.to_csv("urlsModule/nlp_features/xgb_cv_results.csv", index=False)

# 📊 4. Moyennes globales
print("\n📊 Résumé global des scores (moyenne ± std sur 10 folds):")
print(f"✅ Accuracy       : {np.mean(accuracy_scores):.4f} ± {np.std(accuracy_scores):.4f}")
print(f"✅ Precision      : {np.mean(precision_scores):.4f} ± {np.std(precision_scores):.4f}")
print(f"✅ Recall         : {np.mean(recall_scores):.4f} ± {np.std(recall_scores):.4f}")
print(f"✅ F1-score       : {np.mean(f1_scores):.4f} ± {np.std(f1_scores):.4f}")
print(f"🎯 AUPRC          : {np.mean(auprc_scores):.4f} ± {np.std(auprc_scores):.4f}")
print("✅ Résultats sauvegardés dans xgb_cv_results.csv")

# 🔍 5. SHAP sur tout le dataset (dernière version du modèle utilisé)
print("\n🔎 SHAP analysis (last model):")
explainer = shap.Explainer(model, X)
shap_values = explainer(X)

shap.summary_plot(shap_values, X)
shap.plots.bar(shap_values.mean(0), max_display=30)