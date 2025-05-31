import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import shap
import xgboost as xgb 
import matplotlib.pyplot as plt

# **1. Charger les données**
df = pd.read_csv("enron_nazario/features_merged_enron_nazario.csv")

# **2. Séparer les features et la cible**
X = df.drop(columns=['label'], errors='ignore')  # Supprimer la colonne cible
y = df['label']  # Variable cible (0 = légitime, 1 = phishing)

# Sauvegarder `email_id`, `index` et `label` avant de les retirer de X
email_ids = df['id_email'] if 'id_email' in df.columns else pd.Series(range(len(df)))
indices = df['index'] if 'index' in df.columns else pd.Series(range(len(df)))
labels = y  # Conserver le label pour plus tard

# **3. Séparer en train/test**
X_train, X_test, y_train, y_test, email_ids_train, email_ids_test, indices_train, indices_test, labels_train, labels_test = train_test_split(
    X, y, email_ids, indices, labels, test_size=0.2, random_state=42
)

# **4. Entraîner XGBoost**
xgb_model = xgb.XGBClassifier(
    n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42, 
    use_label_encoder=False, eval_metric="logloss"
)
xgb_model.fit(X_train.drop(columns=['index', 'id_email'], errors='ignore'), y_train)

# **5. Prédire la probabilité de phishing**
y_pred = xgb_model.predict(X_test.drop(columns=['index', 'id_email'], errors='ignore'))
y_pred_proba = xgb_model.predict_proba(X_test.drop(columns=['index', 'id_email'], errors='ignore'))[:, 1]

# **6. Création du DataFrame final**
output_df = pd.DataFrame({
    "index": indices_test.values,
    "id_email": email_ids_test.values,
    "phishing_probability": y_pred_proba,
    "label": labels_test.values  # Ajouter la vraie classe de l'email
})

# **7. Trier le DataFrame par `index`**
output_df = output_df.sort_values(by="index")

# **8. Sauvegarde en CSV**
output_df.to_csv("email_phishing_probabilities.csv", index=False)

print("✅ Fichier CSV généré : email_phishing_probabilities.csv")

# **9. Rapport de Classification**
print("Rapport de classification :")
print(classification_report(y_test, y_pred))

# **10. Analyse avec SHAP**
explainer = shap.Explainer(xgb_model, X_train.drop(columns=['index', 'id_email'], errors='ignore'))
shap_values = explainer(X_test.drop(columns=['index', 'id_email'], errors='ignore'))

# **11. Graphiques SHAP**
shap.summary_plot(shap_values, X_test.drop(columns=['index', 'id_email'], errors='ignore'))
shap.plots.bar(shap_values.mean(0), max_display=30)
