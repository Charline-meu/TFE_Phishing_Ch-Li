import pandas as pd
from sklearn.metrics import classification_report
import shap
import xgboost as xgb
import matplotlib.pyplot as plt
import json
from sklearn.metrics import average_precision_score


# **1. Charger les données d'entraînement et de test**
train_df = pd.read_csv("enron_spamassassin_nazario_nigerian_best_compo/extracted_features/features_train_set_60.csv")
train_perso_liza_df = pd.read_csv("email_perso_liza/features_perso_liza_without_pub.csv")
test_df_1 = pd.read_csv("enron_spamassassin_nazario_nigerian_best_compo/extracted_features/features_test_set.csv")
test_df_2 = pd.read_csv("enron_spamassassin_nazario_nigerian_best_compo/extracted_features/features_train_set_20.csv")

# Supposons que ton DataFrame s'appelle train_df et que la colonne du label s'appelle 'label'

# Trouver les lignes avec source == "enron"
enron_rows = train_df[train_df["source"] == "enron"]

# Tirer 10 lignes aléatoires parmi elles
to_remove = enron_rows.sample(n=1692, random_state=42)  # random_state pour reproductibilité

# Supprimer ces lignes du DataFrame
train_df_cleaned = train_df.drop(to_remove.index)

# On prélève 2741 lignes aléatoires parmi celles-cis
print(len(train_perso_liza_df))
train_df_perso_to_remove = train_perso_liza_df.sample(n=0, random_state=42)
# On enlève ces lignes du DataFrame principal
train_df_perso_reduced = train_perso_liza_df.drop(train_df_perso_to_remove.index)


# 2. Concaténer avec les emails perso
train_df = pd.concat([train_df_cleaned, train_df_perso_reduced], ignore_index=True)
train_df.to_csv("structureModule/final_train_dataset.csv", index=False)

# Compter les emails légitimes et phishing
nb_legit = (train_df['label'] == 0).sum()
nb_phish = (train_df['label'] == 1).sum()

# Affichage
print(f"✔️ Legitimate emails : {nb_legit}")
print(f"❗ Phishing emails   : {nb_phish}")
print(f"📊 Total             : {len(train_df)}")

# **2. Supprimer `index` et `email_id` du training set**
X_train = train_df.drop(columns=['label', 'index', 'email_id','source'], errors='ignore')
y_train = train_df['label']


# Sauvegarde des colonnes utilisées par le modèle
with open("structureModule/feature_names.json", "w") as f:
    json.dump(list(X_train.columns), f)

# **3. Entraîner XGBoost**
xgb_model = xgb.XGBClassifier(
    n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42, 
    eval_metric="logloss"
)
xgb_model.fit(X_train, y_train)
xgb_model.save_model("structureModule/xgboost_model_esnnpperso.json")
print("✅ Modèle sauvegardé dans 'models/xgboost_model_esnn.json'")

# **4. Vérifier et ajuster les colonnes du test set**
def check_and_fix_columns(train_features, test_df, test_name):
    """
    Vérifie les colonnes du jeu de test et ajuste en fonction du jeu d'entraînement.
    """
    # Sauvegarder `index` et `email_id` avant de supprimer pour l'entraînement
    email_ids_test = test_df['email_id'] if 'email_id' in test_df.columns else pd.Series(range(len(test_df)))
    indices_test = test_df['index'] if 'index' in test_df.columns else pd.Series(range(len(test_df)))

    # Supprimer `index`, `email_id` et `label` du test set
    X_test = test_df.drop(columns=['label', 'index', 'email_id','source'], errors='ignore')

    # Vérifier les colonnes manquantes et supplémentaires
    train_columns = set(train_features.columns)
    test_columns = set(X_test.columns)

    missing_cols = train_columns - test_columns  # Colonnes manquantes dans test
    extra_cols = test_columns - train_columns  # Colonnes en trop dans test

    print(f"\n🔍 Vérification des colonnes pour {test_name}:")
    if missing_cols:
        print(f"❌ Colonnes présentes dans TRAIN mais absentes dans {test_name}: {list(missing_cols)}")
        for col in missing_cols:
            X_test[col] = 0  # Remplir les colonnes manquantes avec 0

    if extra_cols:
        print(f"⚠️ Colonnes présentes dans {test_name} mais absentes dans TRAIN: {list(extra_cols)}")
        X_test = X_test[train_features.columns]  # Supprimer les colonnes en trop et réordonner

    # Réordonner selon `X_train`
    X_test = X_test[train_features.columns]

    return X_test, email_ids_test, indices_test

# **5. Prédire et sauvegarder les résultats pour chaque jeu de test**
def process_test_set(test_df, test_name):
    # Vérification et correction des colonnes
    X_test, email_ids_test, indices_test = check_and_fix_columns(X_train, test_df, test_name)

    # Prédiction
    y_pred = xgb_model.predict(X_test)
    y_pred_proba = xgb_model.predict_proba(X_test)[:, 1]

    # Création du DataFrame de sortie
    output_df = pd.DataFrame({
        "index": indices_test.values,
        "email_id": email_ids_test.values,
        "structure_proba": y_pred_proba,
        "label": test_df['label'].values  # Vraie classe
    })

    # Trier et sauvegarder
    output_df = output_df.sort_values(by="index")
    output_filename = f"enron_spamassassin_nazario_nigerian_best_compo/extracted_features/features_proba_{test_name}.csv"
    output_df.to_csv(output_filename, index=False)

    #print(f"✅ Fichier CSV généré : {output_filename}")

    # Rapport de classification
    print(f"Rapport de classification pour {test_name} :")
    print(classification_report(test_df['label'], y_pred))

     # AUPRC
    auprc = average_precision_score(test_df['label'], y_pred_proba)
    print(f"AUPRC (Average Precision Score) pour {test_name} : {auprc:.4f}")

    return X_test  # Retourner X_test pour SHAP

# **5. Appliquer aux deux jeux de test**
X_test_1 = process_test_set(test_df_1, "test_set")
X_test_2 = process_test_set(test_df_2, "train_set_20")

#pour avoir 1 tableau de performances
X_test_all = process_test_set(pd.concat([test_df_1,test_df_2], ignore_index=True), "test_all")

# **6. Analyse avec SHAP (sur le premier jeu de test uniquement)**
explainer = shap.Explainer(xgb_model, X_train.drop(columns=['index', 'email_id'], errors='ignore'))
shap_values = explainer(X_test_all.drop(columns=['index', 'email_id'], errors='ignore'))

# **7. Graphiques SHAP**
shap.summary_plot(shap_values, X_test_all.drop(columns=['index', 'email_id'], errors='ignore'))
shap.plots.bar(shap_values.mean(0), max_display=30)
