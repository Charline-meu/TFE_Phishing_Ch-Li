import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import average_precision_score

# 🔄 1. Charger le dataset des 60% balanced
df = pd.read_csv("enron_spamassassin_nazario_nigerian_best_compo/extracted_urls/combined_train_set_features_filtered.csv")

# 🧹 2. Nettoyage
df = df[df['url'] != '-1'].copy()
df.drop_duplicates(subset='url', inplace=True)

# 🎯 3. Séparer features et labels
X = df.drop(columns=['label', 'id', 'url'], errors='ignore')
y = df['label'].astype(int)

# 🔍 4. Définir la grille d'hyperparamètres pour Random Forest
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [5, 10, 15],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2', None]
}

# 🧠 5. Modèle RandomForest
rf_clf = RandomForestClassifier(
    random_state=42,
    n_jobs=-1
)

# 🔁 6. Grid Search avec validation croisée
grid_search = GridSearchCV(
    estimator=rf_clf,
    param_grid=param_grid,
    scoring='average_precision',  # AUPRC
    cv=5,
    verbose=2,
    n_jobs=-1
)

# 🚀 7. Lancer le grid search
print("\n🔍 Grid Search en cours...")
grid_search.fit(X, y)

# ✅ 8. Résultats
print("\n✅ Meilleurs paramètres trouvés :")
print(grid_search.best_params_)

print(f"\n🎯 Meilleur score AUPRC (moyenne CV) : {grid_search.best_score_:.4f}")

# 💾 9. Récupérer le meilleur modèle (pas besoin de save_model pour RF)
best_model = grid_search.best_estimator_
