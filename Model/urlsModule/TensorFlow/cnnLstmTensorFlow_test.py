from sklearn.model_selection import KFold
from sklearn.metrics import (
    classification_report, accuracy_score, precision_score,
    recall_score, f1_score, average_precision_score
)
import tensorflow as tf
from keras import layers, models, optimizers
import numpy as np
import pandas as pd

# -------- 1. Fonction d'encodage URL --------
def encode_url(url, max_len=200):
    encoded = [ord(char) for char in url if ord(char) < 128]
    return np.pad(encoded[:max_len], (0, max(0, max_len - len(encoded))), 'constant')

# -------- 2. Charger les données --------
import pandas as pd

# 1. Charger les datasets
df1 = pd.read_csv('enron_spamassassin_nazario_nigerian_best_compo/extracted_urls/urls_train_set_40.csv')
df2 = pd.read_csv('datasetURLs/ebubekirbbr.csv')
df3 = pd.read_csv('enron_spamassassin_nazario_nigerian_best_compo/extracted_urls/urls_test_set.csv')
df4 = pd.read_csv('enron_spamassassin_nazario_nigerian_best_compo/extracted_urls/urls_train_set_20.csv')

# 2. Concaténer tous les datasets
full_df = pd.concat([df1, df2, df3, df4], ignore_index=True)

# 3. Supprimer les doublons d'URL
full_df.drop_duplicates(subset='url', inplace=True)

# 4. Équilibrer les classes
phishing_df = full_df[full_df['label'] == 1]
legit_df = full_df[full_df['label'] == 0]

min_len = min(len(phishing_df), len(legit_df))

phishing_sample = phishing_df.sample(n=min_len, random_state=42)
legit_sample = legit_df.sample(n=min_len, random_state=42)

balanced_df = pd.concat([phishing_sample, legit_sample], ignore_index=True)

# 5. Mélanger aléatoirement les données
balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)

# 6. Sauvegarder le résultat
balanced_df.to_csv('datasetURLs/urls_all_balanced_cross_val.csv', index=False)
print("✅ Dataset combiné, dédoublonné et équilibré sauvegardé sous 'urls_all_balanced.csv'")

print("Avant suppression :", len(df1) + len(df2) + len(df3) + len(df4))
print("Après suppression des doublons :", len(full_df))


urls = balanced_df['url'].astype(str).str.lower().values
labels = balanced_df['label'].values
X = np.array([encode_url(url) for url in urls])
y = np.array(labels)

# -------- 3. Setup K-Fold --------
kf = KFold(n_splits=10, shuffle=True, random_state=42)
fold = 1

# Stocker les scores pour chaque fold
results = {
    "accuracy": [],
    "precision": [],
    "recall": [],
    "f1": [],
    "auprc": []
}

# -------- 4. Boucle K-Fold --------
for train_idx, test_idx in kf.split(X):
    print(f"\n📂 Fold {fold}")

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # Définir un nouveau modèle à chaque fold
    model = models.Sequential([
        layers.Embedding(input_dim=128, output_dim=128, input_length=200),  # Embedding Layer
        layers.Conv1D(filters=256, kernel_size=4, activation='relu', padding='same'),
        layers.MaxPooling1D(pool_size=4),
        layers.LSTM(units=512, return_sequences=False),
        layers.Dropout(0.1),
        layers.Dense(units=256, activation='relu'),
        layers.Dense(units=1, activation='sigmoid')  # Output en probabilité
    ])

    model.compile(
        loss=tf.keras.losses.BinaryCrossentropy(),
        optimizer=optimizers.Adam(learning_rate=3e-4),
        metrics=['accuracy',tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall'),
        tf.keras.metrics.AUC(name='AUC', curve='PR')]
    )

    model.fit(X_train, y_train, epochs=4, batch_size=32, verbose=0)

    # Prédictions
    y_pred_proba = model.predict(X_test).flatten()
    y_pred = (y_pred_proba > 0.5).astype(int)

    # Calcul des métriques
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auprc = average_precision_score(y_test, y_pred_proba)

    # Sauvegarde des résultats
    results["accuracy"].append(acc)
    results["precision"].append(prec)
    results["recall"].append(rec)
    results["f1"].append(f1)
    results["auprc"].append(auprc)

    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"AUPRC:     {auprc:.4f}")

    fold += 1

# -------- 5. Résumé final --------
print("\n📊 Moyennes sur les 10 folds :")
print(f"Accuracy:  {np.mean(results['accuracy']):.4f}")
print(f"Precision: {np.mean(results['precision']):.4f}")
print(f"Recall:    {np.mean(results['recall']):.4f}")
print(f"F1 Score:  {np.mean(results['f1']):.4f}")
print(f"AUPRC:     {np.mean(results['auprc']):.4f}")
