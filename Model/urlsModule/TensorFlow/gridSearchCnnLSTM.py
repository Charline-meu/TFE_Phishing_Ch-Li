# =====================
# 🔧 IMPORTS & FONCTIONS
# =====================

from sklearn.model_selection import ParameterGrid
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, average_precision_score, classification_report
)
import pandas as pd
import numpy as np
import tensorflow as tf
from keras import layers, models, optimizers
import time

# =====================
# 📥 CHARGEMENT DES DONNÉES
# =====================

train_path = 'urlsModule/Tensorflow/balanced_train_data.csv'
test_path = 'enron_spamassassin_nazario_nigerian_best_compo/extracted_urls/urls_test_set.csv'  # 🔁 CHANGE LE NOM ICI

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

def encode_url(url, max_len=200):
    encoded = [ord(char) for char in str(url) if ord(char) < 128]
    return np.pad(encoded[:max_len], (0, max(0, max_len - len(encoded))), 'constant')

def preprocess_data(df):
    df['url'] = df['url'].astype(str).str.lower()
    X = np.array([encode_url(url) for url in df['url']])
    y = df['label'].values
    metadata = df[['index']].copy() if 'index' in df.columns else None
    return X, y, metadata

X_train, y_train, _ = preprocess_data(train_df)
X_test, y_test, _ = preprocess_data(test_df)

# =====================
# 🧠 DÉFINITION DU MODÈLE
# =====================

def build_model(filters=256, kernel_size=4, lstm_units=256, dropout=0.1,
                dense_units=64, learning_rate=1e-4, max_pool_size=4):
    model = models.Sequential([
        layers.Embedding(input_dim=128, output_dim=128, input_length=200),
        layers.Conv1D(filters=filters, kernel_size=kernel_size, activation='relu', padding='same'),
        layers.MaxPooling1D(pool_size=max_pool_size),
        layers.LSTM(units=lstm_units),
        layers.Dropout(dropout),
        layers.Dense(dense_units, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(
        loss='binary_crossentropy',
        optimizer=optimizers.Adam(learning_rate=learning_rate),
        metrics=['accuracy']
    )
    return model

# =====================
# 🧪 GRILLE D’HYPERPARAMÈTRES
# =====================

param_grid = {
    'dropout': [0.1],
    'lstm_units': [256],
    'dense_units': [64, 128],
    'learning_rate': [1e-4, 3e-4],
    'epochs': [3, 4, 5],
    'max_pool_size': [4, 8]
}

grid = list(ParameterGrid(param_grid))
results = []

# =====================
# 🔁 GRID SEARCH
# =====================

start_time = time.time()

for i, params in enumerate(grid):
    print(f"\n🔍 Test {i+1}/{len(grid)} — {params}")

    n_epochs = params.pop('epochs')
    
    # 1. Entraînement sur tout le jeu d'entraînement
    model = build_model(**params)
    model.fit(X_train, y_train, epochs=n_epochs, batch_size=32, verbose=0)

    # 2. Prédiction sur le jeu de test externe
    y_pred_proba = model.predict(X_test, verbose=0).flatten()
    y_pred = (y_pred_proba > 0.5).astype(int)

    # 3. Évaluation
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auprc = average_precision_score(y_test, y_pred_proba)

    print(f"✔️ AUPRC={auprc:.4f} | Acc={acc:.4f} | Prec={prec:.4f} | Recall={rec:.4f} | F1={f1:.4f}")

    results.append({
        **params,
        'epochs': n_epochs,
        'AUPRC': auprc,
        'Accuracy': acc,
        'Precision': prec,
        'Recall': rec,
        'F1-Score': f1
    })

# =====================
# 💾 RÉSULTATS FINAUX
# =====================

end_time = time.time()
execution_time = end_time - start_time
print(f"\n⏱️ Temps total : {execution_time:.2f} secondes")

df_results = pd.DataFrame(results)
df_results = df_results.sort_values(by='AUPRC', ascending=False)
df_results.to_csv("grid_search_d_fence_results.csv", index=False)

print("\n📄 Résultats enregistrés dans grid_search_d_fence_results.csv")
print("\n🏆 Meilleur modèle :")
print(df_results.iloc[0])
