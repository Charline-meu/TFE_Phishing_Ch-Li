# shap_hybrid_cnn_lstm.py

import torch
import numpy as np
import pandas as pd
import shap
from cnnLstmPyTorch import HybridCNNLSTM  # On importe le modèle

# ===============================
# 1. Charger le modèle et les données de test
# ===============================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Charger le modèle
model = HybridCNNLSTM(num_lexical_features=77).to(device)
model.load_state_dict(torch.load('hybrid_cnn_lstm_model.pth', map_location=device))
model.eval()

# Charger les données de test
X_test = np.load('X_test.npy')

# Conversion en Tensor pour PyTorch
X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)

# ===============================
# 2. Définir le prédicteur pour SHAP
# ===============================
def predict_fn(x):
    model.eval()
    with torch.no_grad():
        x_tensor = torch.tensor(x, dtype=torch.float32).to(device)
        outputs = model(x_tensor)
        return outputs.cpu().numpy()

# ===============================
# 3. Initialisation de SHAP
# ===============================
explainer = shap.KernelExplainer(predict_fn, X_test[:100])
shap_values = explainer.shap_values(X_test[:100])

# ===============================
# 4. Définition des noms des features
# ===============================
# Caractères encodés
feature_names = ['Char_' + str(i) for i in range(200)]

# Noms des features lexicales (issus des colonnes du CSV)
lexical_columns = list(pd.read_csv('datasetURLs/ebubekirbbr.csv').drop(['url', 'label'], axis=1).columns)
feature_names += lexical_columns

# ===============================
# 5. Plot SHAP pour l'interprétation
# ===============================
# SHAP Bar Plot (Importance moyenne des features)
shap.summary_plot(shap_values, feature_names=feature_names, plot_type='bar')

# SHAP Beeswarm Plot (Impact des features sur chaque prédiction)
shap.summary_plot(shap_values, feature_names=feature_names)
