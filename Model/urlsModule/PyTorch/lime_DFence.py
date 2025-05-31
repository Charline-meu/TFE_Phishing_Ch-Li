# lime_explain_cnn.py

import torch
import numpy as np
import pandas as pd
from lime.lime_text import LimeTextExplainer
import torch.nn as nn
from collections import Counter
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# ===============================
# 1. Définition du Modèle CNN-LSTM (Même qu'au moment de l'entraînement)
# ===============================
class PureCNNLSTM(nn.Module):
    def __init__(self):
        super(PureCNNLSTM, self).__init__()
        self.embedding = nn.Embedding(128, 128)
        self.conv1 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=4, stride=1, padding=2)
        self.pool = nn.MaxPool1d(kernel_size=4, stride=4)
        self.lstm = nn.LSTM(input_size=256, hidden_size=512, num_layers=1, batch_first=True)
        
        # Combinaison CNN-LSTM sans MLP (Pas de fc_lexical)
        self.fc1 = nn.Linear(512, 256)
        self.dropout = nn.Dropout(0.1)
        self.fc2 = nn.Linear(256, 1)

    def forward(self, x):
        # Partie CNN-LSTM pour les caractères
        x = self.embedding(x)
        x = x.permute(0, 2, 1)
        x = torch.relu(self.conv1(x))
        x = self.pool(x)
        x = x.permute(0, 2, 1)
        _, (hn, _) = self.lstm(x)
        x = hn[-1]
        
        # Fully Connected Layers
        x = self.fc1(x)
        x = self.dropout(x)
        x = torch.sigmoid(self.fc2(x))
        return x.squeeze()

# ===============================
# 2. Charger le modèle et les données de test
# ===============================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Charger le modèle CNN-LSTM
model = PureCNNLSTM().to(device)
model.load_state_dict(torch.load('cnn_lstm_model.pth', map_location=device))
model.eval()

# Charger le CSV pour avoir les URL en clair
csv_path = 'datasetURLs/ebubekirbbr.csv'
data = pd.read_csv(csv_path)
urls = data['url'].values

# Charger les données de test pour faire les prédictions
X_test = np.load('X_test.npy')

# ===============================
# 3. Définir le prédicteur pour LIME
# ===============================
def encode_url(url, max_len=200):
    encoded = [ord(char) for char in url if ord(char) < 128]
    return np.pad(encoded[:max_len], (0, max(0, max_len - len(encoded))), 'constant')

def predict_fn(text_list):
    model.eval()
    encoded_urls = [encode_url(text) for text in text_list]
    encoded_tensor = torch.tensor(encoded_urls, dtype=torch.long).to(device)
    with torch.no_grad():
        outputs = model(encoded_tensor)
    return np.vstack([(1 - outputs.cpu().numpy()), outputs.cpu().numpy()]).T

# ===============================
# 4. Initialisation de LIME
# ===============================
explainer = LimeTextExplainer(class_names=['Légitime', 'Phishing'])

# ===============================
# 5. Généralisation des explications sur 100 URL
# ===============================
num_samples = 5000  # Nombre d'URL à expliquer
important_features = []

for i in range(num_samples):
    example_url = urls[i]
    exp = explainer.explain_instance(example_url, predict_fn, num_features=10)
    explanation = exp.as_list()
    
    # Récupération des mots clés et de leur importance
    for word, importance in explanation:
        important_features.append(word)
    
    print(f'✅ Explication pour l\'URL {i+1}/{num_samples}')

# ===============================
# 6. Agrégation des explications
# ===============================
feature_counter = Counter(important_features)
most_common_features = feature_counter.most_common(20)

# ===============================
# 7. Visualisation des résultats globaux
# ===============================
# Bar Plot des mots clés les plus influents
words, counts = zip(*most_common_features)
plt.figure(figsize=(12, 8))
plt.barh(words, counts, color='skyblue')
plt.xlabel('Fréquence d\'apparition')
plt.title('🔎 Mots clés les plus influents (LIME)')
plt.gca().invert_yaxis()
plt.show()

# Word Cloud des mots clés les plus influents
wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(feature_counter)
plt.figure(figsize=(15, 7))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('🔎 Word Cloud des mots clés les plus influents (LIME)')
plt.show()
