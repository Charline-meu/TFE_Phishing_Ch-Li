# hybrid_cnn_lstm_train.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
import numpy as np

# ===============================
# 1. Charger et prétraiter les données
# ===============================
csv_path = 'datasetURLs/ebubekirbbr.csv'  # Remplace par le chemin correct
data = pd.read_csv(csv_path)

# Conversion en minuscules
urls = data['url'].str.lower().values
labels = data['label'].values

# Encodage des URL en entiers (caractères)
def encode_url(url, max_len=200):
    encoded = [ord(char) for char in url if ord(char) < 128]
    return np.pad(encoded[:max_len], (0, max(0, max_len - len(encoded))), 'constant')

X_chars = np.array([encode_url(url) for url in urls])  # Encodage des caractères

# Utilisation des features lexicales directement depuis le CSV
lexical_features = data.drop(['url', 'label'], axis=1).values
X_combined = [np.concatenate((chars, lexical)) for chars, lexical in zip(X_chars, lexical_features)]
X = np.array(X_combined)
y = labels

# Division en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ===============================
# 2. Création du Dataset et DataLoader
# ===============================
class URLDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        chars = self.X[idx][:200]  # Caractères encodés
        lexical = self.X[idx][200:]  # Features lexicales
        
        chars_tensor = torch.tensor(chars, dtype=torch.long)
        lexical_tensor = torch.tensor(lexical, dtype=torch.float32)
        
        combined_tensor = torch.cat((chars_tensor, lexical_tensor))
        
        return combined_tensor, torch.tensor(self.y[idx], dtype=torch.float32)

train_dataset = URLDataset(X_train, y_train)
test_dataset = URLDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# ===============================
# 3. Définition du Modèle Hybride CNN-LSTM + MLP
# ===============================
class HybridCNNLSTM(nn.Module):
    def __init__(self, num_lexical_features):
        super(HybridCNNLSTM, self).__init__()
        self.embedding = nn.Embedding(128, 128)
        self.conv1 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=4, stride=1, padding=2)
        self.pool = nn.MaxPool1d(kernel_size=4, stride=4)
        self.lstm = nn.LSTM(input_size=256, hidden_size=512, num_layers=1, batch_first=True)
        
        # MLP pour les features lexicales
        self.fc_lexical = nn.Linear(num_lexical_features, 32)
        
        # Combinaison CNN-LSTM + MLP
        self.fc1 = nn.Linear(512 + 32, 256)
        self.dropout = nn.Dropout(0.1)
        self.fc2 = nn.Linear(256, 1)

    def forward(self, x):
        chars = x[:,:200].long()  # Caractères encodés
        lexical = x[:,200:].float()  # Features lexicales
        
        # Partie CNN-LSTM pour les caractères
        x = self.embedding(chars)
        x = x.permute(0, 2, 1)
        x = torch.relu(self.conv1(x))
        x = self.pool(x)
        x = x.permute(0, 2, 1)
        _, (hn, _) = self.lstm(x)
        x = hn[-1]
        
        # Partie MLP pour les features lexicales
        lexical_out = torch.relu(self.fc_lexical(lexical))
        
        # Combinaison des deux parties
        combined = torch.cat((x, lexical_out), dim=1)
        
        # Fully Connected Layers
        x = self.fc1(combined)
        x = self.dropout(x)
        x = torch.sigmoid(self.fc2(x))
        return x.squeeze()

num_lexical_features = lexical_features.shape[1]  # Nombre de features lexicales
model = HybridCNNLSTM(num_lexical_features).to(device)

criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# ===============================
# 4. Entraînement du modèle
# ===============================
num_epochs = 5
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for X_batch, y_batch in train_loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    print(f'🔄 Époque {epoch + 1}/{num_epochs} - Perte: {total_loss / len(train_loader):.4f}')
# ===============================
# 5. Évaluation du modèle
# ===============================
model.eval()
y_pred = []
y_true = []
with torch.no_grad():
    for X_batch, y_batch in test_loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        
        outputs = model(X_batch)
        predictions = (outputs > 0.5).long()
        y_pred.extend(predictions.cpu().numpy())
        y_true.extend(y_batch.cpu().numpy())

accuracy = accuracy_score(y_true, y_pred)
print(f'✅ Précision sur le test set: {accuracy:.4f}')
print(classification_report(y_true, y_pred, target_names=['Légitime', 'Phishing']))

# ===============================
# 6. Sauvegarde du modèle et des données de test
# ===============================
torch.save(model.state_dict(), 'hybrid_cnn_lstm_model.pth')
print('💾 Modèle hybride CNN-LSTM sauvegardé avec succès !')

np.save('X_test.npy', X_test)
np.save('y_labels.npy', y_test)
print('✅ Données de test sauvegardées sous X_test.npy et y_labels.npy')
