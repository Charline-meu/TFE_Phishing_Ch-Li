# cnn_lstm_train.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

# ===============================
# 1. Définition du Modèle CNN-LSTM (Sans features lexicales)
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
    
class URLDataset(Dataset):
        def __init__(self, X, y):
            # X = URLs encodés en séquence numérique, Y = labels des URL
            self.X = X
            self.y = y

        def __len__(self):
            #nombre total d'URLs
            return len(self.X)

        def __getitem__(self, idx):
            return torch.tensor(self.X[idx], dtype=torch.long), torch.tensor(self.y[idx], dtype=torch.float32)

# ===============================
# 2. Encodage des URL en caractères
# ===============================
def encode_url(url, max_len=200):
    encoded = [ord(char) for char in url if ord(char) < 128]
    return np.pad(encoded[:max_len], (0, max(0, max_len - len(encoded))), 'constant')

# ===============================
# 3. Entraînement du modèle (Uniquement si le fichier est exécuté directement)
# ===============================
if __name__ == "__main__":
    # Charger les données
    csv_path = 'datasetURLs/ebubekirbbr.csv'
    data = pd.read_csv(csv_path)
    urls = data['url'].str.lower().values
    labels = data['label'].values

    X = np.array([encode_url(url) for url in urls])  # Encodage des caractères
    y = labels

    # Division en ensembles d'entraînement et de test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') #si l'hote à un GPU, on utilise CUDA

    train_dataset = URLDataset(X_train, y_train)
    test_dataset = URLDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Initialisation du modèle
    model = PureCNNLSTM().to(device)
    #frequemment utilisé quand la fonction d'activation dans les couches denses finales est une sigmoid 
    criterion = nn.BCELoss()    #fonction de perte
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # Entraînement du modèle
    num_epochs = 4
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        print(f'🔄 Époque {epoch + 1}/{num_epochs} - Perte: {total_loss / len(train_loader):.4f}')

    # Sauvegarde du modèle et des données de test
    torch.save(model.state_dict(), 'cnn_lstm_model.pth')
    print('💾 Modèle CNN-LSTM sauvegardé avec succès !')

    np.save('X_test.npy', X_test)
    np.save('y_labels.npy', y_test)
    print('✅ Données de test sauvegardées sous X_test.npy et y_labels.npy')
