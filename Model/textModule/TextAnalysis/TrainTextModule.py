import pandas as pd
import torch
import re
import numpy as np
from langdetect import detect, LangDetectException
from transformers import BertTokenizer, BertForSequenceClassification
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from torch.utils.data import Dataset, DataLoader
import os
import time

# Track start time
start_time = time.time()

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load dataset
df = pd.read_csv("Data/text_train_set_60_augmented.csv")
df = df[['text', 'label']].dropna()

# Preprocess
def preprocess_text(text):
    if text == "-1":
        return np.nan
    try:
        if detect(text) != 'en':
            return np.nan
    except LangDetectException:
        return np.nan

    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"\S+@\S+\.\S+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

df['text'] = df['text'].apply(preprocess_text)
df = df[df['text'].notna()]

# Hyperparameters (replace with your best) 
best_hyperparams = {
    "batch_size": 16,
    "learning_rate": 3e-05,
    "epochs": 4
}

# Tokenizer
MODEL_NAME = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)

# Dataset
class SpamDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=512):
        self.texts = texts.tolist()
        self.labels = labels.tolist()
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt"
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# Create dataset and loader
dataset = SpamDataset(df['text'], df['label'], tokenizer)
dataloader = DataLoader(dataset, batch_size=best_hyperparams["batch_size"], shuffle=True)

# Model
model = BertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
model.to(device)

# Optimizer, scheduler
optimizer = AdamW(model.parameters(), lr=best_hyperparams["learning_rate"])
total_steps = len(dataloader) * best_hyperparams["epochs"]
scheduler = get_linear_schedule_with_warmup(optimizer, int(0.1 * total_steps), total_steps)

criterion = torch.nn.CrossEntropyLoss()

# Training
model.train()
for epoch in range(best_hyperparams["epochs"]):
    total_loss = 0
    for batch in dataloader:
        optimizer.zero_grad()

        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(input_ids, attention_mask=attention_mask)
        loss = criterion(outputs.logits, labels)
        loss.backward()
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    print(f"Epoch {epoch+1}/{best_hyperparams['epochs']} - Loss: {avg_loss:.4f}")

# Track end time and calculate total runtime
end_time = time.time()
total_time_seconds = end_time - start_time
hours = int(total_time_seconds // 3600)
minutes = int((total_time_seconds % 3600) // 60)
seconds = int(total_time_seconds % 60)
print(f"Total training time: {hours}h {minutes}m {seconds}s")

# Save model and tokenizer
output_dir = "final_bert_model"
os.makedirs(output_dir, exist_ok=True)

model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

print(f"Model and tokenizer saved to `{output_dir}`")