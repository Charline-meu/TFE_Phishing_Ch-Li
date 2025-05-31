import torch
import time
import re
import pandas as pd
import numpy as np
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import Dataset, DataLoader, Subset
import torch.nn.functional as F
from sklearn.metrics import average_precision_score
from sklearn.model_selection import KFold

# Track start time
start_time = time.time()

# Define dataset class
class EmailDataset(Dataset):
    def __init__(self, texts, tokenizer, max_len=512):
        self.texts = texts.tolist()
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt"
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0)
        }

# Load model and tokenizer
MODEL_NAME = "bert-base-uncased"
MODEL_SAVE_PATH = "BERT_fine-tuned/best_bert_spam_classifier_auprc.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device}")

print(f"Loading tokenizer and model: {MODEL_NAME}")
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
model = BertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
print("Loading fine-tuned model...")
model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device))
model.to(device)
model.eval()
print("Model loaded successfully!\n")

# Load dataset
DATASET_PATH = "Data/text_test.csv"
print(f"Loading dataset from {DATASET_PATH}")
df = pd.read_csv(DATASET_PATH)
texts = df['text']
labels = df['label'].values
print(f"Dataset loaded! {len(texts)} emails found.\n")

# Preprocessing
print("Preprocessing email content...")
def preprocess_text(text):
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"\S+@\S+\.\S+", "", text)
    text = re.sub(r"[^A-Za-z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

texts = texts.apply(preprocess_text)
print("Text preprocessing complete!\n")

# Create dataset object
email_dataset = EmailDataset(texts, tokenizer)

# 10-Fold Cross-Validation
kf = KFold(n_splits=10, shuffle=True, random_state=42)
auprc_scores = []

print("Starting 10-Fold Cross-Validation evaluation...\n")

for fold, (train_index, test_index) in enumerate(kf.split(email_dataset)):
    print(f"Fold {fold + 1}/10")
    test_subset = Subset(email_dataset, test_index)
    test_labels = labels[test_index]
    test_loader = DataLoader(test_subset, batch_size=32, shuffle=False)

    pred_probs = []
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            outputs = model(input_ids, attention_mask=attention_mask)
            probs = F.softmax(outputs.logits, dim=1)[:, 1]
            pred_probs.extend(probs.cpu().numpy())

    auprc = average_precision_score(test_labels, pred_probs)
    auprc_scores.append(auprc)
    print(f"Fold {fold + 1} AUPRC: {auprc:.4f}\n")

# Report average AUPRC
avg_auprc = np.mean(auprc_scores)
print(f"Average AUPRC over 10 folds: {avg_auprc:.4f}")

# Runtime
end_time = time.time()
total_time_seconds = end_time - start_time
hours = int(total_time_seconds // 3600)
minutes = int((total_time_seconds % 3600) // 60)
seconds = int(total_time_seconds % 60)
print(f"Total runtime: {hours} hours, {minutes} minutes, {seconds} seconds")
