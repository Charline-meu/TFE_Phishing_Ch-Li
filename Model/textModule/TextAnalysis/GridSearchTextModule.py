import pandas as pd
import time  # Import time module
import re
import torch
from transformers import BertTokenizer, BertForSequenceClassification, get_linear_schedule_with_warmup
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
#from torch.nn.parallel import DistributedDataParallel as DDP
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score, classification_report
from langdetect import detect, LangDetectException
import numpy as np

# Track start time
start_time = time.time()

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}\n")

# Load dataset
file_path = "Data/text_train_set_60_augmented.csv"

print("Loading datasets...")
df = pd.read_csv(file_path)

# Combine datasets
df = df[['text', 'label']].dropna()
#df = df.sample(n=100, random_state=42)  # Select only 100 emails for faster training
print(f"Dataset size: {len(df)} emails\n")
print("Datasets loaded!\n")

# Preprocessing function
def preprocess_text(text):
    if text == "-1":
        return np.nan  # Return NaN for invalid "-1" entries

    try:
        if detect(text) != 'en':
            return np.nan  # Return NaN for non-English entries
    except LangDetectException:
        return np.nan  # Return NaN when detection fails

    text = re.sub(r"http\S+|www\S+|https\S+", "", text)  # Remove URLs
    text = re.sub(r"\d+", "", text)  # Remove numbers
    text = re.sub(r"\S+@\S+\.\S+", "", text)  # Remove email addresses
    text = re.sub(r"\s+", " ", text).strip()  # Normalize spaces
    return text

# Apply text preprocessing
print("Preprocessing email content...")
df['text'] = df['text'].apply(preprocess_text)
df = df[df['text'].notna()]  # Remove NaN entries
print(f"Preprocessed dataset size: {len(df)} emails\n")
print("Text preprocessing complete!\n")

# Load BERT tokenizer
MODEL_NAME = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.2, random_state=42)
print(f"Training set: {len(X_train)} emails, Test set: {len(X_test)} emails\n")

# Define PyTorch Dataset class
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

train_dataset = SpamDataset(X_train, y_train, tokenizer)
test_dataset = SpamDataset(X_test, y_test, tokenizer)

# Define hyperparameter search space
batch_sizes = [16, 32]
learning_rates = [2e-5, 3e-5, 5e-5]
epochs_list = [2, 3, 4]

# Initialize variables to track the best model
best_auprc = 0.0
#best_model = None
best_hyperparams = {}
best_y_preds = []
best_y_true = []

# Iterate through hyperparameter combinations
for batch_size in batch_sizes:
    for lr in learning_rates:
        for epochs in epochs_list:
            print(f"\nTraining with batch_size={batch_size}, lr={lr}, epochs={epochs}")

            # Create DataLoaders
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

            # Initialize process group
            #torch.distributed.init_process_group(backend="nccl")

            # Initialize BERT model with dropout 0.1
            bert_model = BertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
            bert_model.to(device)
            #bert_model = DDP(bert_model)

            # Define optimizer with learning rate warmup over 10,000 steps
            optimizer = AdamW(bert_model.parameters(), lr=lr)
            total_steps = len(train_loader) * epochs
            scheduler = get_linear_schedule_with_warmup(optimizer, int(0.1 * total_steps), num_training_steps=total_steps)

            criterion = torch.nn.CrossEntropyLoss()

            # Training loop
            bert_model.train()
            for epoch in range(epochs):
                total_loss = 0
                for batch_idx, batch in enumerate(train_loader):
                    optimizer.zero_grad()
                    
                    input_ids = batch["input_ids"].to(device)
                    attention_mask = batch["attention_mask"].to(device)
                    labels = batch["labels"].to(device)

                    outputs = bert_model(input_ids, attention_mask=attention_mask)
                    loss = criterion(outputs.logits, labels)
                    loss.backward()
                    optimizer.step()
                    scheduler.step()

                    total_loss += loss.item()

                avg_loss = total_loss / len(train_loader)
                print(f"Epoch {epoch+1}/{epochs} - Avg Loss: {avg_loss:.4f}")

            # Evaluate model using AUPRC
            bert_model.eval()
            y_preds = []
            y_probs = []
            y_true = []

            with torch.no_grad():
                for batch in test_loader:
                    input_ids = batch["input_ids"].to(device)
                    attention_mask = batch["attention_mask"].to(device)
                    labels = batch["labels"].cpu().numpy()

                    outputs = bert_model(input_ids, attention_mask=attention_mask)
                    probs = torch.nn.functional.softmax(outputs.logits, dim=1).cpu().numpy()[:, 1]  # Get positive class probs
                    preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()

                    y_probs.extend(probs)  # Probabilities for AUPRC
                    y_preds.extend(preds)
                    y_true.extend(labels)

            # Calculate AUPRC
            auprc = average_precision_score(y_true, y_probs)
            print(f"AUPRC: {auprc:.4f}")

            # Print classification report
            print("\nFinal Model Performance:")
            print(classification_report(y_true, y_preds))

            # Save the best model
            if auprc > best_auprc:
                best_auprc = auprc
                #best_model = bert_model
                best_y_preds = y_preds
                best_y_true = y_true
                best_hyperparams = {"batch_size": batch_size, "learning_rate": lr, "epochs": epochs}

# Track end time and calculate total runtime
end_time = time.time()
total_time_seconds = end_time - start_time
hours = int(total_time_seconds // 3600)
minutes = int((total_time_seconds % 3600) // 60)
seconds = int(total_time_seconds % 60)


# Save the best model
# MODEL_SAVE_PATH = "best_bert_spam_classifier_auprc.pth"
# print(f"\nBest model found with AUPRC {best_auprc:.4f}: {best_hyperparams}")
# torch.save(best_model.state_dict(), MODEL_SAVE_PATH)

# Save best AUPRC and hyperparameters to a text file
with open("best_model_info.txt", "w") as f:
    f.write(f"Best AUPRC: {best_auprc:.4f}\n")
    f.write(f"Best Hyperparameters: {best_hyperparams}\n")
    f.write(f"Classification Report:\n{classification_report(best_y_true, best_y_preds)}\n")
    f.write(f"Total Execution Time: {hours}h {minutes}m {seconds}s\n")

#print(f"Best model saved to {MODEL_SAVE_PATH}")
print(f"Total Execution Time: {hours}h {minutes}m {seconds}s")