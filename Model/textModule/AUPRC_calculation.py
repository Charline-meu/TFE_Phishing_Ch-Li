import pandas as pd
from sklearn.metrics import average_precision_score
import sys 

# Load the dataset
df = pd.read_csv(sys.argv[1])

# Extract labels and predicted probabilities
labels = df['label']
predictions = df['text_proba']

# Calculate AUPRC
auprc = average_precision_score(labels, predictions)
print(f"AUPRC: {auprc:.4f}")