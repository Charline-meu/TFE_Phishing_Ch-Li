#https://www.kaggle.com/datasets/subhajournal/phishingemails/data
#dataset utilisé dans : An Explainable Transformer-based Model for Phishing Email Detection: A Large Language Model Approach
import kagglehub

# Download latest version
path = kagglehub.dataset_download("subhajournal/phishingemails")

print("Path to dataset files:", path)