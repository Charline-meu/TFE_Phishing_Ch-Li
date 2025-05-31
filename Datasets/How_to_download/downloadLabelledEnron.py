#https://www.kaggle.com/datasets/marcelwiechmann/enron-spam-data?select=README.md
import kagglehub

# Download latest version
path = kagglehub.dataset_download("marcelwiechmann/enron-spam-data")

print("Path to dataset files:", path)