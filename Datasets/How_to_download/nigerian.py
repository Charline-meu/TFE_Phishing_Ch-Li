#https://www.kaggle.com/datasets/rtatman/fraudulent-email-corpus/data
import kagglehub

# Download latest version
path = kagglehub.dataset_download("rtatman/fraudulent-email-corpus")

print("Path to dataset files:", path)