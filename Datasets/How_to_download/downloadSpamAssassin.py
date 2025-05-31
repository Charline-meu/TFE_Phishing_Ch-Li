#https://www.kaggle.com/datasets/ganiyuolalekan/spam-assassin-email-classification-dataset
import kagglehub

# Download latest version
path = kagglehub.dataset_download("ganiyuolalekan/spam-assassin-email-classification-dataset")

print("Path to dataset files:", path)