#https://www.kaggle.com/datasets/mandygu/lingspam-dataset/data
import kagglehub

# Download latest version
path = kagglehub.dataset_download("mandygu/lingspam-dataset")

print("Path to dataset files:", path)