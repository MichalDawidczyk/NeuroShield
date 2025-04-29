import os
import kaggle

datasets = [
    "mlg-ulb/creditcardfraud"
]

os.makedirs("datasets", exist_ok=True)

for dataset in datasets:
    kaggle.api.dataset_download_files(dataset, path="datasets", unzip=True)
    print(f"Downloaded {dataset} successfully!")

print("All datasets are downloaded and extracted in the 'datasets' folder.")
