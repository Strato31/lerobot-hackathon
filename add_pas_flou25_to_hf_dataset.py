import os
from datasets import Dataset, DatasetDict
import pandas as pd

# Chemin vers le dossier pas_flou25
data_dir = "pas_flou25"

# Préparer une liste pour stocker les données
data = []

# Parcourir les fichiers dans le dossier "data"
data_folder = os.path.join(data_dir, "data")
for root, _, files in os.walk(data_folder):
    for file in files:
        if file.endswith(".jpg") or file.endswith(".png"):
            file_path = os.path.join(root, file)
            # Ajouter les données (ajustez les colonnes selon vos besoins)
            data.append({
                "image_path": file_path,
                "label": os.path.basename(root)  # Exemple : le nom du dossier comme étiquette
            })

# Convertir les données en DataFrame
df = pd.DataFrame(data)

# Créer un dataset Hugging Face
dataset = Dataset.from_pandas(df)

# Optionnel : Diviser en train/test
train_test_split = dataset.train_test_split(test_size=0.2)
dataset_dict = DatasetDict({
    "train": train_test_split["train"],
    "test": train_test_split["test"]
})

# Sauvegarder localement
dataset_dict.save_to_disk("pas_flou25_hf_dataset")

print("Dataset sauvegardé dans le dossier 'pas_flou25_hf_dataset'.")

# Optionnel : Téléverser sur le Hub Hugging Face
# from huggingface_hub import HfApi
# api = HfApi()
# api.upload_folder(folder_path="pas_flou25_hf_dataset", repo_id="<votre_nom_utilisateur>/<nom_du_dataset>")