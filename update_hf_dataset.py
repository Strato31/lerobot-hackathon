import os
from datasets import Dataset, DatasetDict, load_from_disk
import pandas as pd

# Chemin vers le dataset Hugging Face existant
existing_dataset_path = "datasets/Strato31/hackathon_team02"  # Remplacez par le chemin réel

# Charger le dataset existant
dataset_dict = load_from_disk(existing_dataset_path)

# Chemin vers le dossier pas_flou25
data_dir = "pas_flou25"

# Préparer une liste pour stocker les nouvelles données
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

# Convertir les nouvelles données en DataFrame
df = pd.DataFrame(data)

# Créer un dataset Hugging Face pour les nouvelles données
new_dataset = Dataset.from_pandas(df)

# Ajouter les nouvelles données au dataset existant
updated_train = dataset_dict["train"].concat(new_dataset)
updated_test = dataset_dict["test"]  # Garder le test inchangé (ou ajuster si nécessaire)

# Mettre à jour le dataset dict
updated_dataset_dict = DatasetDict({
    "train": updated_train,
    "test": updated_test
})

# Sauvegarder le dataset mis à jour
updated_dataset_dict.save_to_disk("updated_hf_dataset")

print("Dataset mis à jour et sauvegardé dans le dossier 'updated_hf_dataset'.")