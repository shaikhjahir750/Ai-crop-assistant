import os

with open('train_plant_disease_model.py', 'r') as f:
    content = f.read()

content = content.replace('Path("NewPlantDiseaseDataset")', 'Path("Plant Village Dataset")')
content = content.replace('Path("plant disease detection plots")', 'Path("plant village detection plots")')
content = content.replace("['train', 'valid']", "['Train', 'Val']")
# Change the model prefix name to prevent overlapping models
content = content.replace("disease_detection_model_", "plant_village_model_")

with open('train_plant_village_model.py', 'w') as f:
    f.write(content)
