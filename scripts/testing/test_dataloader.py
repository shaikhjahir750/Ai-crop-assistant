import os
from pathlib import Path
from train_plant_disease_model import ArrangedPlantDiseaseDataset

def test_dataloader():
    data_dir = Path("NewPlantDiseaseDataset")
    
    train_ds = ArrangedPlantDiseaseDataset(data_dir, split='train')
    valid_ds = ArrangedPlantDiseaseDataset(data_dir, split='valid')
    
    print(f"Train Dataset size: {len(train_ds)}")
    print(f"Valid Dataset size: {len(valid_ds)}")
    
    # Check for leakage
    train_uuids = set()
    for path, _ in train_ds.samples:
        filename = Path(path).name
        uuid = filename.split('___')[0] if '___' in filename else filename
        train_uuids.add(uuid)
        
    valid_uuids = set()
    for path, _ in valid_ds.samples:
        filename = Path(path).name
        uuid = filename.split('___')[0] if '___' in filename else filename
        valid_uuids.add(uuid)
        
    overlap = train_uuids.intersection(valid_uuids)
    print(f"Unique Train UUIDs: {len(train_uuids)}")
    print(f"Unique Valid UUIDs: {len(valid_uuids)}")
    print(f"Overlapping UUIDs between Train and Valid: {len(overlap)}")

if __name__ == '__main__':
    test_dataloader()
