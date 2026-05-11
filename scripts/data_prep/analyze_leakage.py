import os
import re
from pathlib import Path

def analyze(dataset_name, split_names):
    p = Path(dataset_name)
    if not p.exists():
        print(f"Skipping {dataset_name}, not found.")
        return
        
    uuids = {split: set() for split in split_names}
    for split in split_names:
        split_path = p / split
        if not split_path.exists(): continue
        
        files = list(split_path.rglob('*.JPG')) + list(split_path.rglob('*.jpg'))
        files = [f.name for f in files]
        # Extract UUID (everything before '___')
        for f in files:
            uuid = f.split('___')[0] if '___' in f else f
            uuids[split].add(uuid)
            
    if 'Train' in uuids and 'Val' in uuids:
        intersect = uuids['Train'].intersection(uuids['Val'])
        print(f"{dataset_name} -> Overlapping base images between Train and Val: {len(intersect)}")
    if 'train' in uuids and 'valid' in uuids:
        intersect = uuids['train'].intersection(uuids['valid'])
        print(f"{dataset_name} -> Overlapping base images between train and valid: {len(intersect)}")

analyze('Plant Village Dataset', ['Train', 'Val', 'Test'])
analyze('PlantDiseasesDataset', ['train', 'valid', 'test'])
