import os
import shutil
from pathlib import Path
import re

def arrange_dataset(src_dir="PlantDiseasesDataset", dst_dir="ArrangedPlantDiseasesDataset"):
    src_path = Path(src_dir)
    dst_path = Path(dst_dir)
    
    if not src_path.exists():
        print(f"Source directory {src_dir} does not exist.")
        return
        
    if dst_path.exists():
        print(f"Destination directory {dst_dir} already exists. Cleaning it up...")
        shutil.rmtree(dst_path)
        
    print(f"Creating structured dataset at {dst_dir}...")
    
    # Valid image extensions
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    
    # We find all images and determine their crop, split, and disease
    # Original structure: PlantDiseasesDataset/<Crop>/<split>/<Crop_Disease>/image.jpg
    count = 0
    for img_path in src_path.rglob('*.*'):
        if img_path.suffix.lower() not in valid_extensions:
            continue
            
        parts = img_path.relative_to(src_path).parts
        if len(parts) < 4:
            continue
            
        crop = parts[0]
        split = parts[1]
        raw_disease = parts[2]
        filename = parts[3]
        
        # Strip the crop name from the beginning of the disease name
        # We use a case-insensitive regex to replace the crop name if it's at the start
        pattern = re.compile(rf"^{re.escape(crop)}\s+", re.IGNORECASE)
        clean_disease = pattern.sub("", raw_disease)
        
        # Some edge cases mapping to be fully consistent with NewPlantDiseaseDataset
        clean_disease = clean_disease.title()
        
        # Construct new path: ArrangedPlantDiseasesDataset/<split>/<crop>/<clean_disease>/<filename>
        new_dir = dst_path / split / crop / clean_disease
        new_dir.mkdir(parents=True, exist_ok=True)
        
        new_img_path = new_dir / filename
        shutil.copy2(img_path, new_img_path)
        count += 1
        
        if count % 1000 == 0:
            print(f"Processed {count} images...")
            
    print(f"Done! Copied and arranged {count} images.")

if __name__ == "__main__":
    arrange_dataset()
