import os
import json

def parse_disease_guide(file_path):
    if not os.path.exists(file_path):
        print(f"Error: {file_path} does not exist.")
        return {}
        
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    disease_map = {}
    current_crop_names = []
    current_disease_name = None
    current_tips = []
    current_fert = []
    current_mode = None
    
    for line in lines:
        line_str = line.strip()
        if not line_str:
            continue
            
        if line_str.startswith("## ") and not line_str.startswith("### "):
            if current_crop_names and current_disease_name:
                for cn in current_crop_names:
                    disease_map[(cn, current_disease_name.lower())] = {
                        "tips": "\n".join(f"- {t}" for t in current_tips),
                        "fertilizer": "\n".join(f"- {f}" for f in current_fert)
                    }
                current_disease_name = None
                current_tips = []
                current_fert = []
                current_mode = None
                
            crop_part = line_str[3:]
            for emoji in ["🍎", "🫑", "🍒", "🌽", "🍇", "🍑", "🥔", "🍓", "🍅"]:
                crop_part = crop_part.replace(emoji, "")
            crop_part = crop_part.split("(")[0].strip()
            current_crop_names = [c.strip().lower() for c in crop_part.split("/")]
            
        elif line_str.startswith("### "):
            if current_crop_names and current_disease_name:
                for cn in current_crop_names:
                    disease_map[(cn, current_disease_name.lower())] = {
                        "tips": "\n".join(f"- {t}" for t in current_tips),
                        "fertilizer": "\n".join(f"- {f}" for f in current_fert)
                    }
                current_tips = []
                current_fert = []
                current_mode = None
                
            disease_part = line_str[4:]
            current_disease_name = disease_part.split("(")[0].strip()
            
        elif current_disease_name:
            if line_str.startswith("Cultivation Tips:") or line_str.startswith("* Cultivation Tips:"):
                current_mode = "tips"
                parts = line_str.split(":", 1)
                content_str = parts[1].strip() if len(parts) > 1 else ""
                if content_str:
                    current_tips.append(content_str)
            elif line_str.startswith("Fertilizer Recommendation:") or line_str.startswith("* Fertilizer Recommendation:"):
                current_mode = "fert"
                parts = line_str.split(":", 1)
                content_str = parts[1].strip() if len(parts) > 1 else ""
                if content_str:
                    current_fert.append(content_str)
            elif line_str.startswith("*") or line_str.startswith("-"):
                content_str = line_str.lstrip("*- ").strip()
                if current_mode == "tips":
                    current_tips.append(content_str)
                elif current_mode == "fert":
                    current_fert.append(content_str)
                    
    if current_crop_names and current_disease_name:
        for cn in current_crop_names:
            disease_map[(cn, current_disease_name.lower())] = {
                "tips": "\n".join(f"- {t}" for t in current_tips),
                "fertilizer": "\n".join(f"- {f}" for f in current_fert)
            }
            
    return disease_map

def parse_crop_guide(file_path):
    if not os.path.exists(file_path):
        print(f"Error: {file_path} does not exist.")
        return {}
        
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    crop_map = {}
    current_crop = None
    current_tips = []
    current_fert = []
    current_mode = None
    
    for line in lines:
        line_str = line.strip()
        if not line_str:
            continue
            
        if line_str.startswith("### "):
            if current_crop:
                crop_map[current_crop] = {
                    "tips": "\n".join(f"- {t}" for t in current_tips),
                    "fertilizer": "\n".join(f"- {f}" for f in current_fert)
                }
                current_tips = []
                current_fert = []
                current_mode = None
                
            crop_name = line_str[4:].split("(")[0].strip().lower()
            current_crop = crop_name
            
        elif current_crop:
            if "**Cultivation Tips:**" in line_str or "Cultivation Tips:" in line_str:
                current_mode = "tips"
                parts = line_str.split(":", 1)
                content_str = parts[1].strip() if len(parts) > 1 else ""
                content_str = content_str.lstrip("*- ").strip()
                if content_str:
                    current_tips.append(content_str)
            elif "**Fertilizer Recommendation:**" in line_str or "Fertilizer Recommendation:" in line_str:
                current_mode = "fert"
                parts = line_str.split(":", 1)
                content_str = parts[1].strip() if len(parts) > 1 else ""
                content_str = content_str.lstrip("*- ").strip()
                if content_str:
                    current_fert.append(content_str)
            elif line_str.startswith("*") or line_str.startswith("-"):
                content_str = line_str.lstrip("*- ").strip()
                if current_mode == "tips":
                    current_tips.append(content_str)
                elif current_mode == "fert":
                    current_fert.append(content_str)
            elif line_str.startswith("##") and not line_str.startswith("### "):
                if current_crop:
                    crop_map[current_crop] = {
                        "tips": "\n".join(f"- {t}" for t in current_tips),
                        "fertilizer": "\n".join(f"- {f}" for f in current_fert)
                    }
                    current_crop = None
                    current_tips = []
                    current_fert = []
                    current_mode = None
                    
    if current_crop:
        crop_map[current_crop] = {
            "tips": "\n".join(f"- {t}" for t in current_tips),
            "fertilizer": "\n".join(f"- {f}" for f in current_fert)
        }
        
    return crop_map

def normalize_str(s):
    return "".join(c for c in s.lower() if c.isalnum())

if __name__ == "__main__":
    indices_path = "models/class_indices_20260511_204122.json"
    with open(indices_path, "r") as f:
        class_to_idx = json.load(f)
            
    disease_guide = "Copy of disease_management_and_cultivation_guide.txt"
    crop_guide = "cultivation_guide_22_crops.txt"
    
    d_map = parse_disease_guide(disease_guide)
    c_map = parse_crop_guide(crop_guide)
    
    print(f"Parsed {len(d_map)} disease guide entries.")
    print(f"Parsed {len(c_map)} crop guide entries.")
    
    print("\nMatching disease labels from model:")
    all_matched = True
    for label in class_to_idx.keys():
        if label == ".ipynb_checkpoints":
            continue
        parts = label.split(" - ")
        crop = parts[0].strip().lower()
        disease = parts[1].strip().lower()
        
        crop_options = [crop]
        if "(" in crop:
            cleaned_crop = crop.split("(")[0].strip()
            paren_crop = crop.split("(")[1].replace(")", "").strip()
            crop_options.extend([cleaned_crop, paren_crop])
            
        matched = False
        for co in crop_options:
            norm_disease = normalize_str(disease)
            for (map_crop, map_disease), content in d_map.items():
                norm_map_disease = normalize_str(map_disease)
                if map_crop == co and (norm_disease == norm_map_disease or norm_disease in norm_map_disease or norm_map_disease in norm_disease):
                    matched = True
                    break
            if matched:
                break
                
        if not matched:
            print(f"[X] Unmatched: {label} (Crop options: {crop_options}, Disease: {disease})")
            all_matched = False
            
    if all_matched:
        print("[OK] ALL disease labels successfully matched!")
        
    print("\nMatching crop labels from model:")
    crops = ['apple', 'banana', 'blackgram', 'chickpea', 'coconut', 'coffee', 'cotton', 'grapes', 'jute', 'kidneybeans', 'lentil', 'maize', 'mango', 'mothbeans', 'mungbean', 'muskmelon', 'orange', 'papaya', 'pigeonpeas', 'pomegranate', 'rice', 'watermelon']
    all_crops_matched = True
    for crop in crops:
        if crop in c_map:
            pass
        else:
            print(f"[X] Crop Unmatched: {crop}")
            all_crops_matched = False
            
    if all_crops_matched:
        print("[OK] ALL crop labels successfully matched!")
