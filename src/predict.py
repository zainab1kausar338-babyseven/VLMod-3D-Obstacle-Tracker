import json
import glob
import os
import zipfile
import numpy as np

INPUT_JSON_DIR = r"D:\vlmod\MonoMulti3D\test"
OUTPUT_RESULT_DIR = r"D:\vlmod\result"
ZIP_OUTPUT_PATH = r"D:\vlmod\submission.zip"

os.makedirs(OUTPUT_RESULT_DIR, exist_ok=True)

def parse_spatial_position(test_data_str):
    tokens = test_data_str.strip().split()
    if not tokens:
        return "", 0.0, 0.0, ""
    
    obj_type = tokens[0].lower()
    colors = ['white', 'black', 'red', 'blue', 'silver', 'gray', 'grey', 'yellow', 'green', 'orange', 'brown']
    obj_color = tokens[-1].lower() if tokens[-1].lower() in colors else ""
    
    x_pos, depth = 0.0, 0.0
    floats = []
    for token in tokens[1:-1]:
        try:
            floats.append(float(token))
        except ValueError:
            continue
            
    if len(floats) >= 3:
        x_pos = floats[0]
        depth = floats[2]
        
    return obj_type, x_pos, depth, obj_color

def evaluate_match_score(test_data_str, description):
    obj_type, x_pos, depth, obj_color = parse_spatial_position(test_data_str)
    desc = description.lower()
    
    # Strict Category Gate with Synonyms
    category_match = False
    car_keywords = ['car', 'vehicle', 'sedan', 'suv', 'automobile', 'auto', 'crossover']
    truck_keywords = ['truck', 'van', 'pickup', 'lorry', 'freight']
    bus_keywords = ['bus', 'coach']
    bike_keywords = ['bike', 'bicycle', 'motorcycle', 'cyclist', 'rider', 'scooter']
    person_keywords = ['pedestrian', 'person', 'walker', 'man', 'woman', 'people', 'child']

    if obj_type in ['car', 'sedan', 'suv', 'vehicle', 'automobile'] and any(k in desc for k in car_keywords):
        category_match = True
    elif obj_type in ['truck', 'van', 'pickup'] and any(k in desc for k in truck_keywords):
        category_match = True
    elif obj_type in ['bus'] and any(k in desc for k in bus_keywords):
        category_match = True
    elif obj_type in ['cyclist', 'bicycle', 'motorcycle'] and any(k in desc for k in bike_keywords):
        category_match = True
    elif obj_type in ['pedestrian', 'person'] and any(k in desc for k in person_keywords):
        category_match = True

    if not category_match:
        return 0.0 
        
    # Base score for matching category safely
    score = 0.40
    
    # Color Match Weight (Higher precision reward if color matches)
    if obj_color and obj_color in desc:
        score += 0.35
        
    # Spatial Alignment (X-Axis)
    if x_pos < -1.0 and any(k in desc for k in ['left', 'west', 'leftside']):
        score += 0.20
    elif x_pos > 1.0 and any(k in desc for k in ['right', 'east', 'rightside']):
        score += 0.20
    elif -1.0 <= x_pos <= 1.0 and any(k in desc for k in ['center', 'middle', 'front', 'ahead']):
        score += 0.15
        
    # Depth Alignment (Z-Axis)
    if depth > 18.0 and any(k in desc for k in ['far', 'distant', 'back', 'background']):
        score += 0.15
    elif depth < 15.0 and any(k in desc for k in ['near', 'close', 'front', 'foreground']):
        score += 0.15

    return score

def get_binary_flags(test_data_str, public_descriptions, threshold=0.74):
    # Balanced threshold to cut false positives while maintaining healthy recall
    flags = [0, 0, 0]
    scores = [evaluate_match_score(test_data_str, desc) for desc in public_descriptions[:3]]
    
    for idx, score in enumerate(scores):
        if score >= threshold:
            flags[idx] = 1
            
    return flags

# Execution Loop
json_files = glob.glob(os.path.join(INPUT_JSON_DIR, "*.json"))
print(f"--- Starting Balanced Optimized Pipeline ({len(json_files)} files) ---")

for json_path in json_files:
    base_name = os.path.splitext(os.path.basename(json_path))[0]
    output_txt_path = os.path.join(OUTPUT_RESULT_DIR, f"{base_name}.txt")
    
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    descriptions = data.get("public_description", [])
    test_objects = data.get("test_data", [])
    
    output_lines = []
    for obj_entry in test_objects:
        flags = get_binary_flags(obj_entry, descriptions, threshold=0.74)
        output_lines.append(f"{flags[0]} {flags[1]} {flags[2]}")
        
    with open(output_txt_path, 'w', encoding='utf-8') as f_out:
        f_out.write("\n".join(output_lines) + "\n")

# Zip Results into single submission file
with zipfile.ZipFile(ZIP_OUTPUT_PATH, 'w', zipfile.ZIP_DEFLATED) as zipf:
    for root, _, files in os.walk(OUTPUT_RESULT_DIR):
        for file in files:
            if file.endswith('.txt'):
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, os.path.dirname(OUTPUT_RESULT_DIR))
                zipf.write(file_path, arcname)

print(f"COMPLETE! Balanced submission package ready at: {ZIP_OUTPUT_PATH}")