import os
from dataset import load_dataset

DATASET_PATH = r"D:\vlmod\MonoMulti3D\test"
OUTPUT_DIR = r"D:\vlmod\result"

def match_object_to_text(obj, text_description):
    """
    Evaluates whether a 3D object matches a text description query.
    Extracts category and visual attributes (e.g., color) to verify matches.
    """
    desc_lower = text_description.lower()
    
    # Check object category/class match
    class_name = obj.get("class_name", "").lower()
    class_match = class_name in desc_lower if class_name else False
    
    # Check object color match (if color exists)
    color = obj.get("color", "unknown").lower()
    color_match = (color in desc_lower) if color != "unknown" else True
    
    # Return 1 if both match conditions are satisfied, else 0
    return 1 if (class_match and color_match) else 0

def process_and_generate_results():
    print(f"Loading test dataset from: {DATASET_PATH}")
    dataset, errors = load_dataset(DATASET_PATH)
    
    print(f"Loaded {len(dataset)} JSON files.")
    if errors:
        print(f"Encountered {len(errors)} errors during loading.")
        
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    generated_count = 0

    for item in dataset:
        parsed_objects = item["objects"]
        descriptions = item["descriptions"]
        
        matrix = []
        for obj in parsed_objects:
            row = []
            for desc in descriptions:
                score = match_object_to_text(obj, desc)
                row.append(str(score))
            matrix.append(" ".join(row))

        # Save result txt file matching the JSON filename
        base_name = os.path.splitext(item["file_name"])[0]
        out_filepath = os.path.join(OUTPUT_DIR, f"{base_name}.txt")

        with open(out_filepath, "w", encoding="utf-8") as f:
            f.write("\n".join(matrix))
        
        generated_count += 1

    print(f"\nSUCCESS: Generated {generated_count} prediction .txt files in '{OUTPUT_DIR}'")

if __name__ == "__main__":
    process_and_generate_results()