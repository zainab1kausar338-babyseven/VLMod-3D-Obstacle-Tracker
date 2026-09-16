import json
import os

def parse_test_data_line(line):
    """
    Parses a single string entry from test_data.
    Example line:
    "car 0 1 1.80179 432.32 161.03 489.91 195.76 1.33 1.72 4.59 -23.59 -17.34 107.76 1.58 white"
    """
    parts = line.strip().split()
    if not parts:
        return None
    
    return {
        "class_name": parts[0],
        "truncated": float(parts[1]),
        "occluded": int(parts[2]),
        "alpha": float(parts[3]),
        "bbox_2d": [float(x) for x in parts[4:8]],    # [left, top, right, bottom]
        "dimensions": [float(x) for x in parts[8:11]], # [height, width, length]
        "location": [float(x) for x in parts[11:14]],  # [x, y, z] in camera coordinates
        "rotation_y": float(parts[14]),
        "color": parts[15] if len(parts) > 15 else "unknown"
    }

def load_json_file(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Extract public descriptions and raw test_data
    descriptions = data.get("public_description", [])
    raw_objects = data.get("test_data", [])

    # Parse 3D object attributes
    parsed_objects = [parse_test_data_line(obj) for obj in raw_objects if obj]

    return {
        "file_name": os.path.basename(filepath),
        "descriptions": descriptions,
        "objects": parsed_objects,
        "raw_objects": raw_objects
    }

def load_dataset(folder_path):
    dataset = []
    errors = []

    files = [f for f in os.listdir(folder_path) if f.endswith(".json")]
    print(f"Found {len(files)} JSON files.")

    for filename in files:
        filepath = os.path.join(folder_path, filename)
        try:
            item = load_json_file(filepath)
            dataset.append(item)
        except Exception as e:
            errors.append((filename, str(e)))

    print(f"Successfully loaded: {len(dataset)} files")
    print(f"Errors: {len(errors)}")

    return dataset, errors