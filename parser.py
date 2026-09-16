import json
import os


def parse_object(label_string):
    """Parses an individual space-separated object string into a structured dictionary."""
    if isinstance(label_string, str):
        # Split raw KITTI-style space-separated string
        obj = label_string.strip().split()
    elif isinstance(label_string, list):
        obj = label_string
    else:
        return label_string

    return {
        "type": obj[0],
        "field_1": float(obj[1]),
        "field_2": float(obj[2]),
        "field_3": float(obj[3]),
        "bbox": [float(val) for val in obj[4:8]],
        "height": float(obj[8]),
        "width": float(obj[9]),
        "length": float(obj[10]),
        "x": float(obj[11]),
        "y": float(obj[12]),
        "z": float(obj[13]),
        "rotation": float(obj[14]),
        "color": obj[15] if len(obj) > 15 else "unknown",
    }


def parse_annotation(annotation):
    """Parses annotations by accessing object candidates inside test_data."""
    test_data = annotation.get("test_data", {})

    if isinstance(test_data, list):
        raw_objects = test_data
    elif isinstance(test_data, dict):
        raw_objects = (
            test_data.get("label_3")
            or test_data.get("labels")
            or test_data.get("objects")
            or test_data.get("candidates", [])
        )
    else:
        raw_objects = []

    parsed_objects = [parse_object(obj) for obj in raw_objects]

    return {
        "description": annotation.get("public_description", []),
        "objects": parsed_objects,
    }


if __name__ == "__main__":
    sample_path = r"D:\vlmod\MonoMulti3D\test"
    files = [f for f in os.listdir(sample_path) if f.endswith(".json")]

    if files:
        file_to_test = os.path.join(sample_path, files[0])
        with open(file_to_test, "r", encoding="utf-8") as f:
            data = json.load(f)

        parsed_data = parse_annotation(data)

        print("--- Parser Test Output ---")
        print(f"File Tested: {files[0]}")
        print(f"Descriptions Found: {len(parsed_data['description'])}")
        print(f"Total Objects Parsed: {len(parsed_data['objects'])}")
        if parsed_data["objects"]:
            print("First Parsed Object:", parsed_data["objects"][0])
    else:
        print("No JSON test files found in destination.")