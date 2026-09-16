import json
import csv
from pathlib import Path

csv_path = Path(r"D:\vlmod\combined_obstacles.csv")
output_dir = Path(r"D:\vlmod\processed_data")
output_dir.mkdir(exist_ok=True)

active_frames = []
sensor_stats = {}
time_series_by_file = {}

with open(csv_path, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        filename = row["filename"]
        sensor_id = row["sensor_id"]
        is_active = int(row["col1"]) == 1
        
        # Track sequence per file
        if filename not in time_series_by_file:
            time_series_by_file[filename] = []
        time_series_by_file[filename].append([int(row["col1"]), int(row["col2"]), int(row["col3"])])
        
        # Track sensor statistics
        if sensor_id not in sensor_stats:
            sensor_stats[sensor_id] = {"total_frames": 0, "active_frames": 0}
        sensor_stats[sensor_id]["total_frames"] += 1
        
        if is_active:
            active_frames.append(row)
            sensor_stats[sensor_id]["active_frames"] += 1

# 1. Save filtered active-only CSV
active_csv_path = output_dir / "active_obstacles_only.csv"
if active_frames:
    with open(active_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=active_frames[0].keys())
        writer.writeheader()
        writer.writerows(active_frames)

# 2. Save JSON sequence dataset
json_path = output_dir / "sequences.json"
with open(json_path, "w", encoding="utf-8") as f:
    json.dump(time_series_by_file, f, indent=2)

# Print Summary Report
print("=== DATASET SUMMARY REPORT ===")
print(f"Total Frames Processed : {sum(s['total_frames'] for s in sensor_stats.values())}")
print(f"Total Active Frames    : {len(active_frames)}")
print(f"Files Processed        : {len(time_series_by_file)}")
print("\n--- Sensor Breakdown ---")
for sensor, stats in sensor_stats.items():
    pct = (stats['active_frames'] / stats['total_frames']) * 100
    print(f"Sensor '{sensor}': {stats['active_frames']}/{stats['total_frames']} active frames ({pct:.1f}%)")

print("\n--- Exported Files ---")
print(f"1. Active-only CSV : {active_csv_path}")
print(f"2. JSON Sequences  : {json_path}")