import glob
import os

active_files = glob.glob(r"D:\vlmod\result\active\*.txt")

stats = {}
for filepath in active_files:
    filename = os.path.basename(filepath)
    dataset_id = filename.split('_')[0]
    
    with open(filepath, 'r') as f:
        # Count lines containing non-zero numbers
        valid_lines = sum(1 for line in f if any(c in line for c in '123456789'))
        
    stats[dataset_id] = stats.get(dataset_id, 0) + valid_lines

print("Valid Detection Counts per Dataset ID:", stats)