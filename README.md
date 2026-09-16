
# VLMod: Multi-Camera Spatial Obstacle Detection & Vector Tracking

A lightweight, automated processing pipeline for **multi-camera spatial obstacle detection, frame-level vector extraction, and temporal track annotation** across distributed video streams.

VLMod processes detection result files, filters inactive/background entries, analyzes detection statistics, and consolidates active detections into a structured CSV dataset suitable for downstream machine learning and computer vision workflows.

---

## 🔍 Features

### Multi-View Parsing

Extracts timestamped obstacle trajectory information from detection files following the naming convention:

```text
Dataset_Camera_Res_StartTS_EndTS_ObstacleID
```

### Automated Filtering

Separates active obstacle detections from zero-padded/background entries.

### Dataset Analytics

Generates detection statistics, including:

* Total detection files
* Active detection files
* Active frame detections
* Detection counts by dataset
* Detection density information

### CSV Exporting

Consolidates multiple processed `.txt` detection files into a single CSV file for use in:

* PyTorch
* TensorFlow
* OpenCV
* Pandas
* Machine learning pipelines
* Data visualization

---

## 📁 Repository Structure

```text
vlmod/
│
├── result/
│   ├── active/
│   │   └── # Processed active detection files
│   │
│   └── active_detections_summary.csv
│       # Consolidated CSV dataset export
│
├── parse_stats.py
│   # Dataset statistics and detection-count analysis
│
├── export_csv.py
│   # Converts detection TXT files into a consolidated CSV
│
└── README.md
    # Project documentation
```

---

## ⚙️ Requirements

### Python

Python **3.8+** is recommended.

### Python Dependencies

Install the required dependency using:

```bash
pip install pandas
```

---

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone (https://github.com/zainab1kausar338-babyseven/VLMod-3D-Obstacle-Tracker)
cd vlmod
```

---

### 2. Install Dependencies

```bash
pip install pandas
```

---

### 3. Analyze Detection Statistics

Run:

```bash
python parse_stats.py
```

The script analyzes the detection result files and reports active frame detections grouped by dataset.

---

### 4. Create the Active Detection Directory

On Windows PowerShell:

```powershell
New-Item -ItemType Directory -Path "D:\vlmod\result\active" -Force
```

---

### 5. Move Active Detection Files

The following PowerShell command identifies files containing non-zero detection entries and moves them into the `active` directory:

```powershell
Get-ChildItem -Path "D:\vlmod\result\*.txt" |
Where-Object {
    Select-String -Path $_.FullName -Pattern "[1-9]"
} |
Move-Item -Destination "D:\vlmod\result\active"
```

> **Note:** Make sure your result directory path matches your local project location before running the command.

---

## 📊 Detection Data Format

VLMod extracts metadata from the detection filenames and combines it with frame-level detection data.

The exported file is:

```text
result/active_detections_summary.csv
```

### CSV Fields

| Field         | Description                              | Example            |
| ------------- | ---------------------------------------- | ------------------ |
| `dataset_id`  | Unique identifier of the dataset capture | `145044`           |
| `camera`      | Camera sensor identifier                 | `fa2sd4a06W152AIR` |
| `resolution`  | Input frame vertical resolution          | `420`              |
| `start_ts`    | Tracking epoch start timestamp           | `1626155724`       |
| `end_ts`      | Tracking epoch end timestamp             | `1626155908`       |
| `obstacle_id` | Assigned spatial tracking ID             | `243`              |
| `frame_index` | Position of the frame inside the stream  | `1`                |
| `data`        | Extracted coordinate/bounding vector     | `1 0 0`            |

---

## 🔄 Processing Pipeline

The overall processing workflow is:

```text
Raw Detection Files
        │
        ▼
Filename / Metadata Parsing
        │
        ▼
Frame-Level Detection Extraction
        │
        ▼
Active vs. Empty Detection Filtering
        │
        ▼
Dataset Statistics
        │
        ▼
CSV Consolidation
        │
        ▼
active_detections_summary.csv
        │
        ▼
ML / Computer Vision Pipeline
```

---

## 🧹 Active Detection Filtering

VLMod distinguishes between:

### Empty / Background Entries

Entries containing only zero-padded values, for example:

```text
0 0 0
```

These entries represent frames without an active obstacle detection.

### Active Entries

Entries containing non-zero values, for example:

```text
1 0 0
```

These records are retained as active detections for further analysis.

This filtering reduces unnecessary background records and produces a more compact dataset for downstream processing.

---

## 📈 Dataset Analytics

`parse_stats.py` can be used to inspect detection activity across datasets.

Example output concept:

```text
Dataset ID: 145044
Active Files: 12
Active Frame Detections: 1,284
```

The statistics can help identify:

* Detection density
* Dataset activity
* Number of active obstacle tracks
* Distribution of detections across datasets
* Potentially sparse or inactive captures

---

## 📤 CSV Export

`export_csv.py` consolidates processed detection files into a single tabular dataset.

Run:

```bash
python export_csv.py
```

The resulting file is:

```text
result/active_detections_summary.csv
```

This format makes the processed data easier to inspect, analyze, visualize, and integrate into machine learning workflows.

---

## 🤖 Machine Learning Use

The generated CSV can serve as an intermediate dataset for tasks such as:

* Spatial obstacle detection
* Object trajectory analysis
* Multi-camera tracking
* Temporal sequence analysis
* Vector-based representation learning
* Detection-density analysis
* Computer vision experiments

It can be loaded with Pandas:

```python
import pandas as pd

df = pd.read_csv("result/active_detections_summary.csv")

print(df.head())
print(df.shape)
```

---

## 🧪 Example Data

```text
dataset_id,camera,resolution,start_ts,end_ts,obstacle_id,frame_index,data
145044,fa2sd4a06W152AIR,420,1626155724,1626155908,243,1,"1 0 0"
145044,fa2sd4a06W152AIR,420,1626155724,1626155908,243,2,"1 0 0"
145044,fa2sd4a06W152AIR,420,1626155724,1626155908,243,3,"0 1 0"
```

---

## 💻 Technologies

* **Python**
* **Pandas**
* **CSV**
* **PowerShell**
* **Computer Vision**
* **Multi-Camera Data Processing**
* **Spatial Tracking**
* **Machine Learning Data Preparation**

---

## 🎯 Project Objective

The goal of VLMod is to provide a lightweight preprocessing layer between raw multi-camera detection outputs and downstream machine learning or computer vision systems.

Instead of manually inspecting large numbers of detection files, the pipeline automates:

1. Detection file parsing
2. Metadata extraction
3. Active detection filtering
4. Dataset-level statistics
5. Frame-level organization
6. CSV dataset generation

This creates a structured representation of multi-camera spatial detection data that can be used for further experimentation and model development.

---

## 📌 Current Status

**Project stage:** Data processing and dataset preparation

Current components:

* [x] Detection file parsing
* [x] Metadata extraction
* [x] Active detection filtering
* [x] Dataset statistics
* [x] CSV consolidation
* [x] Training-ready tabular export

Potential future extensions:

* [ ] Automated trajectory visualization
* [ ] Multi-camera trajectory synchronization
* [ ] Vector normalization
* [ ] Track continuity analysis
* [ ] Visualization dashboard
* [ ] PyTorch dataset loader
* [ ] Automated model-training pipeline

---

## 📜 Citation

If you use this project in academic or research work, you may cite it as:

```bibtex
@article{vlmod2026,
  title={VLMod: Multi-Camera Spatial Obstacle Detection and Vector Tracking},
  author={VLMod Development Team},
  year={2026}
}
```

---

## 📄 License

Add the appropriate license for your project before publishing the repository.

---

## 👩‍💻 Author

**Zainab Kausar**

Computer Science Undergraduate
Interests: Data Science, Machine Learning, Computer Vision, and Data Processing
