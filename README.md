# VLMod-Obstacle-2026: Visual-Language Obstacle Matching

A practical Python-based implementation and evaluation workflow for the **VLMod-Obstacle-2026** track on the Visual Language Perception (VLP) platform.

The project focuses on matching **textual descriptions with objects in 3D scene data**, generating evaluation-compatible predictions, and analyzing the effect of different matching thresholds on precision, recall, and F1 score.

---

## 📌 Project Overview

The VLMod-Obstacle-2026 task involves structured scene data containing information about objects and their corresponding textual descriptions.

The data includes information such as:

* Textual object descriptions
* 3D object information
* Object position and spatial information
* Object dimensions
* Orientation
* Color
* Object category

The objective is to determine whether each object corresponds to each given textual description.

For each test file, the prediction output contains **binary values indicating whether an object matches the provided descriptions**.

---

## 🎯 Project Objectives

The main objectives of this project were to:

1. Understand the structure of the VLMod obstacle dataset.
2. Process structured JSON and CSV data.
3. Extract object and description information programmatically.
4. Develop an automated matching and prediction workflow.
5. Generate evaluation-compatible `.txt` prediction files.
6. Validate and organize generated outputs.
7. Create submission ZIP files.
8. Experiment with different matching thresholds.
9. Analyze precision, recall, F1 score, TP, FP, and FN.

---

## 🔍 What I Worked On

I developed a Python-based workflow to process the evaluation data and prepare the required submission files.

The workflow includes:

1. Reading and parsing JSON test files
2. Extracting object and description information
3. Processing available spatial and object attributes
4. Applying an object-description matching approach
5. Generating binary predictions
6. Creating prediction `.txt` files
7. Checking and organizing generated results
8. Preparing submission ZIP files
9. Evaluating different matching thresholds

The workflow was applied to **300 test JSON files**, producing approximately **4,220 lines of evaluation data**.

---

## ⚙️ Processing Pipeline

```text
Test JSON Files
       │
       ▼
JSON Parsing
       │
       ▼
Extract Objects & Text Descriptions
       │
       ▼
Object-Description Matching
       │
       ▼
Similarity / Matching Calculation
       │
       ▼
Apply Matching Threshold
       │
       ▼
Generate Binary Predictions
       │
       ▼
Create TXT Files
       │
       ▼
Validate Output
       │
       ▼
Create Submission ZIP
       │
       ▼
VLP Evaluation
       │
       ▼
Precision / Recall / F1
```

---

## 🧠 Matching and Threshold Experiment

A major part of the project was experimenting with the **matching threshold** used to determine whether an object and textual description should be considered a match.

Two experiments were compared:

| Experiment | Threshold | Precision | Recall | F1 Score |    TP |    FP |    FN |
| ---------- | --------: | --------: | -----: | -------: | ----: | ----: | ----: |
| Initial    |      0.66 |    31.12% | 52.66% |   39.12% | 1,614 | 3,573 | 1,451 |
| Strict     |      0.74 |    39.68% | 20.26% |   26.83% |   621 |   944 | 2,444 |

The stricter threshold produced **higher precision and fewer false positives**, while recall decreased and false negatives increased.

This experiment provided practical experience with the relationship between **matching thresholds, precision, recall, and system behavior**.

---

## 📊 Final Submission Result

The final submitted result on the VLP evaluation platform was:

| Metric          |      Result |
| --------------- | ----------: |
| F1 Score        | **26.8251** |
| Precision       | **39.6805** |
| Recall          | **20.2610** |
| True Positives  |     **621** |
| False Positives |     **944** |
| False Negatives |   **2,444** |

These values were obtained from the VLP evaluation platform after submitting the generated prediction files.

---

## 📈 Evaluation Metrics

### Precision

Measures the proportion of predicted matches that were correct.

```text
Precision = TP / (TP + FP)
```

### Recall

Measures the proportion of relevant matches that were successfully identified.

```text
Recall = TP / (TP + FN)
```

### F1 Score

Combines precision and recall into a single metric.

```text
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```

These metrics were used to understand how changes to the matching threshold affected the prediction results.

---

## 📁 Repository Structure

The repository is organized into source code, processed data, and experiment outputs.

```text
VLMod-3D-Obstacle-Tracker/
│
├── README.md
│
├── combined_obstacles.csv
│
├── src/
│   ├── main.py
│   ├── predict.py
│   ├── parser.py
│   ├── evaluation.py
│   ├── extract_zip.py
│   └── parse_stats.py
│
└── zip/
    ├── submission.zip
    ├── submission_v2.zip
    └── result.zip
```

### `src/`

Contains the Python source code used for dataset processing, prediction, matching, evaluation, and statistical analysis.

| File             | Purpose                                    |
| ---------------- | ------------------------------------------ |
| `main.py`        | Main processing workflow                   |
| `predict.py`     | Prediction and object-description matching |
| `parser.py`      | Input data parsing                         |
| `evaluation.py`  | Evaluation and metric processing           |
| `extract_zip.py` | ZIP extraction and file handling           |
| `parse_stats.py` | Dataset statistics and analysis            |

### `zip/`

Contains the generated archives used during the experiments and evaluation.

| File                | Description                                                        |
| ------------------- | ------------------------------------------------------------------ |
| `submission.zip`    | Initial submission archive                                         |
| `submission_v2.zip` | Revised submission generated after changing the matching threshold |
| `result.zip`        | Processed result files                                             |

### `combined_obstacles.csv`

Contains combined processed obstacle information used during the project.

---

## 📦 Submission Files

The experiment generated multiple archives during the evaluation process.

### `submission.zip`

The initial submission containing the generated prediction files.

### `submission_v2.zip`

A revised submission generated after modifying the matching threshold.

### `result.zip`

Contains processed result files generated during the workflow.

All three archives are stored in the:

```text
zip/
```

directory.

---

## 🛠️ Technologies

* **Python**
* **Pandas**
* **JSON**
* **CSV**
* **ZIP file processing**
* **Data preprocessing**
* **Data analysis**
* **3D spatial data**
* **Computer Vision**
* **Visual-Language evaluation**
* **Machine Learning evaluation**

---

## 🚀 Running the Workflow

### 1. Clone the Repository

```bash
git clone https://github.com/zainab1kausar338-babyseven/VLMod-3D-Obstacle-Tracker.git
cd VLMod-3D-Obstacle-Tracker
```

### 2. Navigate to the Source Folder

```bash
cd src
```

### 3. Install Dependencies

The primary dependency used in the processing workflow is:

```bash
pip install pandas
```

Additional dependencies may be required depending on the individual script and dataset configuration.

### 4. Run the Main Workflow

```bash
python main.py
```

Individual components can also be executed when required:

```bash
python parser.py
python predict.py
python evaluation.py
python parse_stats.py
```

---

## 🧪 Experiments

The project involved several stages of experimentation, including:

* Dataset structure analysis
* JSON parsing
* Object extraction
* Description processing
* Data preprocessing
* Object-description matching
* Matching threshold adjustment
* Prediction generation
* Evaluation metric analysis
* Submission generation

The threshold experiment was particularly useful for understanding how stricter matching criteria affect false positives, false negatives, precision, and recall.

---

## 📚 What I Learned

Through this project, I gained practical experience in:

* Working with an unfamiliar research dataset
* Understanding structured JSON and CSV data
* Building Python data-processing workflows
* Processing 3D spatial object information
* Working with visual-language evaluation data
* Designing automated prediction pipelines
* Generating evaluation-compatible outputs
* Performing parameter-based experiments
* Interpreting precision, recall, and F1 score
* Understanding TP, FP, and FN
* Preparing reproducible submission files
* Analyzing how matching criteria influence system behavior

---

## 🔮 Future Improvements

Potential future improvements include:

* More advanced object-description matching
* Additional similarity methods
* Improved spatial relationship modeling
* Better temporal information handling
* Visualization of object and matching results
* Improved object-category and attribute matching
* Multimodal model integration
* Automated evaluation reports
* Further experimentation with matching strategies

---

## 📌 Project Status

**Status: Completed evaluation experiment**

The repository contains the Python processing scripts, processed obstacle data, and experimental submission archives used during the **VLMod-Obstacle-2026** task.

The project represents a practical exploration of:

```text
3D Spatial Data
       ↓
Data Parsing
       ↓
Object-Description Matching
       ↓
Prediction Generation
       ↓
Evaluation
       ↓
Submission
```

---

## 👩‍💻 Author

**Zainab Kausar**

BS Computer Science Undergraduate
Sarhad University of Science and Technology, Pakistan

### Interests

* Data Science
* Machine Learning
* Computer Vision
* Visual-Language Models
* Data Processing
* Database Systems

### GitHub

https://github.com/zainab1kausar338-babyseven

---

## ⭐ Project Focus

This project demonstrates practical experience in taking a research-oriented dataset through a complete processing and evaluation workflow:

**Raw Data → Parsing → Matching → Prediction → Evaluation → Submission**

It combines **Python programming, structured data processing, 3D spatial information, object-description matching, and evaluation analysis** in a single practical project.
