# VLMod-Obstacle-2026: Visual-Language Obstacle Matching

A practical implementation and evaluation workflow for the **VLMod-Obstacle-2026** track on the Visual Language Perception (VLP) platform.

The task focuses on matching **textual descriptions with objects in 3D scene data** and generating predictions in the required evaluation format.

---

## 📌 Project Overview

The VLMod-Obstacle-2026 task provides JSON files containing:

* Textual descriptions of objects
* 3D object information
* Object position and spatial information
* Object dimensions
* Orientation
* Color and category information

The goal is to determine whether each object corresponds to each given textual description.

For each test file, the required prediction contains **three binary values for each object**, representing whether the object matches the three descriptions.

---

## 🔍 What I Worked On

I developed a Python-based workflow to process the evaluation data and prepare the required submission files.

The workflow includes:

1. Reading and parsing the JSON test files
2. Extracting object and description information
3. Applying a matching approach
4. Generating prediction `.txt` files
5. Checking and organizing the generated results
6. Preparing the final submission ZIP file
7. Evaluating different matching thresholds

The workflow was applied to **300 test JSON files**, producing **4,220 lines of evaluation data**.

---

## 🧠 Matching and Threshold Experiment

An important part of my work was testing how the matching threshold affected the evaluation results.

I compared two experiments:

| Experiment | Threshold | Precision | Recall | F1 Score |   TP |   FP |   FN |
| ---------- | --------: | --------: | -----: | -------: | ---: | ---: | ---: |
| Initial    |      0.66 |    31.12% | 52.66% |   39.12% | 1614 | 3573 | 1451 |
| Strict     |      0.74 |    39.68% | 20.26% |   26.83% |  621 |  944 | 2444 |

The stricter threshold reduced false positives and increased precision, but it also reduced recall and increased false negatives.

This experiment helped me understand the practical trade-off between **precision and recall** rather than viewing the evaluation metrics only theoretically.

---

## 📊 Final Submission Result

My submitted VLMod-Obstacle-2026 result was:

* **F1 Score:** 26.8251
* **Precision:** 39.6805
* **Recall:** 20.2610
* **True Positives:** 621
* **False Positives:** 944
* **False Negatives:** 2444

The results were obtained from the VLP evaluation platform after submitting the generated predictions.

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
Matching / Similarity Calculation
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

## 📁 Repository Structure

```text
VLMod-3D-Obstacle-Tracker/
│
├── README.md
│
├── main.py
├── predict.py
├── parser.py
├── evaluation.py
├── extract_zip.py
├── parse_stats.py
│
├── combined_obstacles.csv
│
├── result.zip
├── submission.zip
└── submission_v2.zip
```

### Python Files

| File             | Purpose                         |
| ---------------- | ------------------------------- |
| `main.py`        | Main processing workflow        |
| `predict.py`     | Prediction and matching process |
| `parser.py`      | Parsing input data              |
| `evaluation.py`  | Evaluation-related processing   |
| `extract_zip.py` | Extraction and file handling    |
| `parse_stats.py` | Dataset/statistical analysis    |

---

## 📦 Submission Files

The repository contains the generated submission archives used during experimentation:

* `submission.zip` — initial submission
* `submission_v2.zip` — revised submission after changing the matching threshold
* `result.zip` — processed result files

The two submissions allowed me to compare how different matching conditions affected the final evaluation metrics.

---

## 🛠️ Technologies

* **Python**
* **JSON**
* **Pandas**
* **CSV**
* **ZIP file processing**
* **Data preprocessing**
* **Visual-language evaluation**
* **3D spatial data**
* **Computer Vision**
* **Machine Learning evaluation**

---

## 🚀 Running the Workflow

### 1. Clone the Repository

```bash
git clone https://github.com/zainab1kausar338-babyseven/VLMod-3D-Obstacle-Tracker.git
cd VLMod-3D-Obstacle-Tracker
```

### 2. Install Dependencies

```bash
pip install pandas
```

Additional dependencies may be required depending on the individual script being executed.

### 3. Run the Main Workflow

```bash
python main.py
```

Other scripts can be executed individually when needed:

```bash
python predict.py
python parser.py
python evaluation.py
python parse_stats.py
```

---

## 📈 Key Observation

The main observation from my experiments was that changing the matching threshold had a direct effect on the balance between precision and recall.

A lower threshold allowed more potential matches to be detected, resulting in higher recall but also more false positives. Increasing the threshold made the matching condition stricter, reducing false positives while also causing more relevant cases to be missed.

This experiment gave me practical experience with the relationship between **matching criteria, evaluation metrics, and system behavior**.

---

## 🎯 Project Objective

The objective of this project was not only to obtain an evaluation score, but also to understand the complete workflow of a visual-language evaluation task.

Through this work, I practiced:

* Understanding an unfamiliar dataset structure
* Processing structured JSON data
* Designing an automated processing workflow
* Generating evaluation-compatible outputs
* Running experiments with different parameters
* Interpreting precision, recall, F1 score, TP, FP, and FN
* Preparing a reproducible submission

---

## 📌 Current Status

**Status:** Completed evaluation experiment

The current repository contains the Python processing scripts, processed data, and submission files used during the VLMod-Obstacle-2026 task.

Future work could include:

* Improving object-description matching
* Testing additional similarity methods
* Better spatial relationship modeling
* Visualizing matching results
* Experimenting with multimodal models
* Improving precision and recall through better matching strategies

---

## 👩‍💻 Author

**Zainab Kausar**

BS Computer Science Undergraduate
Sarhad University of Science and Technology, Pakistan

**Interests:** Data Science, Machine Learning, Computer Vision, Visual-Language Models, and Data Processing.



