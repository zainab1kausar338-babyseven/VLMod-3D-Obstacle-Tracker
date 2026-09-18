import os
import json
import random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertModel, DistilBertTokenizer

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


# ============================================================
# 1. CONFIGURATION (OPTIMIZED FOR FAST CPU TRAINING)
# ============================================================

DATASET_PATH = r"D:\vlmod\MonoMulti3D\test"

MODEL_NAME = "distilbert-base-uncased"

BATCH_SIZE = 8
EPOCHS = 1             # Reduced to 1 epoch for quick testing
LEARNING_RATE = 2e-5

MAX_LENGTH = 64
EMBEDDING_DIM = 128

MAX_SAMPLES = 500      # Truncate dataset size to prevent CPU freeze

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 60)
print("MULTIMODAL TEXT-TO-3D MATCHING NETWORK (FAST MODE)")
print("=" * 60)

print("Dataset:", DATASET_PATH)
print("Device :", DEVICE)


# ============================================================
# 2. RANDOM SEED
# ============================================================

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ============================================================
# 3. CLASS AND COLOR ENCODING
# ============================================================

CLASS_MAP = {
    "car": 0,
    "truck": 1,
    "van": 2,
    "bus": 3,
    "cyclist": 4,
    "bicycle": 5,
    "motorcycle": 6,
    "pedestrian": 7,
    "person": 8,
    "unknown": 9
}


COLOR_MAP = {
    "white": 0,
    "black": 1,
    "red": 2,
    "blue": 3,
    "silver": 4,
    "gray": 5,
    "grey": 6,
    "yellow": 7,
    "green": 8,
    "orange": 9,
    "brown": 10,
    "unknown": 11
}


# ============================================================
# 4. PARSE 3D OBJECT
# ============================================================

def parse_object(label_string):

    if not isinstance(label_string, str):
        return None

    parts = label_string.strip().split()

    if len(parts) < 15:
        return None

    try:

        return {
            "class_name": parts[0].lower(),
            "truncated": float(parts[1]),
            "occluded": float(parts[2]),
            "alpha": float(parts[3]),
            "bbox_left": float(parts[4]),
            "bbox_top": float(parts[5]),
            "bbox_right": float(parts[6]),
            "bbox_bottom": float(parts[7]),
            "height": float(parts[8]),
            "width": float(parts[9]),
            "length": float(parts[10]),
            "x": float(parts[11]),
            "y": float(parts[12]),
            "z": float(parts[13]),
            "rotation": float(parts[14]),
            "color": (
                parts[15].lower()
                if len(parts) > 15
                else "unknown"
            )
        }

    except (ValueError, IndexError):

        return None


# ============================================================
# 5. LOAD JSON FILES
# ============================================================

def load_json_files(folder_path):

    examples = []

    if not os.path.exists(folder_path):
        raise FileNotFoundError(
            f"Dataset folder not found:\n{folder_path}"
        )

    json_files = [
        f
        for f in os.listdir(folder_path)
        if f.lower().endswith(".json")
    ]

    print(f"\nFound {len(json_files)} JSON files.")

    for filename in json_files:

        filepath = os.path.join(folder_path, filename)

        try:

            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)

            descriptions = data.get("public_description", [])
            raw_objects = data.get("test_data", [])

            parsed_objects = []

            for raw_object in raw_objects:
                object_data = parse_object(raw_object)
                if object_data is not None:
                    parsed_objects.append(object_data)

            for obj in parsed_objects:
                for description in descriptions:
                    examples.append({
                        "object": obj,
                        "description": str(description)
                    })

        except Exception as e:
            print(f"Could not process {filename}: {e}")

    return examples


# ============================================================
# 6. WEAK LABEL GENERATION
# ============================================================

def generate_label(obj, description):

    desc = description.lower()
    class_name = obj["class_name"]
    color = obj["color"]

    class_groups = {
        "car": ["car", "vehicle", "sedan", "suv", "automobile", "auto", "crossover"],
        "truck": ["truck", "van", "pickup", "lorry", "freight"],
        "bus": ["bus", "coach"],
        "cyclist": ["bike", "bicycle", "motorcycle", "cyclist", "rider", "scooter"],
        "bicycle": ["bike", "bicycle", "cyclist", "rider"],
        "motorcycle": ["motorcycle", "bike", "rider"],
        "pedestrian": ["pedestrian", "person", "walker", "man", "woman", "people", "child"],
        "person": ["pedestrian", "person", "walker", "man", "woman", "people", "child"]
    }

    class_match = False
    possible_words = class_groups.get(class_name, [class_name])
    for word in possible_words:
        if word in desc:
            class_match = True
            break

    color_match = False
    if color != "unknown" and color in desc:
        color_match = True

    x = obj["x"]
    left_match = (x < -1.0 and any(w in desc for w in ["left", "west", "leftside"]))
    right_match = (x > 1.0 and any(w in desc for w in ["right", "east", "rightside"]))
    center_match = (-1.0 <= x <= 1.0 and any(w in desc for w in ["center", "middle"]))
    spatial_x_match = (left_match or right_match or center_match)

    z = obj["z"]
    near_match = (z < 15.0 and any(w in desc for w in ["near", "close", "front", "foreground"]))
    far_match = (z > 18.0 and any(w in desc for w in ["far", "distant", "back", "background"]))
    depth_match = (near_match or far_match)

    if class_match and (color_match or spatial_x_match or depth_match):
        return 1

    return 0


# ============================================================
# 7. CREATE TRAINING DATA (WITH SPEED TRUNCATION)
# ============================================================

def create_training_data(raw_examples):

    dataset = []

    for example in raw_examples:
        obj = example["object"]
        description = example["description"]
        label = generate_label(obj, description)

        dataset.append({
            "text": description,
            "features": [
                obj["x"], obj["y"], obj["z"],
                obj["height"], obj["width"], obj["length"],
                obj["rotation"],
                CLASS_MAP.get(obj["class_name"], CLASS_MAP["unknown"]),
                COLOR_MAP.get(obj["color"], COLOR_MAP["unknown"])
            ],
            "label": float(label)
        })

    return dataset


# ============================================================
# 8. NORMALIZE NUMERICAL FEATURES
# ============================================================

def normalize_features(dataset):

    features = np.array(
        [item["features"] for item in dataset],
        dtype=np.float32
    )

    mean = features.mean(axis=0)
    std = features.std(axis=0)
    std[std < 1e-6] = 1.0

    for item in dataset:
        item["features"] = (
            (np.array(item["features"], dtype=np.float32) - mean) / std
        ).tolist()

    return dataset


# ============================================================
# 9. PYTORCH DATASET
# ============================================================

class MonoMulti3DDataset(Dataset):

    def __init__(self, data_samples, tokenizer, max_length=64):
        self.samples = data_samples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        encoded_text = self.tokenizer(
            sample["text"],
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt"
        )

        return {
            "input_ids": encoded_text["input_ids"].squeeze(0),
            "attention_mask": encoded_text["attention_mask"].squeeze(0),
            "features": torch.tensor(sample["features"], dtype=torch.float32),
            "label": torch.tensor(sample["label"], dtype=torch.float32)
        }


# ============================================================
# 10. MULTIMODAL MODEL
# ============================================================

class Text3DMatchingModel(nn.Module):

    def __init__(self, text_model_name=MODEL_NAME, feature_dim=9, embedding_dim=128):
        super().__init__()

        self.text_encoder = DistilBertModel.from_pretrained(text_model_name)
        text_hidden_size = self.text_encoder.config.hidden_size

        self.text_projection = nn.Sequential(
            nn.Linear(text_hidden_size, embedding_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        self.object_encoder = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.ReLU(),
            nn.Linear(64, embedding_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim * 2, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )

    def forward(self, input_ids, attention_mask, obj_features):
        text_outputs = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        cls_embedding = text_outputs.last_hidden_state[:, 0, :]
        text_emb = self.text_projection(cls_embedding)

        obj_emb = self.object_encoder(obj_features)

        combined = torch.cat((text_emb, obj_emb), dim=1)
        logits = self.classifier(combined)

        return logits.squeeze(-1)


# ============================================================
# 11. LOAD REAL DATA & TRUNCATE FOR SPEED
# ============================================================

raw_examples = load_json_files(DATASET_PATH)

print(f"\nTotal raw pairs found: {len(raw_examples)}")

# Truncate dataset size to speed up CPU processing
if len(raw_examples) > MAX_SAMPLES:
    raw_examples = raw_examples[:MAX_SAMPLES]
    print(f"Truncated dataset to first {MAX_SAMPLES} samples for fast testing.")


# ============================================================
# 12. CREATE LABELS
# ============================================================

dataset = create_training_data(raw_examples)
print("Training examples:", len(dataset))


# ============================================================
# 13. LABEL DISTRIBUTION
# ============================================================

positive = sum(item["label"] == 1 for item in dataset)
negative = sum(item["label"] == 0 for item in dataset)

print(f"Positive examples: {positive}")
print(f"Negative examples: {negative}")

if positive == 0 or negative == 0:
    raise RuntimeError("Only one class generated. Adjust rules or dataset.")


# ============================================================
# 14. NORMALIZE FEATURES
# ============================================================

dataset = normalize_features(dataset)


# ============================================================
# 15. TRAIN / VALIDATION SPLIT
# ============================================================

labels = [int(item["label"]) for item in dataset]

train_data, val_data = train_test_split(
    dataset,
    test_size=0.2,
    random_state=SEED,
    stratify=labels
)

print(f"Training samples  : {len(train_data)}")
print(f"Validation samples: {len(val_data)}")


# ============================================================
# 16. TOKENIZER & DATALOADERS
# ============================================================

print("\nLoading DistilBERT tokenizer...")
tokenizer = DistilBertTokenizer.from_pretrained(MODEL_NAME)

train_dataset = MonoMulti3DDataset(train_data, tokenizer, MAX_LENGTH)
val_dataset = MonoMulti3DDataset(val_data, tokenizer, MAX_LENGTH)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)


# ============================================================
# 17. MODEL, LOSS, OPTIMIZER
# ============================================================

print("Loading multimodal model...")
model = Text3DMatchingModel().to(DEVICE)

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)


# ============================================================
# 18. TRAINING LOOP
# ============================================================

print("\n" + "=" * 60)
print("STARTING TRAINING")
print("=" * 60)

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0.0

    for batch in train_loader:
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        features = batch["features"].to(DEVICE)
        labels_tensor = batch["label"].to(DEVICE)

        optimizer.zero_grad()
        logits = model(input_ids, attention_mask, features)
        loss = criterion(logits, labels_tensor)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch + 1}/{EPOCHS} | Loss: {avg_loss:.4f}")


# ============================================================
# 19. VALIDATION & METRICS
# ============================================================

print("\n" + "=" * 60)
print("VALIDATION")
print("=" * 60)

model.eval()
predictions = []
true_labels = []

with torch.no_grad():
    for batch in val_loader:
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        features = batch["features"].to(DEVICE)
        labels_tensor = batch["label"].to(DEVICE)

        logits = model(input_ids, attention_mask, features)
        probs = torch.sigmoid(logits)
        preds = (probs >= 0.5).long()

        predictions.extend(preds.cpu().numpy())
        true_labels.extend(labels_tensor.cpu().numpy())

accuracy = accuracy_score(true_labels, predictions)
precision = precision_score(true_labels, predictions, zero_division=0)
recall = recall_score(true_labels, predictions, zero_division=0)
f1 = f1_score(true_labels, predictions, zero_division=0)

print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")


# ============================================================
# 20. SAVE MODEL
# ============================================================

MODEL_OUTPUT = "text3d_matching_model.pth"
torch.save({
    "model_state_dict": model.state_dict(),
    "model_name": MODEL_NAME,
    "feature_dim": 9,
    "embedding_dim": EMBEDDING_DIM,
    "max_length": MAX_LENGTH,
    "class_map": CLASS_MAP,
    "color_map": COLOR_MAP
}, MODEL_OUTPUT)

print(f"\nModel saved to: {os.path.abspath(MODEL_OUTPUT)}")
print("Training complete.")