import os
import json
import zipfile
import torch
from torch import nn
from transformers import DistilBertTokenizer, DistilBertModel

# Define paths
TEST_DIR = r"D:\vlmod\MonoMulti3D\test"
MODEL_PATH = r"D:\vlmod\text3d_matching_model.pth"
RESULT_DIR = r"D:\vlmod\result"
ZIP_PATH = r"D:\vlmod\result.zip"

os.makedirs(RESULT_DIR, exist_ok=True)

# Define the Model architecture matching your training checkpoint shapes
class Text3DMatchingModel(nn.Module):
    def __init__(self, text_dim=768, feat_dim=9):
        super().__init__()
        self.text_encoder = DistilBertModel.from_pretrained("distilbert-base-uncased")
        for param in self.text_encoder.parameters():
            param.requires_grad = False
            
        self.object_encoder = nn.Sequential(
            nn.Linear(feat_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128)
        )
        
        self.text_projection = nn.Sequential(
            nn.Linear(text_dim, 128)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1)
        )

    def forward(self, input_ids, attention_mask, obstacle_features):
        text_out = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        text_emb = text_out.last_hidden_state[:, 0, :] # [CLS] token
        text_emb = self.text_projection(text_emb)
        
        obs_emb = self.object_encoder(obstacle_features)
        
        combined = torch.cat([text_emb, obs_emb], dim=1)
        score = self.classifier(combined).squeeze(-1)
        
        return score

print("Loading tokenizer and trained model...")
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
device = torch.device("cpu")

model = Text3DMatchingModel()

# Load checkpoint
checkpoint = torch.load(MODEL_PATH, map_location=device)
if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
    state_dict = checkpoint["model_state_dict"]
else:
    state_dict = checkpoint

model.load_state_dict(state_dict)
model.to(device)
model.eval()

json_files = [f for f in os.listdir(TEST_DIR) if f.endswith(".json")]
print(f"Found {len(json_files)} test JSON files. Generating predictions with batch optimization...")

THRESHOLD = 0.4
BATCH_SIZE = 32  # Process multiple descriptions/features at once

with torch.no_grad():
    for json_file in json_files:
        json_path = os.path.join(TEST_DIR, json_file)
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        descriptions = data.get("public_description", [])
        test_data = data.get("test_data", [])
        
        if not descriptions or not test_data:
            base_name = json_file.replace(".json", ".txt")
            with open(os.path.join(RESULT_DIR, base_name), "w", encoding="utf-8") as out_f:
                out_f.write("")
            continue

        # Pre-tokenize all descriptions at once for efficiency
        encodings = tokenizer(
            descriptions, 
            padding="max_length", 
            truncation=True, 
            max_length=32, 
            return_tensors="pt"
        )
        desc_input_ids = encodings["input_ids"].to(device)
        desc_attention_mask = encodings["attention_mask"].to(device)
        
        # Pre-extract text embeddings once per file (since descriptions don't change per object)
        text_outputs = []
        for i in range(0, len(descriptions), BATCH_SIZE):
            batch_ids = desc_input_ids[i:i+BATCH_SIZE]
            batch_mask = desc_attention_mask[i:i+BATCH_SIZE]
            text_out = model.text_encoder(input_ids=batch_ids, attention_mask=batch_mask)
            text_emb = text_out.last_hidden_state[:, 0, :]
            text_emb = model.text_projection(text_emb)
            text_outputs.append(text_emb)
        text_embs = torch.cat(text_outputs, dim=0) # Shape: [num_desc, 128]

        predictions_lines = []
        
        for obj_str in test_data:
            parts = obj_str.split()
            try:
                features = [float(p) for p in parts[2:11]]
            except ValueError:
                features = [0.0] * 9
                
            obs_tensor = torch.tensor([features], dtype=torch.float32).to(device)
            # Expand obstacle tensor to match number of descriptions for parallel scoring
            obs_expanded = obs_tensor.expand(len(descriptions), -1)
            obs_embs = model.object_encoder(obs_expanded) # Shape: [num_desc, 128]
            
            # Combine and score all descriptions in one batch pass
            combined = torch.cat([text_embs, obs_embs], dim=1)
            scores = model.classifier(combined).squeeze(-1)
            
            match_flags = [1 if s >= THRESHOLD else 0 for s in scores.tolist()]
            
            while len(match_flags) < 3:
                match_flags.append(0)
                
            predictions_lines.append(f"{match_flags[0]} {match_flags[1]} {match_flags[2]}")
            
        base_name = json_file.replace(".json", ".txt")
        out_txt_path = os.path.join(RESULT_DIR, base_name)
        with open(out_txt_path, "w", encoding="utf-8") as out_f:
            out_f.write("\n".join(predictions_lines) + "\n")

print("All prediction text files generated successfully.")

print("Creating submission.zip...")
with zipfile.ZipFile(ZIP_PATH, 'w', zipfile.ZIP_DEFLATED) as zipf:
    for root, dirs, files in os.walk(RESULT_DIR):
        for file in files:
            file_path = os.path.join(root, file)
            arcname = file
            zipf.write(file_path, arcname)

print(f"Submission zip ready at: {ZIP_PATH}")