import os
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models import resnet50
from datasets import load_dataset

# -----------------------------
# Device (M4 GPU)
# -----------------------------
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("Using device:", device)

# -----------------------------
# 1. Load Hugging Face metadata
# -----------------------------
dataset = load_dataset("mohanty/PlantVillage", "default")

# -----------------------------
# 2. Path to your downloaded images
# -----------------------------
BASE = "/Users/atulkumar01/PlantVillage-Dataset"

# -----------------------------
# 3. Attach real images
# -----------------------------
def load_image(example):
    relative_path = example["text"]  # raw/color/...jpg
    full_path = os.path.join(BASE, relative_path)
    img = Image.open(full_path).convert("RGB")
    example["image"] = img
    return example

dataset = dataset.map(load_image)

# -----------------------------
# 4. Extract disease from folder name
# -----------------------------
def extract_disease(example):
    relative_path = example["text"]  # raw/color/Raspberry___healthy/xxx.JPG
    folder = relative_path.split("/")[2]  # Raspberry___healthy
    disease = folder.split("___")[1]      # healthy
    example["disease"] = disease
    return example

dataset = dataset.map(extract_disease)

# -----------------------------
# 5. Build label mapping
# -----------------------------
disease_names = list(set(dataset["train"]["disease"]))
disease_to_id = {name: i for i, name in enumerate(disease_names)}

print("Number of diseases:", len(disease_names))

# -----------------------------
# 6. Transforms
# -----------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# -----------------------------
# 7. Collate function
# -----------------------------
def collate_fn(batch):
    images = [transform(item["image"]) for item in batch]
    labels = [disease_to_id[item["disease"]] for item in batch]

    images = torch.stack(images).to(device)
    labels = torch.tensor(labels).to(device)

    return {"image": images, "label": labels}

# -----------------------------
# 8. DataLoaders
# -----------------------------
train_loader = DataLoader(dataset["train"], batch_size=32, shuffle=True, collate_fn=collate_fn)
test_loader = DataLoader(dataset["test"], batch_size=32, collate_fn=collate_fn)

# -----------------------------
# 9. Model (ResNet50)
# -----------------------------
model = resnet50(weights="IMAGENET1K_V2")
model.fc = nn.Linear(model.fc.in_features, len(disease_names))
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# -----------------------------
# 10. Training loop
# -----------------------------
for epoch in range(3):
    model.train()
    for batch in train_loader:
        images = batch["image"]
        labels = batch["label"]

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1} complete")

# -----------------------------
# 11. Evaluation
# -----------------------------
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for batch in test_loader:
        images = batch["image"]
        labels = batch["label"]
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print("Accuracy:", correct / total)
