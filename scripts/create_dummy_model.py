import torch
import timm
import json
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "checkpoints"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

class_names = [
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy"
]

num_classes = len(class_names)
model = timm.create_model("efficientnet_b0", pretrained=False, num_classes=num_classes)

torch.save({
    "epoch": 0,
    "model_state_dict": model.state_dict(),
    "val_acc": 0.0,
    "val_loss": 0.0,
    "class_names": class_names,
    "num_classes": num_classes,
}, str(OUTPUT_DIR / "best_model.pth"))

with open(str(OUTPUT_DIR / "class_names.json"), "w") as f:
    json.dump(class_names, f, indent=2)

print("Dummy checkpoint created at checkpoints/best_model.pth")
