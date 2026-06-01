from __future__ import annotations

import json
from pathlib import Path

import torch
from PIL import Image

from config import CLASS_NAMES, IMAGE_SIZE, MODELS_DIR, REPORTS_DIR
from data_utils import build_transforms, load_normalization_stats
from models import build_model
from stacking import STACKING_MODEL_NAME, predict_with_stacking


def load_best_model():
    metadata = load_metadata()
    if metadata["best_model"] == STACKING_MODEL_NAME:
        return None, STACKING_MODEL_NAME
    checkpoint = torch.load(MODELS_DIR / "best_model.pt", map_location="cpu", weights_only=False)
    model = build_model(checkpoint["model_name"])
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, checkpoint["model_name"]


def load_metadata():
    with open(MODELS_DIR / "best_model_metadata.json", "r", encoding="utf-8") as file_handle:
        return json.load(file_handle)


def predict_image(image: Image.Image):
    metadata = load_metadata()
    if metadata["best_model"] == STACKING_MODEL_NAME:
        return predict_with_stacking(image)
    model, model_name = load_best_model()
    normalization_stats = load_normalization_stats(REPORTS_DIR / "normalization_stats.json")
    _, eval_transform = build_transforms(normalization_stats)
    tensor = eval_transform(image.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))).unsqueeze(0)

    with torch.no_grad():
        outputs = model(tensor)
        logits = outputs["logits"] if isinstance(outputs, dict) else outputs
        probabilities = torch.softmax(logits, dim=1).squeeze(0).numpy()
        predicted_index = int(probabilities.argmax())

    return {
        "model_name": model_name,
        "predicted_class": CLASS_NAMES[predicted_index],
        "probabilities": {class_name: float(probability) for class_name, probability in zip(CLASS_NAMES, probabilities)},
    }
