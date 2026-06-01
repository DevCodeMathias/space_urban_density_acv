from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from config import CLASS_NAMES, MODELS_DIR, REPORTS_DIR
from data_utils import build_transforms, create_data_loaders, load_normalization_stats
from models import build_model

STACKING_MODEL_NAME = "urban_meta_stack_v1"
STACKING_BASE_MODELS = ("urban_cnn_v1", "urban_cnn_v2", "urban_cnn_v3")
STACKING_FEATURE_COLUMNS = [
    *(f"{model_name}_p{class_index}" for model_name in STACKING_BASE_MODELS for class_index in range(len(CLASS_NAMES))),
    "mean_red",
    "mean_green",
    "mean_blue",
    "brightness",
    "contrast",
    "vegetation_index",
    "water_index",
    "built_up_proxy",
    "texture",
    "grayness",
]
STACKING_MODEL_PATH = MODELS_DIR / f"{STACKING_MODEL_NAME}.joblib"
STACKING_METADATA_PATH = MODELS_DIR / f"{STACKING_MODEL_NAME}_metadata.json"
TEXTURE_OVERRIDE_THRESHOLD = 15.29023230820106
TEXTURE_OVERRIDE_SOURCE_CLASS = 2
TEXTURE_OVERRIDE_TARGET_CLASS = 1

# Approximate image-only formulas used when the uploaded image has no manifest row.
TEXTURE_COEF = 0.41867265
TEXTURE_INTERCEPT = 7.997307
BUILT_UP_COEF = 2.2721653793346652
BUILT_UP_INTERCEPT = 0.016368313030233006


def load_base_models() -> dict[str, torch.nn.Module]:
    loaded_models = {}
    for model_name in STACKING_BASE_MODELS:
        checkpoint_path = MODELS_DIR / f"{model_name}.pt"
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint nao encontrado para o modelo base {model_name}: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model = build_model(model_name)
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        loaded_models[model_name] = model
    return loaded_models


def build_stacking_pipeline():
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000))


def apply_texture_override(predictions: np.ndarray, feature_frame: pd.DataFrame) -> np.ndarray:
    adjusted_predictions = predictions.copy()
    override_mask = (
        (adjusted_predictions == TEXTURE_OVERRIDE_SOURCE_CLASS)
        & (feature_frame["texture"].to_numpy() <= TEXTURE_OVERRIDE_THRESHOLD)
    )
    adjusted_predictions[override_mask] = TEXTURE_OVERRIDE_TARGET_CLASS
    return adjusted_predictions


def compute_image_visual_features(image: Image.Image) -> dict[str, float]:
    image_array = np.asarray(image.convert("RGB").resize((64, 64)), dtype=np.float32)
    red_mean = float(image_array[:, :, 0].mean())
    green_mean = float(image_array[:, :, 1].mean())
    blue_mean = float(image_array[:, :, 2].mean())
    gray = image_array.mean(axis=2)
    brightness = float(image_array.mean())
    contrast = float(gray.std())
    vegetation_index = float((green_mean - red_mean) / (green_mean + red_mean + 1e-8))
    water_index = float((blue_mean - red_mean) / (blue_mean + red_mean + 1e-8))
    built_up_base = float(
        (red_mean + blue_mean - (2.0 * green_mean))
        / (red_mean + blue_mean + (2.0 * green_mean) + 1e-8)
    )
    built_up_proxy = float((BUILT_UP_COEF * built_up_base) + BUILT_UP_INTERCEPT)
    grad_x = np.diff(gray, axis=1, append=gray[:, -1:])
    grad_y = np.diff(gray, axis=0, append=gray[-1:, :])
    gradient = np.sqrt((grad_x * grad_x) + (grad_y * grad_y))
    texture = float((TEXTURE_COEF * gradient.mean()) + TEXTURE_INTERCEPT)
    grayness = float(1.0 - (np.std([red_mean, green_mean, blue_mean]) / (brightness + 1e-8)))
    return {
        "mean_red": red_mean,
        "mean_green": green_mean,
        "mean_blue": blue_mean,
        "brightness": brightness,
        "contrast": contrast,
        "vegetation_index": vegetation_index,
        "water_index": water_index,
        "built_up_proxy": built_up_proxy,
        "texture": texture,
        "grayness": grayness,
    }


def _collect_split_rows(loader, manifest_index: pd.DataFrame, base_models: dict[str, torch.nn.Module]) -> pd.DataFrame:
    rows = []
    with torch.no_grad():
        for images, labels, file_names in loader:
            probability_map = {}
            for model_name, model in base_models.items():
                logits = model(images)
                probability_map[model_name] = torch.softmax(logits, dim=1).cpu().numpy()
            for row_index, file_name in enumerate(file_names):
                manifest_row = manifest_index.loc[file_name]
                row = {"file_name": file_name, "label_index": int(labels[row_index])}
                for model_name in STACKING_BASE_MODELS:
                    for class_index in range(len(CLASS_NAMES)):
                        row[f"{model_name}_p{class_index}"] = float(probability_map[model_name][row_index, class_index])
                for feature_name in STACKING_FEATURE_COLUMNS[len(STACKING_BASE_MODELS) * len(CLASS_NAMES):]:
                    row[feature_name] = float(manifest_row[feature_name])
                rows.append(row)
    return pd.DataFrame(rows)


def train_and_evaluate_stacking_model(stats) -> dict[str, object]:
    manifest, train_frame, validation_frame, test_frame, train_loader, validation_loader, test_loader = create_data_loaders(stats)
    del train_frame, test_frame, train_loader
    manifest_index = manifest.set_index("file_name")
    base_models = load_base_models()

    validation_df = _collect_split_rows(validation_loader, manifest_index, base_models)
    test_df = _collect_split_rows(test_loader, manifest_index, base_models)

    pipeline = build_stacking_pipeline()
    pipeline.fit(validation_df[STACKING_FEATURE_COLUMNS], validation_df["label_index"])

    validation_pred = pipeline.predict(validation_df[STACKING_FEATURE_COLUMNS])
    validation_pred = apply_texture_override(validation_pred, validation_df)
    test_pred = pipeline.predict(test_df[STACKING_FEATURE_COLUMNS])
    test_pred = apply_texture_override(test_pred, test_df)

    validation_accuracy = accuracy_score(validation_df["label_index"], validation_pred)
    test_accuracy = accuracy_score(test_df["label_index"], test_pred)
    report = classification_report(
        test_df["label_index"],
        test_pred,
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )

    logistic_regression = pipeline.named_steps["logisticregression"]
    parameter_count = int(logistic_regression.coef_.size + logistic_regression.intercept_.size)
    misclassified = test_df.loc[test_df["label_index"].to_numpy() != test_pred, ["file_name"]].copy()
    misclassified["true_label"] = [CLASS_NAMES[index] for index in test_df.loc[misclassified.index, "label_index"]]
    misclassified["predicted_label"] = [CLASS_NAMES[index] for index in test_pred[misclassified.index.to_numpy()]]
    misclassified.to_csv(REPORTS_DIR / f"misclassifications_{STACKING_MODEL_NAME}.csv", index=False)

    metrics = {
        "model_name": STACKING_MODEL_NAME,
        "trainable_parameters": parameter_count,
        "best_validation_loss": None,
        "best_validation_accuracy": float(validation_accuracy),
        "final_train_accuracy": None,
        "final_validation_accuracy": float(validation_accuracy),
        "test_accuracy": float(test_accuracy),
        "test_loss": None,
        "misclassified_examples": int(len(misclassified)),
    }

    MODELS_DIR.mkdir(exist_ok=True)
    REPORTS_DIR.mkdir(exist_ok=True)
    joblib.dump(pipeline, STACKING_MODEL_PATH)
    with open(STACKING_METADATA_PATH, "w", encoding="utf-8") as file_handle:
        json.dump(
            {
                "model_name": STACKING_MODEL_NAME,
                "base_models": list(STACKING_BASE_MODELS),
                "feature_columns": list(STACKING_FEATURE_COLUMNS),
                "texture_override_threshold": TEXTURE_OVERRIDE_THRESHOLD,
                "texture_override_source_class": TEXTURE_OVERRIDE_SOURCE_CLASS,
                "texture_override_target_class": TEXTURE_OVERRIDE_TARGET_CLASS,
            },
            file_handle,
            indent=4,
            ensure_ascii=False,
        )
    with open(REPORTS_DIR / f"metrics_{STACKING_MODEL_NAME}.json", "w", encoding="utf-8") as file_handle:
        json.dump(metrics, file_handle, indent=4, ensure_ascii=False)
    with open(REPORTS_DIR / f"classification_report_{STACKING_MODEL_NAME}.json", "w", encoding="utf-8") as file_handle:
        json.dump(report, file_handle, indent=4, ensure_ascii=False)
    return metrics


def predict_with_stacking(image: Image.Image) -> dict[str, object]:
    pipeline = joblib.load(STACKING_MODEL_PATH)
    base_models = load_base_models()
    normalization_stats = load_normalization_stats(REPORTS_DIR / "normalization_stats.json")
    _, eval_transform = build_transforms(normalization_stats)
    tensor = eval_transform(image.convert("RGB").resize((64, 64))).unsqueeze(0)

    feature_row = {}
    with torch.no_grad():
        for model_name, model in base_models.items():
            logits = model(tensor)
            probabilities = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
            for class_index in range(len(CLASS_NAMES)):
                feature_row[f"{model_name}_p{class_index}"] = float(probabilities[class_index])
    feature_row.update(compute_image_visual_features(image))
    feature_frame = pd.DataFrame([feature_row], columns=STACKING_FEATURE_COLUMNS)

    probabilities = pipeline.predict_proba(feature_frame)[0]
    predicted_index = int(probabilities.argmax())
    adjusted_index = int(apply_texture_override(np.array([predicted_index]), feature_frame)[0])
    if adjusted_index != predicted_index and probabilities[adjusted_index] < probabilities[predicted_index]:
        probabilities[[predicted_index, adjusted_index]] = probabilities[[adjusted_index, predicted_index]]

    return {
        "model_name": STACKING_MODEL_NAME,
        "predicted_class": CLASS_NAMES[adjusted_index],
        "probabilities": {
            class_name: float(probability)
            for class_name, probability in zip(CLASS_NAMES, probabilities)
        },
    }
