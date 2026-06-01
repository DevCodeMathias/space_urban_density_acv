from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, classification_report, confusion_matrix
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

from config import (
    CLASS_NAMES,
    LABEL_SMOOTHING,
    LEARNING_RATE,
    MODELS_DIR,
    NOTEBOOKS_DIR,
    NUM_EPOCHS,
    PATIENCE,
    RANDOM_SEED,
    REPORTS_DIR,
    USE_CLASS_WEIGHTS,
    WEIGHT_DECAY,
)
from data_utils import compute_normalization_stats, create_data_loaders, get_class_weights, load_normalization_stats, prepare_dataset, save_normalization_stats, set_seed
from models import build_model, count_parameters
from stacking import STACKING_MODEL_NAME, train_and_evaluate_stacking_model

MODEL_NAMES = ["urban_cnn_v1", "urban_cnn_v2", "urban_cnn_v3", "urban_cnn_v4"]
MULTITASK_MODEL_NAMES = {"urban_cnn_v4"}
AUXILIARY_RATIO_LOSS_WEIGHT = 1.75


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def is_multitask_model(model_name: str) -> bool:
    return model_name in MULTITASK_MODEL_NAMES


def unpack_batch(batch):
    if len(batch) == 4:
        images, labels, file_names, built_ratios = batch
        return images, labels, file_names, built_ratios
    images, labels, file_names = batch
    return images, labels, file_names, None


def extract_model_outputs(model, images):
    outputs = model(images)
    if isinstance(outputs, dict):
        return outputs
    return {"logits": outputs}


def compute_total_loss(outputs, labels, classification_criterion, built_ratios=None, regression_criterion=None):
    classification_loss = classification_criterion(outputs["logits"], labels)
    total_loss = classification_loss
    built_ratio_loss = None
    if built_ratios is not None and regression_criterion is not None and "built_ratio" in outputs:
        built_ratio_loss = regression_criterion(outputs["built_ratio"].view(-1), built_ratios)
        total_loss = total_loss + (AUXILIARY_RATIO_LOSS_WEIGHT * built_ratio_loss)
    return total_loss, classification_loss, built_ratio_loss


def train_one_epoch(model, loader, classification_criterion, regression_criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    predictions = []
    targets = []

    for batch in loader:
        images, labels, _, built_ratios = unpack_batch(batch)
        images = images.to(device)
        labels = labels.to(device)
        built_ratios = built_ratios.to(device, dtype=torch.float32) if built_ratios is not None else None

        optimizer.zero_grad()
        outputs = extract_model_outputs(model, images)
        loss, _, _ = compute_total_loss(
            outputs,
            labels,
            classification_criterion,
            built_ratios=built_ratios,
            regression_criterion=regression_criterion,
        )
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        predictions.extend(outputs["logits"].argmax(dim=1).detach().cpu().numpy())
        targets.extend(labels.detach().cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)
    epoch_accuracy = accuracy_score(targets, predictions)
    return epoch_loss, epoch_accuracy


def evaluate(model, loader, classification_criterion, regression_criterion, device):
    model.eval()
    running_loss = 0.0
    predictions = []
    probabilities = []
    targets = []
    file_names = []
    predicted_built_ratios = []
    target_built_ratios = []

    with torch.no_grad():
        for batch in loader:
            images, labels, batch_file_names, built_ratios = unpack_batch(batch)
            images = images.to(device)
            labels = labels.to(device)
            built_ratios = built_ratios.to(device, dtype=torch.float32) if built_ratios is not None else None

            outputs = extract_model_outputs(model, images)
            logits = outputs["logits"]
            loss, _, _ = compute_total_loss(
                outputs,
                labels,
                classification_criterion,
                built_ratios=built_ratios,
                regression_criterion=regression_criterion,
            )
            probs = torch.softmax(logits, dim=1)

            running_loss += loss.item() * images.size(0)
            predictions.extend(logits.argmax(dim=1).cpu().numpy())
            probabilities.extend(probs.cpu().numpy())
            targets.extend(labels.cpu().numpy())
            file_names.extend(batch_file_names)
            if built_ratios is not None and "built_ratio" in outputs:
                predicted_built_ratios.extend(outputs["built_ratio"].view(-1).cpu().numpy())
                target_built_ratios.extend(built_ratios.cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)
    epoch_accuracy = accuracy_score(targets, predictions)
    built_ratio_mae = None
    if target_built_ratios:
        built_ratio_mae = float(
            np.mean(np.abs(np.array(predicted_built_ratios) - np.array(target_built_ratios)))
        )
    return {
        "loss": epoch_loss,
        "accuracy": epoch_accuracy,
        "predictions": np.array(predictions),
        "probabilities": np.array(probabilities),
        "targets": np.array(targets),
        "file_names": file_names,
        "built_ratio_mae": built_ratio_mae,
    }


def plot_training_curves(history: pd.DataFrame, model_name: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history["epoch"], history["train_loss"], label="treino")
    axes[0].plot(history["epoch"], history["validation_loss"], label="validacao")
    axes[0].set_title("Loss por epoca")
    axes[0].set_xlabel("Epoca")
    axes[0].set_ylabel("Loss")
    axes[0].legend()

    axes[1].plot(history["epoch"], history["train_accuracy"], label="treino")
    axes[1].plot(history["epoch"], history["validation_accuracy"], label="validacao")
    axes[1].set_title("Acuracia por epoca")
    axes[1].set_xlabel("Epoca")
    axes[1].set_ylabel("Acuracia")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(REPORTS_DIR / f"training_curves_{model_name}.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_confusion_matrix(targets: np.ndarray, predictions: np.ndarray, model_name: str) -> None:
    matrix = confusion_matrix(targets, predictions, labels=list(range(len(CLASS_NAMES))))
    fig, ax = plt.subplots(figsize=(6, 6))
    display = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=CLASS_NAMES)
    display.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(f"Matriz de confusao - {model_name}")
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / f"confusion_matrix_{model_name}.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_error_examples(manifest: pd.DataFrame, evaluation_result: dict, model_name: str, limit: int = 9) -> pd.DataFrame:
    errors = []
    manifest_index = manifest.set_index("file_name")
    for file_name, target, prediction, probabilities in zip(
        evaluation_result["file_names"],
        evaluation_result["targets"],
        evaluation_result["predictions"],
        evaluation_result["probabilities"],
    ):
        if target == prediction:
            continue
        row = manifest_index.loc[file_name]
        errors.append(
            {
                "file_name": file_name,
                "true_label": CLASS_NAMES[int(target)],
                "predicted_label": CLASS_NAMES[int(prediction)],
                "confidence": float(np.max(probabilities)),
                "image_path": row["local_image_path"],
            }
        )

    errors_frame = pd.DataFrame(errors).sort_values("confidence", ascending=False)
    errors_frame.to_csv(REPORTS_DIR / f"misclassifications_{model_name}.csv", index=False)

    if errors_frame.empty:
        return errors_frame

    subset = errors_frame.head(limit)
    rows = int(np.ceil(len(subset) / 3))
    fig, axes = plt.subplots(rows, 3, figsize=(10, 3.5 * rows))
    axes = np.array(axes).reshape(-1)

    for axis in axes:
        axis.axis("off")

    for axis, (_, error_row) in zip(axes, subset.iterrows()):
        image_path = Path(REPORTS_DIR).parents[0] / error_row["image_path"]
        image = Image.open(image_path).convert("RGB")
        axis.imshow(image)
        axis.set_title(
            f"T: {error_row['true_label']}\nP: {error_row['predicted_label']}\nConf: {error_row['confidence']:.2f}",
            fontsize=9,
        )
        axis.axis("off")

    fig.tight_layout()
    fig.savefig(REPORTS_DIR / f"error_examples_{model_name}.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    return errors_frame


def save_architecture_report() -> dict[str, dict[str, object]]:
    architecture_report = {}
    lines = []
    for model_name in MODEL_NAMES:
        model = build_model(model_name)
        architecture_report[model_name] = {
            "trainable_parameters": count_parameters(model),
            "architecture": str(model),
        }
        lines.append(f"=== {model_name} ===")
        lines.append(f"Trainable parameters: {count_parameters(model)}")
        lines.append(str(model))
        lines.append("")

    with open(REPORTS_DIR / "model_architectures.json", "w", encoding="utf-8") as file_handle:
        json.dump(architecture_report, file_handle, indent=4, ensure_ascii=False)
    with open(REPORTS_DIR / "model_architectures.txt", "w", encoding="utf-8") as file_handle:
        file_handle.write("\n".join(lines))
    return architecture_report


def train_model(model_name: str, train_loader, validation_loader, test_loader, train_frame, manifest, device):
    model = build_model(model_name).to(device)
    class_weights = get_class_weights(train_frame).to(device) if USE_CLASS_WEIGHTS else None
    classification_criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=LABEL_SMOOTHING)
    regression_criterion = nn.SmoothL1Loss() if is_multitask_model(model_name) else None
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

    best_state_dict = None
    best_validation_loss = float("inf")
    best_validation_accuracy = 0.0
    epochs_without_improvement = 0
    history_rows = []

    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss, train_accuracy = train_one_epoch(
            model,
            train_loader,
            classification_criterion,
            regression_criterion,
            optimizer,
            device,
        )
        validation_result = evaluate(
            model,
            validation_loader,
            classification_criterion,
            regression_criterion,
            device,
        )
        scheduler.step(validation_result["loss"])

        history_rows.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "validation_loss": validation_result["loss"],
                "validation_accuracy": validation_result["accuracy"],
                "learning_rate": optimizer.param_groups[0]["lr"],
            }
        )

        improved_accuracy = validation_result["accuracy"] > best_validation_accuracy
        improved_loss = validation_result["loss"] < best_validation_loss
        if improved_accuracy or (
            np.isclose(validation_result["accuracy"], best_validation_accuracy) and improved_loss
        ):
            best_validation_loss = validation_result["loss"]
            best_validation_accuracy = validation_result["accuracy"]
            best_state_dict = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        print(
            f"[{model_name}] epoca {epoch:02d} | "
            f"train_acc={train_accuracy:.4f} val_acc={validation_result['accuracy']:.4f} "
            f"train_loss={train_loss:.4f} val_loss={validation_result['loss']:.4f}"
            + (
                f" val_ratio_mae={validation_result['built_ratio_mae']:.4f}"
                if validation_result["built_ratio_mae"] is not None
                else ""
            ),
            flush=True,
        )

        if epochs_without_improvement >= PATIENCE:
            print(f"[{model_name}] early stopping na epoca {epoch}.", flush=True)
            break

    history_frame = pd.DataFrame(history_rows)
    history_frame.to_csv(REPORTS_DIR / f"history_{model_name}.csv", index=False)
    plot_training_curves(history_frame, model_name)

    if best_state_dict is None:
        raise RuntimeError(f"Nenhum estado salvo para o modelo {model_name}.")
    model.load_state_dict(best_state_dict)

    test_result = evaluate(model, test_loader, classification_criterion, regression_criterion, device)
    plot_confusion_matrix(test_result["targets"], test_result["predictions"], model_name)
    errors_frame = plot_error_examples(manifest, test_result, model_name)

    metrics = {
        "model_name": model_name,
        "trainable_parameters": count_parameters(model),
        "best_validation_loss": float(best_validation_loss),
        "best_validation_accuracy": float(best_validation_accuracy),
        "final_train_accuracy": float(history_frame.iloc[-1]["train_accuracy"]),
        "final_validation_accuracy": float(history_frame.iloc[-1]["validation_accuracy"]),
        "test_accuracy": float(test_result["accuracy"]),
        "test_loss": float(test_result["loss"]),
        "misclassified_examples": int(len(errors_frame)),
    }
    if test_result["built_ratio_mae"] is not None:
        metrics["test_built_ratio_mae"] = float(test_result["built_ratio_mae"])

    report = classification_report(
        test_result["targets"],
        test_result["predictions"],
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )
    with open(REPORTS_DIR / f"classification_report_{model_name}.json", "w", encoding="utf-8") as file_handle:
        json.dump(report, file_handle, indent=4, ensure_ascii=False)
    with open(REPORTS_DIR / f"metrics_{model_name}.json", "w", encoding="utf-8") as file_handle:
        json.dump(metrics, file_handle, indent=4, ensure_ascii=False)

    checkpoint = {
        "model_name": model_name,
        "state_dict": model.state_dict(),
        "class_names": CLASS_NAMES,
        "image_size": 64,
        "test_accuracy": metrics["test_accuracy"],
    }
    torch.save(checkpoint, MODELS_DIR / f"{model_name}.pt")
    return metrics, model


def collect_available_metrics() -> pd.DataFrame:
    rows = []
    for metrics_path in sorted(REPORTS_DIR.glob("metrics_*.json")):
        with open(metrics_path, "r", encoding="utf-8") as file_handle:
            rows.append(json.load(file_handle))
    if not rows:
        raise FileNotFoundError("Nenhum arquivo de metricas encontrado para montar a comparacao.")
    return pd.DataFrame(rows).sort_values(["test_accuracy", "best_validation_accuracy"], ascending=False)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models",
        nargs="+",
        default=MODEL_NAMES,
        choices=MODEL_NAMES,
        help="Lista de modelos a treinar.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(RANDOM_SEED)
    torch.set_num_threads(max(1, min(8, (os.cpu_count() or 4) // 2)))
    MODELS_DIR.mkdir(exist_ok=True)
    REPORTS_DIR.mkdir(exist_ok=True)
    NOTEBOOKS_DIR.mkdir(exist_ok=True)

    manifest = prepare_dataset()
    normalization_stats_path = REPORTS_DIR / "normalization_stats.json"
    train_manifest = manifest[manifest["split"] == "train"].copy()
    if normalization_stats_path.exists():
        stats = load_normalization_stats(normalization_stats_path)
    else:
        stats = compute_normalization_stats(train_manifest)
        save_normalization_stats(stats, normalization_stats_path)

    include_built_ratio = any(is_multitask_model(model_name) for model_name in args.models)
    manifest, train_frame, validation_frame, test_frame, train_loader, validation_loader, test_loader = create_data_loaders(
        stats,
        include_built_ratio=include_built_ratio,
    )

    device = get_device()
    print(f"Treinando em: {device}", flush=True)
    print(
        f"Treino: {len(train_frame)} | Validacao: {len(validation_frame)} | Teste: {len(test_frame)}",
        flush=True,
    )

    save_architecture_report()

    trained_models = {}
    for model_name in args.models:
        metrics, model = train_model(
            model_name=model_name,
            train_loader=train_loader,
            validation_loader=validation_loader,
            test_loader=test_loader,
            train_frame=train_frame,
            manifest=manifest,
            device=device,
        )
        trained_models[model_name] = model

    try:
        stacking_metrics = train_and_evaluate_stacking_model(stats)
        print(
            f"[{STACKING_MODEL_NAME}] val_acc={stacking_metrics['best_validation_accuracy']:.4f} "
            f"test_acc={stacking_metrics['test_accuracy']:.4f}",
            flush=True,
        )
    except FileNotFoundError as error:
        print(f"[{STACKING_MODEL_NAME}] pulado: {error}", flush=True)

    comparison_frame = collect_available_metrics()
    comparison_frame.to_csv(REPORTS_DIR / "model_comparison.csv", index=False)

    best_model_name = comparison_frame.iloc[0]["model_name"]
    if best_model_name != STACKING_MODEL_NAME:
        best_model_path = MODELS_DIR / f"{best_model_name}.pt"
        best_checkpoint = torch.load(best_model_path, map_location="cpu", weights_only=False)
        torch.save(best_checkpoint, MODELS_DIR / "best_model.pt")

    best_summary = {
        "best_model": best_model_name,
        "target_task": "Classificacao da densidade urbana visual em imagens de satelite",
        "classes": CLASS_NAMES,
        "test_accuracy": float(comparison_frame.iloc[0]["test_accuracy"]),
        "best_validation_accuracy": float(comparison_frame.iloc[0]["best_validation_accuracy"]),
        "device_used": str(device),
    }
    with open(MODELS_DIR / "best_model_metadata.json", "w", encoding="utf-8") as file_handle:
        json.dump(best_summary, file_handle, indent=4, ensure_ascii=False)

    with open(REPORTS_DIR / "summary.json", "w", encoding="utf-8") as file_handle:
        json.dump(
            {
                "split_sizes": {
                    "train": int(len(train_frame)),
                "validation": int(len(validation_frame)),
                "test": int(len(test_frame)),
            },
                "model_comparison": comparison_frame.to_dict(orient="records"),
                "best_model": best_model_name,
            },
            file_handle,
            indent=4,
            ensure_ascii=False,
        )

    print("\nComparacao final:", flush=True)
    print(
        comparison_frame[["model_name", "test_accuracy", "best_validation_accuracy", "trainable_parameters"]],
        flush=True,
    )
    print(f"\nMelhor modelo: {best_model_name}", flush=True)


if __name__ == "__main__":
    main()
