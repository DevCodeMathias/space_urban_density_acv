from __future__ import annotations

import json
import random
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from PIL.Image import Resampling
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torchvision import transforms

from config import (
    BATCH_SIZE,
    CLASS_DISTRIBUTION_PATH,
    CLASS_NAMES,
    CLASS_TO_INDEX,
    IMAGE_SIZE,
    LABEL_COLUMN,
    MANIFEST_PATH,
    RAW_IMAGES_DIR,
    RANDOM_SEED,
    SOURCE_IMAGES_DIR,
    SOURCE_METADATA_PATH,
    SPLITS_DIR,
)


def set_seed(seed: int = RANDOM_SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def ensure_project_dirs() -> None:
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    RAW_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    CLASS_DISTRIBUTION_PATH.parent.mkdir(parents=True, exist_ok=True)


def prepare_dataset() -> pd.DataFrame:
    ensure_project_dirs()
    metadata = pd.read_csv(SOURCE_METADATA_PATH)
    metadata[LABEL_COLUMN] = pd.cut(
        metadata["built_ratio"],
        bins=[-np.inf, 0.35, 0.65, np.inf],
        labels=CLASS_NAMES,
    ).astype(str)
    metadata["file_name"] = metadata["image_path"].apply(lambda path: Path(path).name)
    metadata["source_file"] = metadata["file_name"].apply(lambda file_name: SOURCE_IMAGES_DIR / file_name)
    metadata["target_file"] = metadata["file_name"].apply(lambda file_name: RAW_IMAGES_DIR / file_name)

    for source_file, target_file in metadata[["source_file", "target_file"]].itertuples(index=False):
        if not target_file.exists():
            shutil.copy2(source_file, target_file)

    train_frame, temp_frame = train_test_split(
        metadata,
        test_size=0.30,
        random_state=RANDOM_SEED,
        stratify=metadata[LABEL_COLUMN],
    )
    validation_frame, test_frame = train_test_split(
        temp_frame,
        test_size=0.50,
        random_state=RANDOM_SEED,
        stratify=temp_frame[LABEL_COLUMN],
    )

    train_frame = train_frame.assign(split="train")
    validation_frame = validation_frame.assign(split="validation")
    test_frame = test_frame.assign(split="test")

    manifest = pd.concat([train_frame, validation_frame, test_frame], ignore_index=True)
    manifest["label_index"] = manifest[LABEL_COLUMN].map(CLASS_TO_INDEX)
    manifest["local_image_path"] = manifest["target_file"].apply(lambda path: str(path.relative_to(MANIFEST_PATH.parents[1])))
    manifest["source_image_path"] = manifest["source_file"].astype(str)
    manifest = manifest.drop(columns=["source_file", "target_file"])
    manifest.to_csv(MANIFEST_PATH, index=False)

    for split_name, split_frame in manifest.groupby("split"):
        split_frame.to_csv(SPLITS_DIR / f"{split_name}.csv", index=False)

    class_distribution = (
        manifest.groupby(["split", LABEL_COLUMN]).size().rename("images").reset_index()
    )
    class_distribution.to_csv(CLASS_DISTRIBUTION_PATH, index=False)
    return manifest


def load_manifest() -> pd.DataFrame:
    if not MANIFEST_PATH.exists():
        return prepare_dataset()
    return pd.read_csv(MANIFEST_PATH)


@dataclass
class NormalizationStats:
    mean: list[float]
    std: list[float]


def compute_normalization_stats(train_manifest: pd.DataFrame) -> NormalizationStats:
    channel_sum = np.zeros(3, dtype=np.float64)
    channel_squared_sum = np.zeros(3, dtype=np.float64)
    total_pixels = 0

    for relative_path in train_manifest["local_image_path"]:
        image_path = MANIFEST_PATH.parents[1] / relative_path
        image = Image.open(image_path).convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))
        image_array = np.asarray(image, dtype=np.float32) / 255.0
        channel_sum += image_array.reshape(-1, 3).sum(axis=0)
        channel_squared_sum += (image_array.reshape(-1, 3) ** 2).sum(axis=0)
        total_pixels += image_array.shape[0] * image_array.shape[1]

    mean = channel_sum / total_pixels
    variance = (channel_squared_sum / total_pixels) - (mean ** 2)
    std = np.sqrt(np.maximum(variance, 1e-8))
    return NormalizationStats(mean=mean.tolist(), std=std.tolist())


def save_normalization_stats(stats: NormalizationStats, output_path: Path) -> None:
    with open(output_path, "w", encoding="utf-8") as file_handle:
        json.dump({"mean": stats.mean, "std": stats.std}, file_handle, indent=4, ensure_ascii=False)


def load_normalization_stats(path: Path) -> NormalizationStats:
    with open(path, "r", encoding="utf-8") as file_handle:
        payload = json.load(file_handle)
    return NormalizationStats(mean=payload["mean"], std=payload["std"])


class UrbanDensityImageDataset(Dataset):
    def __init__(self, manifest: pd.DataFrame, transform=None, include_built_ratio: bool = False):
        self.manifest = manifest.reset_index(drop=True)
        self.transform = transform
        self.include_built_ratio = include_built_ratio

    def __len__(self) -> int:
        return len(self.manifest)

    def __getitem__(self, index: int):
        row = self.manifest.iloc[index]
        image_path = MANIFEST_PATH.parents[1] / row["local_image_path"]
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        label = int(row["label_index"])
        if self.include_built_ratio:
            built_ratio = float(row["built_ratio"])
            return image, label, row["file_name"], built_ratio
        return image, label, row["file_name"]


class RandomDiscreteRotation:
    def __init__(self, angles: tuple[int, ...] = (0, 90, 180, 270)):
        self.angles = angles

    def __call__(self, image: Image.Image) -> Image.Image:
        angle = random.choice(self.angles)
        if angle == 0:
            return image
        return image.rotate(angle, resample=Resampling.BILINEAR)


def build_transforms(stats: NormalizationStats):
    normalize = transforms.Normalize(mean=stats.mean, std=stats.std)
    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop((IMAGE_SIZE, IMAGE_SIZE), scale=(0.90, 1.0), ratio=(0.95, 1.05)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            RandomDiscreteRotation(),
            transforms.ColorJitter(brightness=0.18, contrast=0.18, saturation=0.12, hue=0.02),
            transforms.ToTensor(),
            normalize,
            transforms.RandomErasing(p=0.20, scale=(0.02, 0.12), ratio=(0.5, 1.8), value="random"),
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            normalize,
        ]
    )
    return train_transform, eval_transform


def create_data_loaders(stats: NormalizationStats, include_built_ratio: bool = False):
    manifest = load_manifest()
    train_frame = manifest[manifest["split"] == "train"].copy()
    validation_frame = manifest[manifest["split"] == "validation"].copy()
    test_frame = manifest[manifest["split"] == "test"].copy()

    train_transform, eval_transform = build_transforms(stats)

    train_dataset = UrbanDensityImageDataset(
        train_frame,
        transform=train_transform,
        include_built_ratio=include_built_ratio,
    )
    validation_dataset = UrbanDensityImageDataset(
        validation_frame,
        transform=eval_transform,
        include_built_ratio=include_built_ratio,
    )
    test_dataset = UrbanDensityImageDataset(
        test_frame,
        transform=eval_transform,
        include_built_ratio=include_built_ratio,
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
    )
    validation_loader = torch.utils.data.DataLoader(
        validation_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )
    return manifest, train_frame, validation_frame, test_frame, train_loader, validation_loader, test_loader


def get_class_weights(train_frame: pd.DataFrame) -> torch.Tensor:
    counts = train_frame[LABEL_COLUMN].value_counts().reindex(CLASS_NAMES)
    weights = len(train_frame) / (len(CLASS_NAMES) * counts)
    return torch.tensor(weights.values, dtype=torch.float32)
