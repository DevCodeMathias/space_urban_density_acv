from __future__ import annotations

import json
import random
import shutil
from dataclasses import asdict, dataclass
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
from satellite_imagery import (
    DEFAULT_ARCGIS_IMAGE_SIZE,
    build_arcgis_world_imagery_request,
    download_arcgis_world_imagery,
)

REAL_DATASET_DATE = "2025-05-01"
REAL_DATASET_IMAGE_SIZE = DEFAULT_ARCGIS_IMAGE_SIZE
REAL_DATASET_SOURCE = "ArcGIS World Imagery"


@dataclass(frozen=True)
class RealSatelliteExample:
    label: str
    name: str
    latitude: float
    longitude: float
    bbox_size_meters: int
    zone_type: str
    note: str


REAL_SATELLITE_EXAMPLES = (
    RealSatelliteExample("alta", "new_york_midtown", 40.7580, -73.9855, 2500, "centro_vertical", "Centro urbano vertical e muito adensado."),
    RealSatelliteExample("alta", "hong_kong_central", 22.2819, 114.1589, 2500, "centro_vertical", "Alta concentracao de edificios e infraestrutura."),
    RealSatelliteExample("alta", "tokyo_shinjuku", 35.6938, 139.7034, 2500, "centro_metropolitano", "Zona metropolitana compacta."),
    RealSatelliteExample("alta", "sao_paulo_paulista", -23.5614, -46.6559, 2500, "centro_metropolitano", "Trecho denso da Avenida Paulista."),
    RealSatelliteExample("alta", "mexico_city_centro", 19.4326, -99.1332, 2500, "centro_metropolitano", "Centro urbano compacto."),
    RealSatelliteExample("alta", "seoul_gangnam", 37.4979, 127.0276, 2500, "centro_metropolitano", "Distrito urbano consolidado."),
    RealSatelliteExample("alta", "paris_ladefense", 48.8919, 2.2370, 2500, "centro_empresarial", "Regiao empresarial muito construida."),
    RealSatelliteExample("alta", "london_city", 51.5155, -0.0922, 2500, "centro_historico", "Centro financeiro com malha urbana densa."),
    RealSatelliteExample("alta", "singapore_downtown", 1.2834, 103.8607, 2500, "centro_vertical", "Centro urbano de alta densidade."),
    RealSatelliteExample("alta", "mumbai_colaba", 18.9067, 72.8147, 3000, "centro_metropolitano", "Ocupacao urbana intensa."),
    RealSatelliteExample("alta", "shanghai_pudong", 31.2304, 121.4737, 3000, "centro_vertical", "Distrito urbano verticalizado."),
    RealSatelliteExample("alta", "chicago_loop", 41.8789, -87.6359, 2500, "centro_vertical", "Centro urbano com forte presenca construida."),
    RealSatelliteExample("alta", "buenos_aires_microcentro", -34.6037, -58.3816, 2500, "centro_metropolitano", "Malha urbana central compacta."),
    RealSatelliteExample("alta", "santiago_centro", -33.4489, -70.6693, 2500, "centro_metropolitano", "Centro urbano consolidado."),
    RealSatelliteExample("alta", "dubai_marina", 25.0800, 55.1400, 2500, "centro_vertical", "Concentracao vertical costeira."),
    RealSatelliteExample("media", "sao_paulo_centro_amplo", -23.5505, -46.6333, 7000, "urbano_misto", "Area urbana ampla com vias, construcoes e vegetacao."),
    RealSatelliteExample("media", "brasilia_eixo", -15.7939, -47.8828, 8000, "urbano_planejado", "Area planejada de densidade intermediaria."),
    RealSatelliteExample("media", "campinas", -22.9056, -47.0608, 7000, "urbano_misto", "Cidade consolidada, menos densa que megacentros."),
    RealSatelliteExample("media", "curitiba", -25.4284, -49.2733, 7000, "urbano_arborizado", "Capital com malha urbana e areas verdes."),
    RealSatelliteExample("media", "goiania", -16.6869, -49.2648, 7000, "urbano_misto", "Padrao intermediario entre ocupacao e vegetacao."),
    RealSatelliteExample("media", "belo_horizonte", -19.9167, -43.9345, 7000, "urbano_misto", "Ocupacao urbana relevante, mas nao extrema."),
    RealSatelliteExample("media", "austin", 30.2672, -97.7431, 8000, "suburbano", "Cidade espalhada com densidade moderada."),
    RealSatelliteExample("media", "orlando_suburb", 28.5383, -81.3792, 8000, "suburbano", "Suburbio urbano misto."),
    RealSatelliteExample("media", "lyon", 45.7640, 4.8357, 7000, "urbano_misto", "Centro urbano europeu intermediario."),
    RealSatelliteExample("media", "porto_alegre", -30.0346, -51.2177, 7000, "urbano_misto", "Capital com ocupacao urbana e areas abertas."),
    RealSatelliteExample("media", "recife", -8.0476, -34.8770, 7000, "urbano_costeiro", "Area urbana costeira de densidade intermediaria."),
    RealSatelliteExample("media", "fortaleza", -3.7319, -38.5267, 7000, "urbano_costeiro", "Capital costeira com ocupacao continua."),
    RealSatelliteExample("media", "lisbon", 38.7223, -9.1393, 7000, "urbano_misto", "Area urbana com relevo e espacos abertos."),
    RealSatelliteExample("media", "montreal", 45.5017, -73.5673, 7000, "urbano_misto", "Malha urbana consolidada com densidade moderada."),
    RealSatelliteExample("media", "melbourne", -37.8136, 144.9631, 8000, "urbano_misto", "Centro e entorno urbano de densidade intermediaria."),
    RealSatelliteExample("baixa", "amazonas_forest", -3.4653, -62.2159, 9000, "floresta", "Cobertura vegetal dominante."),
    RealSatelliteExample("baixa", "pantanal", -16.4897, -56.3689, 9000, "area_natural", "Area natural aberta com pouca ocupacao."),
    RealSatelliteExample("baixa", "chapada_diamantina", -12.5564, -41.3924, 9000, "area_natural", "Paisagem natural predominante."),
    RealSatelliteExample("baixa", "yellowstone", 44.4280, -110.5885, 9000, "parque_natural", "Parque natural com baixissima ocupacao."),
    RealSatelliteExample("baixa", "patagonia", -50.9423, -73.4068, 9000, "area_natural", "Area aberta e pouco ocupada."),
    RealSatelliteExample("baixa", "cerrado", -14.2350, -51.9253, 9000, "area_natural", "Paisagem natural dominante."),
    RealSatelliteExample("baixa", "alaska_forest", 61.2181, -149.9003, 9000, "floresta", "Predominio natural."),
    RealSatelliteExample(
        "media",
        "santa_cruz_bolivia",
        -17.7833,
        -63.1821,
        9000,
        "urbano_misto",
        "Area urbana extensa de Santa Cruz de la Sierra.",
    ),
    RealSatelliteExample("baixa", "sahara", 23.4162, 25.6628, 9000, "deserto", "Area desertica sem ocupacao urbana relevante."),
    RealSatelliteExample("baixa", "serengeti", -2.3333, 34.8333, 9000, "area_natural", "Paisagem natural aberta."),
    RealSatelliteExample("baixa", "andes_rural", -13.1631, -72.5450, 9000, "rural_montanhoso", "Regiao montanhosa de baixa ocupacao."),
    RealSatelliteExample("baixa", "namib_desert", -24.7500, 15.3000, 9000, "deserto", "Paisagem desertica com baixa ocupacao."),
    RealSatelliteExample("baixa", "canadian_rockies", 51.4968, -115.9281, 9000, "area_natural", "Area montanhosa natural."),
    RealSatelliteExample("baixa", "mongolia_steppe", 47.8864, 106.9057, 9000, "area_natural", "Estepe com ocupacao rarefeita."),
    RealSatelliteExample("baixa", "manaus_periferia_verde", -3.1019, -60.0250, 9000, "urbano_esparso", "Predominio de vegetacao e ocupacao espalhada."),
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
    existing_splits = {}
    if MANIFEST_PATH.exists():
        existing_manifest = pd.read_csv(MANIFEST_PATH)
        if {"name", "split"}.issubset(existing_manifest.columns):
            existing_splits = dict(zip(existing_manifest["name"], existing_manifest["split"]))

    rows = []
    for example in REAL_SATELLITE_EXAMPLES:
        file_name = f"{example.name}_{REAL_DATASET_DATE}.png"
        target_file = RAW_IMAGES_DIR / file_name
        _, request_params = build_arcgis_world_imagery_request(
            latitude=example.latitude,
            longitude=example.longitude,
            bbox_size_meters=example.bbox_size_meters,
            image_size=REAL_DATASET_IMAGE_SIZE,
        )
        source_url = None
        if not target_file.exists():
            _, source_url, _ = download_arcgis_world_imagery(
                latitude=example.latitude,
                longitude=example.longitude,
                bbox_size_meters=example.bbox_size_meters,
                output_path=target_file,
                image_size=REAL_DATASET_IMAGE_SIZE,
            )
        image = Image.open(target_file).convert("RGB")
        visual_features = compute_image_visual_features(image)
        class_index = CLASS_TO_INDEX[example.label]
        rows.append(
            {
                **asdict(example),
                "file_name": file_name,
                "image_path": str(target_file),
                "local_image_path": str(target_file.relative_to(MANIFEST_PATH.parents[1])),
                "source_image_path": source_url or REAL_DATASET_SOURCE,
                "source_api": REAL_DATASET_SOURCE,
                "source_layer": "World_Imagery",
                "source_date": REAL_DATASET_DATE,
                "request_bbox_meters": example.bbox_size_meters,
                "request_params": json.dumps(request_params, ensure_ascii=False),
                LABEL_COLUMN: example.label,
                "label_index": class_index,
                "built_ratio": float(np.clip([0.18, 0.52, 0.84][class_index] + visual_features["built_up_proxy"], 0.0, 1.0)),
                **visual_features,
            }
        )

    metadata = pd.DataFrame(rows)

    if set(metadata["name"]).issubset(existing_splits):
        metadata["split"] = metadata["name"].map(existing_splits)
        manifest = metadata
    else:
        train_frame, temp_frame = train_test_split(
            metadata,
            test_size=0.40,
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


def compute_image_visual_features(image: Image.Image) -> dict[str, float]:
    image_array = np.asarray(image.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE)), dtype=np.float32)
    red_mean = float(image_array[:, :, 0].mean())
    green_mean = float(image_array[:, :, 1].mean())
    blue_mean = float(image_array[:, :, 2].mean())
    gray = image_array.mean(axis=2)
    brightness = float(image_array.mean())
    contrast = float(gray.std())
    vegetation_index = float((green_mean - red_mean) / (green_mean + red_mean + 1e-8))
    water_index = float((blue_mean - red_mean) / (blue_mean + red_mean + 1e-8))
    built_up_proxy = float(
        (red_mean + blue_mean - (2.0 * green_mean))
        / (red_mean + blue_mean + (2.0 * green_mean) + 1e-8)
    )
    grad_x = np.diff(gray, axis=1, append=gray[:, -1:])
    grad_y = np.diff(gray, axis=0, append=gray[-1:, :])
    texture = float(np.sqrt((grad_x * grad_x) + (grad_y * grad_y)).mean())
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
