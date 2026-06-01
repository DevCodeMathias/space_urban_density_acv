from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import joblib
import pandas as pd
import torch
from PIL import Image

from config import CLASS_NAMES, MODELS_DIR, REPORTS_DIR, ROOT
from data_utils import build_transforms, load_normalization_stats
from models import build_model
from nasa_gibs import DEFAULT_DATE, DEFAULT_LAYER, build_real_image_output_path, download_real_image
from stacking import STACKING_BASE_MODELS, compute_image_visual_features

REAL_DOMAIN_MODEL_NAME = "urban_real_nasa_adapter_v1"
REAL_DOMAIN_DATASET_PATH = REPORTS_DIR / "real_domain_dataset.csv"
REAL_DOMAIN_METRICS_PATH = REPORTS_DIR / f"metrics_{REAL_DOMAIN_MODEL_NAME}.json"
REAL_DOMAIN_MODEL_PATH = MODELS_DIR / f"{REAL_DOMAIN_MODEL_NAME}.joblib"
REAL_DOMAIN_METADATA_PATH = MODELS_DIR / f"{REAL_DOMAIN_MODEL_NAME}_metadata.json"
REAL_DOMAIN_FEATURE_COLUMNS = [
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


@dataclass(frozen=True)
class RealLabeledExample:
    label: str
    name: str
    latitude: float
    longitude: float
    bbox_size_meters: int
    note: str


REAL_LABELED_EXAMPLES = (
    RealLabeledExample("alta", "new_york_midtown", 40.7580, -73.9855, 2500, "Centro urbano vertical e muito adensado."),
    RealLabeledExample("alta", "hong_kong_central", 22.2819, 114.1589, 2500, "Centro muito adensado com alta concentracao de edificios."),
    RealLabeledExample("alta", "tokyo_shinjuku", 35.6938, 139.7034, 2500, "Zona metropolitana com forte concentracao urbana."),
    RealLabeledExample("alta", "sao_paulo_paulista", -23.5614, -46.6559, 2500, "Trecho denso da Avenida Paulista."),
    RealLabeledExample("alta", "mexico_city_centro", 19.4326, -99.1332, 2500, "Centro urbano compacto e muito ocupado."),
    RealLabeledExample("alta", "santiago_centro", -33.4489, -70.6693, 2500, "Centro urbano com alta presenca de edificacoes."),
    RealLabeledExample("alta", "buenos_aires_microcentro", -34.6037, -58.3816, 2500, "Area central densa com malha urbana concentrada."),
    RealLabeledExample("alta", "seoul_gangnam", 37.4979, 127.0276, 2500, "Distrito urbano muito consolidado."),
    RealLabeledExample("alta", "paris_ladefense", 48.8919, 2.2370, 2500, "Regiao empresarial com forte densidade construtiva."),
    RealLabeledExample("media", "sao_paulo_centro", -23.5505, -46.6333, 12000, "Centro amplo com mistura de vias, construcoes e vegetacao."),
    RealLabeledExample("media", "brasilia_eixo", -15.7939, -47.8828, 14000, "Area planejada com densidade intermediaria."),
    RealLabeledExample("media", "campinas", -22.9056, -47.0608, 12000, "Area urbana consolidada, porem menos densa que megacentros."),
    RealLabeledExample("media", "curitiba", -25.4284, -49.2733, 12000, "Capital com malha urbana clara e espacos arborizados."),
    RealLabeledExample("media", "goiania", -16.6869, -49.2648, 12000, "Padrao intermediario entre ocupacao e vegetacao."),
    RealLabeledExample("media", "belo_horizonte", -19.9167, -43.9345, 12000, "Trecho urbano com ocupacao relevante mas nao extrema."),
    RealLabeledExample("media", "austin", 30.2672, -97.7431, 15000, "Cidade espalhada com ocupacao intermediaria."),
    RealLabeledExample("media", "orlando_suburb", 28.5383, -81.3792, 15000, "Suburbio urbano misto com densidade moderada."),
    RealLabeledExample("media", "lyon", 45.7640, 4.8357, 12000, "Centro urbano europeu de densidade intermediaria."),
    RealLabeledExample("baixa", "manaus_periferia", -3.1019, -60.0250, 18000, "Predominio de vegetacao e ocupacao mais espalhada."),
    RealLabeledExample("baixa", "amazonas_forest", -3.4653, -62.2159, 20000, "Cobertura vegetal dominante."),
    RealLabeledExample("baixa", "pantanal", -16.4897, -56.3689, 20000, "Area natural aberta com pouca ocupacao."),
    RealLabeledExample("baixa", "chapada", -12.5564, -41.3924, 20000, "Area natural com baixa presenca de infraestrutura."),
    RealLabeledExample("baixa", "yellowstone", 44.4280, -110.5885, 16000, "Parque natural com baixissima ocupacao urbana."),
    RealLabeledExample("baixa", "patagonia", -50.9423, -73.4068, 20000, "Area aberta e pouco ocupada."),
    RealLabeledExample("baixa", "cerrado", -14.2350, -51.9253, 20000, "Paisagem natural dominante."),
    RealLabeledExample("baixa", "alaska_forest", 61.2181, -149.9003, 20000, "Predominio natural com baixa densidade urbana."),
    RealLabeledExample("baixa", "bolivia_rural", -17.7833, -63.1821, 20000, "Area rural extensa com ocupacao esparsa."),
)


def load_base_models() -> dict[str, torch.nn.Module]:
    loaded_models = {}
    for model_name in STACKING_BASE_MODELS:
        checkpoint = torch.load(MODELS_DIR / f"{model_name}.pt", map_location="cpu", weights_only=False)
        model = build_model(model_name)
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        loaded_models[model_name] = model
    return loaded_models


def extract_model_probability_features(image: Image.Image, base_models: dict[str, torch.nn.Module]) -> dict[str, float]:
    normalization_stats = load_normalization_stats(REPORTS_DIR / "normalization_stats.json")
    _, eval_transform = build_transforms(normalization_stats)
    tensor = eval_transform(image.convert("RGB")).unsqueeze(0)
    features = {}
    with torch.no_grad():
        for model_name, model in base_models.items():
            logits = model(tensor)
            probabilities = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
            for class_index in range(len(CLASS_NAMES)):
                features[f"{model_name}_p{class_index}"] = float(probabilities[class_index])
    return features


def build_real_domain_dataset(
    *,
    date: str = DEFAULT_DATE,
    layer: str = DEFAULT_LAYER,
    force_redownload: bool = False,
) -> pd.DataFrame:
    base_models = load_base_models()
    rows = []
    for example in REAL_LABELED_EXAMPLES:
        image_path = build_real_image_output_path(example.name, date)
        if force_redownload or not image_path.exists():
            download_real_image(
                latitude=example.latitude,
                longitude=example.longitude,
                bbox_size_meters=example.bbox_size_meters,
                date=date,
                layer=layer,
                output_path=image_path,
                image_size=256,
            )
        image = Image.open(image_path).convert("RGB")
        row = {
            "name": example.name,
            "label": example.label,
            "label_index": CLASS_NAMES.index(example.label),
            "latitude": example.latitude,
            "longitude": example.longitude,
            "bbox_size_meters": example.bbox_size_meters,
            "date": date,
            "layer": layer,
            "local_image_path": str(image_path.relative_to(ROOT)),
            "note": example.note,
        }
        row.update(extract_model_probability_features(image, base_models))
        row.update(compute_image_visual_features(image))
        rows.append(row)

    dataset = pd.DataFrame(rows)
    REAL_DOMAIN_DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(REAL_DOMAIN_DATASET_PATH, index=False)
    return dataset


def save_real_domain_metadata(metadata: dict[str, object]) -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    with open(REAL_DOMAIN_METADATA_PATH, "w", encoding="utf-8") as file_handle:
        json.dump(metadata, file_handle, indent=4, ensure_ascii=False)


def export_real_domain_catalog(path: Path) -> None:
    frame = pd.DataFrame([asdict(example) for example in REAL_LABELED_EXAMPLES])
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def predict_real_domain_image(image: Image.Image) -> dict[str, object]:
    if not REAL_DOMAIN_MODEL_PATH.exists():
        raise FileNotFoundError(f"Modelo de adaptacao real nao encontrado: {REAL_DOMAIN_MODEL_PATH}")
    estimator = joblib.load(REAL_DOMAIN_MODEL_PATH)
    base_models = load_base_models()
    feature_row = extract_model_probability_features(image, base_models)
    feature_row.update(compute_image_visual_features(image))
    feature_frame = pd.DataFrame([feature_row], columns=REAL_DOMAIN_FEATURE_COLUMNS)
    probabilities = estimator.predict_proba(feature_frame)[0]
    predicted_index = int(probabilities.argmax())
    return {
        "model_name": REAL_DOMAIN_MODEL_NAME,
        "predicted_class": CLASS_NAMES[predicted_index],
        "probabilities": {
            class_name: float(probability)
            for class_name, probability in zip(CLASS_NAMES, probabilities)
        },
    }
