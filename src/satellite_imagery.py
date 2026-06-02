from __future__ import annotations

import math
from pathlib import Path

import requests

ARCGIS_WORLD_IMAGERY_EXPORT_URL = (
    "https://services.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/export"
)
DEFAULT_ARCGIS_IMAGE_SIZE = 1024


def meters_to_latlon_bbox(
    latitude: float,
    longitude: float,
    bbox_size_meters: int,
) -> tuple[float, float, float, float]:
    half_size = bbox_size_meters / 2
    meters_per_degree_latitude = 111_320.0
    meters_per_degree_longitude = meters_per_degree_latitude * math.cos(math.radians(latitude))
    if abs(meters_per_degree_longitude) < 1e-8:
        raise ValueError("Longitude indefinida perto dos polos para bbox em graus.")

    latitude_delta = half_size / meters_per_degree_latitude
    longitude_delta = half_size / meters_per_degree_longitude
    return (
        longitude - longitude_delta,
        latitude - latitude_delta,
        longitude + longitude_delta,
        latitude + latitude_delta,
    )


def build_arcgis_world_imagery_request(
    *,
    latitude: float,
    longitude: float,
    bbox_size_meters: int,
    image_size: int = DEFAULT_ARCGIS_IMAGE_SIZE,
) -> tuple[str, dict[str, object]]:
    bbox = meters_to_latlon_bbox(latitude, longitude, bbox_size_meters)
    params = {
        "bbox": ",".join(str(value) for value in bbox),
        "bboxSR": 4326,
        "imageSR": 4326,
        "size": f"{image_size},{image_size}",
        "format": "png",
        "f": "image",
    }
    return ARCGIS_WORLD_IMAGERY_EXPORT_URL, params


def download_arcgis_world_imagery(
    *,
    latitude: float,
    longitude: float,
    bbox_size_meters: int,
    output_path: Path | None = None,
    image_size: int = DEFAULT_ARCGIS_IMAGE_SIZE,
) -> tuple[bytes, str, Path | None]:
    response = None
    for candidate_size in (image_size, 768, 512):
        url, params = build_arcgis_world_imagery_request(
            latitude=latitude,
            longitude=longitude,
            bbox_size_meters=bbox_size_meters,
            image_size=candidate_size,
        )
        response = requests.get(url, params=params, timeout=120)
        if response.ok:
            break
    if response is None:
        raise RuntimeError("Nenhuma tentativa de download foi executada.")
    response.raise_for_status()

    saved_path = None
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(response.content)
        saved_path = output_path

    return response.content, response.url, saved_path
