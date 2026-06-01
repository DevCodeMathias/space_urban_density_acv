from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import requests

from config import ROOT

NASA_GIBS_WMS_URL = "https://gibs.earthdata.nasa.gov/wms/epsg3857/best/wms.cgi"
DEFAULT_LAYER = "VIIRS_SNPP_CorrectedReflectance_TrueColor"
DEFAULT_DATE = "2025-05-01"
DEFAULT_IMAGE_SIZE = 512
EARTH_RADIUS_METERS = 6378137.0
REAL_IMAGES_DIR = ROOT / "data" / "real_images"


@dataclass(frozen=True)
class RealImageExample:
    name: str
    latitude: float
    longitude: float
    bbox_size_meters: int
    note: str


DEFAULT_EXAMPLES = (
    RealImageExample(
        name="sao_paulo_centro",
        latitude=-23.5505,
        longitude=-46.6333,
        bbox_size_meters=12000,
        note="Centro urbano real com mistura de area construida, vias e vegetacao.",
    ),
    RealImageExample(
        name="manaus_periferia",
        latitude=-3.1019,
        longitude=-60.0250,
        bbox_size_meters=18000,
        note="Regiao real com predominancia de vegetacao e ocupacao mais espalhada.",
    ),
    RealImageExample(
        name="brasilia_eixo",
        latitude=-15.7939,
        longitude=-47.8828,
        bbox_size_meters=14000,
        note="Area real planejada, util para testar padrao intermediario de densidade.",
    ),
)


def latlon_to_web_mercator(latitude: float, longitude: float) -> tuple[float, float]:
    x = math.radians(longitude) * EARTH_RADIUS_METERS
    y = math.log(math.tan((math.pi / 4) + (math.radians(latitude) / 2))) * EARTH_RADIUS_METERS
    return x, y


def build_bbox(latitude: float, longitude: float, bbox_size_meters: int) -> tuple[float, float, float, float]:
    center_x, center_y = latlon_to_web_mercator(latitude, longitude)
    half_size = bbox_size_meters / 2
    return (
        center_x - half_size,
        center_y - half_size,
        center_x + half_size,
        center_y + half_size,
    )


def build_nasa_gibs_request(
    *,
    latitude: float,
    longitude: float,
    bbox_size_meters: int,
    date: str,
    layer: str = DEFAULT_LAYER,
    image_size: int = DEFAULT_IMAGE_SIZE,
) -> tuple[str, dict[str, object]]:
    bbox = build_bbox(latitude, longitude, bbox_size_meters)
    params = {
        "SERVICE": "WMS",
        "REQUEST": "GetMap",
        "VERSION": "1.1.1",
        "LAYERS": layer,
        "STYLES": "",
        "FORMAT": "image/png",
        "SRS": "EPSG:3857",
        "WIDTH": image_size,
        "HEIGHT": image_size,
        "BBOX": ",".join(str(value) for value in bbox),
        "TIME": date,
    }
    return NASA_GIBS_WMS_URL, params


def download_real_image(
    *,
    latitude: float,
    longitude: float,
    bbox_size_meters: int,
    date: str,
    layer: str = DEFAULT_LAYER,
    output_path: Path | None = None,
    image_size: int = DEFAULT_IMAGE_SIZE,
) -> tuple[bytes, str, Path | None]:
    url, params = build_nasa_gibs_request(
        latitude=latitude,
        longitude=longitude,
        bbox_size_meters=bbox_size_meters,
        date=date,
        layer=layer,
        image_size=image_size,
    )
    response = requests.get(url, params=params, timeout=120)
    response.raise_for_status()

    saved_path = None
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(response.content)
        saved_path = output_path

    return response.content, response.url, saved_path


def build_real_image_output_path(name: str, date: str) -> Path:
    safe_name = name.strip().replace(" ", "_").lower()
    return REAL_IMAGES_DIR / f"{safe_name}_{date}.png"
