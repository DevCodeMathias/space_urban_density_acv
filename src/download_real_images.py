from __future__ import annotations

import argparse

import pandas as pd
from PIL import Image

from config import ROOT
from inference import predict_image
from nasa_gibs import (
    DEFAULT_DATE,
    DEFAULT_EXAMPLES,
    DEFAULT_LAYER,
    REAL_IMAGES_DIR,
    build_real_image_output_path,
    download_real_image,
)

REAL_IMAGES_REPORT_PATH = ROOT / "reports" / "real_images_predictions.csv"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default=DEFAULT_DATE, help="Data da imagem no formato YYYY-MM-DD.")
    parser.add_argument("--layer", default=DEFAULT_LAYER, help="Layer WMS da NASA GIBS.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = []

    for example in DEFAULT_EXAMPLES:
        output_path = build_real_image_output_path(example.name, args.date)
        _, source_url, _ = download_real_image(
            latitude=example.latitude,
            longitude=example.longitude,
            bbox_size_meters=example.bbox_size_meters,
            date=args.date,
            layer=args.layer,
            output_path=output_path,
        )
        prediction = predict_image(Image.open(output_path))
        rows.append(
            {
                "file_name": output_path.name,
                "local_image_path": str(output_path.relative_to(ROOT)),
                "latitude": example.latitude,
                "longitude": example.longitude,
                "bbox_size_meters": example.bbox_size_meters,
                "date": args.date,
                "layer": args.layer,
                "predicted_class": prediction["predicted_class"],
                "prob_baixa": prediction["probabilities"]["baixa"],
                "prob_media": prediction["probabilities"]["media"],
                "prob_alta": prediction["probabilities"]["alta"],
                "note": example.note,
                "source_url": source_url,
            }
        )

    REAL_IMAGES_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(REAL_IMAGES_REPORT_PATH, index=False)
    print(f"Imagens reais salvas em: {REAL_IMAGES_DIR}")
    print(f"Relatorio salvo em: {REAL_IMAGES_REPORT_PATH}")


if __name__ == "__main__":
    main()
