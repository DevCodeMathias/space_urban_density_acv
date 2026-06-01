from __future__ import annotations

from config import DATA_DIR, LABEL_COLUMN, REPORTS_DIR
from data_utils import compute_normalization_stats, load_manifest, prepare_dataset, save_normalization_stats


def main() -> None:
    DATA_DIR.mkdir(exist_ok=True)
    REPORTS_DIR.mkdir(exist_ok=True)

    manifest = prepare_dataset()
    train_manifest = manifest[manifest["split"] == "train"].copy()
    normalization_stats = compute_normalization_stats(train_manifest)
    save_normalization_stats(normalization_stats, REPORTS_DIR / "normalization_stats.json")

    print(f"Manifest salvo em: {DATA_DIR / 'manifest.csv'}")
    print(f"Total de imagens: {len(manifest)}")
    print("\nDistribuicao por split e classe:")
    print(manifest.groupby(["split", LABEL_COLUMN]).size())
    print("\nNormalizacao:")
    print(normalization_stats)


if __name__ == "__main__":
    main()
