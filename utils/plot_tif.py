from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image


DATA_ROOT = Path("/home/marius/Asus/CVPR_Workshop/ICML_Workshop_Paper/data/FCV/FVC2004/Dbs/DB1_A")


def _resolve_image_path(image_name: str) -> Path:
    candidate = DATA_ROOT / image_name
    if candidate.exists():
        return candidate
    if not candidate.suffix:
        tif_candidate = candidate.with_suffix(".tif")
        if tif_candidate.exists():
            return tif_candidate
    return candidate


def main() -> None:
    parser = argparse.ArgumentParser(description="Load and plot a .tif image from DB1_A.")
    parser.add_argument("image_name", help="Filename inside DB1_A (with or without .tif)")
    args = parser.parse_args()

    image_path = _resolve_image_path(args.image_name)
    if not image_path.exists():
        raise SystemExit(f"Image not found: {image_path}")

    image = Image.open(image_path)
    plt.imshow(image, cmap="gray")
    plt.axis("off")
    plt.title(image_path.name)
    plt.show()


if __name__ == "__main__":
    main()
