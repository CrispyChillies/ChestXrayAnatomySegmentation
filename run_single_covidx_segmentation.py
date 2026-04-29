from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from cxas import CXAS
from cxas.label_mapper import id2label_dict


REGION_SPECS = [
    ("right_upper_lung_zone", "right upper zone lung", (255, 80, 80)),
    ("right_mid_lung_zone", "right mid zone lung", (255, 170, 80)),
    ("right_lower_lung_zone", "right lung base", (255, 230, 80)),
    ("left_upper_lung_zone", "left upper zone lung", (80, 160, 255)),
    ("left_mid_lung_zone", "left mid zone lung", (80, 220, 255)),
    ("left_lower_lung_zone", "left lung base", (120, 255, 160)),
]


def build_label_index() -> dict[str, int]:
    return {label_name: int(label_id) for label_id, label_name in id2label_dict.items()}


def mask_to_bbox(mask: np.ndarray) -> list[int] | None:
    ys, xs = np.where(mask)
    if xs.size == 0 or ys.size == 0:
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]


def save_binary_mask(mask: np.ndarray, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((mask.astype(np.uint8) * 255)).save(output_path)


def create_overlay(image: Image.Image, regions: list[dict]) -> Image.Image:
    base = image.convert("RGB")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    for region in regions:
        if not region["present"]:
            continue

        mask = np.array(Image.open(region["mask_path"]).convert("L")) > 0
        color = tuple(region["color"])
        rgba = color + (70,)

        overlay_np = np.array(overlay)
        overlay_np[mask] = rgba
        overlay = Image.fromarray(overlay_np, mode="RGBA")

        bbox = region["bbox_xyxy"]
        if bbox is not None:
            x1, y1, x2, y2 = bbox
            draw = ImageDraw.Draw(overlay)
            draw.rectangle([x1, y1, x2 - 1, y2 - 1], outline=color + (255,), width=2)
            draw.text((x1 + 3, max(0, y1 - 14)), region["region_name"], fill=color + (255,))

    return Image.alpha_composite(base.convert("RGBA"), overlay).convert("RGB")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run CXAS on one chest X-ray and save the 6 lung-zone masks."
    )
    parser.add_argument("--image", required=True, help="Path to one chest X-ray image.")
    parser.add_argument("--output-dir", required=True, help="Output directory.")
    parser.add_argument(
        "--gpus",
        default="cpu",
        help="GPU setting for CXAS. Use 'cpu' for simple testing.",
    )
    args = parser.parse_args()

    image_path = Path(args.image)
    output_dir = Path(args.output_dir)
    masks_dir = output_dir / "masks"
    output_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)

    model = CXAS(model_name="UNet_ResNet50_default", gpus=args.gpus)
    predictions = model.process_file(str(image_path))
    resized_mask = model.resize_to_numpy(
        predictions["segmentation_preds"][0],
        predictions["file_size"][0],
    )

    image = Image.open(image_path).convert("RGB")
    label_index = build_label_index()
    region_payloads = []

    for region_name, cxas_label, color in REGION_SPECS:
        class_index = label_index[cxas_label]
        region_mask = resized_mask[class_index].astype(bool)
        bbox = mask_to_bbox(region_mask)
        mask_path = masks_dir / f"{region_name}.png"
        if bbox is not None:
            save_binary_mask(region_mask, mask_path)
        else:
            mask_path = None

        area_pixels = int(region_mask.sum())
        region_payloads.append(
            {
                "region_name": region_name,
                "cxas_label": cxas_label,
                "present": bbox is not None,
                "bbox_xyxy": bbox,
                "area_pixels": area_pixels,
                "area_ratio": float(area_pixels / region_mask.size),
                "mask_path": str(mask_path) if mask_path is not None else None,
                "color": list(color),
            }
        )

    overlay = create_overlay(image, region_payloads)
    overlay.save(output_dir / "overlay.png")

    with open(output_dir / "regions.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "image_path": str(image_path),
                "regions": region_payloads,
            },
            f,
            indent=2,
        )

    print(f"Saved results to: {output_dir}")
    print(f"- overlay: {output_dir / 'overlay.png'}")
    print(f"- metadata: {output_dir / 'regions.json'}")
    print(f"- masks: {masks_dir}")


if __name__ == "__main__":
    main()
