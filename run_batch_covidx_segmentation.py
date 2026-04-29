from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from cxas import CXAS
from cxas.label_mapper import id2label_dict
from cxas.covidx_pipeline import parse_covidx_manifest


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

    for region in regions:
        if not region["present"]:
            continue

        mask = np.array(Image.open(region["mask_path"]).convert("L")) > 0
        color = tuple(region["color"])
        rgba = color + (70,)

        overlay_np = np.array(overlay)
        overlay_np[mask] = rgba
        overlay = Image.fromarray(overlay_np, mode="RGBA")

        draw = ImageDraw.Draw(overlay)
        x1, y1, x2, y2 = region["bbox_xyxy"]
        draw.rectangle([x1, y1, x2 - 1, y2 - 1], outline=color + (255,), width=2)
        draw.text((x1 + 3, max(0, y1 - 14)), region["region_name"], fill=color + (255,))

    return Image.alpha_composite(base.convert("RGBA"), overlay).convert("RGB")


def process_single_image(model: CXAS, image_path: Path, output_dir: Path, label_index: dict[str, int]) -> dict:
    predictions = model.process_file(str(image_path))
    resized_mask = model.resize_to_numpy(
        predictions["segmentation_preds"][0],
        predictions["file_size"][0],
    )
    image = Image.open(image_path).convert("RGB")

    image_output_dir = output_dir / image_path.stem
    masks_dir = image_output_dir / "masks"
    masks_dir.mkdir(parents=True, exist_ok=True)

    region_payloads = []
    six_zone_stack = []

    for region_name, cxas_label, color in REGION_SPECS:
        class_index = label_index[cxas_label]
        region_mask = resized_mask[class_index].astype(bool)
        bbox = mask_to_bbox(region_mask)
        area_pixels = int(region_mask.sum())
        mask_path = masks_dir / f"{region_name}.png"

        if bbox is not None:
            save_binary_mask(region_mask, mask_path)
            mask_path_value = str(mask_path)
        else:
            mask_path_value = None

        region_payloads.append(
            {
                "region_name": region_name,
                "cxas_label": cxas_label,
                "present": bbox is not None,
                "bbox_xyxy": bbox,
                "area_pixels": area_pixels,
                "area_ratio": float(area_pixels / region_mask.size),
                "mask_path": mask_path_value,
                "color": list(color),
            }
        )
        six_zone_stack.append(region_mask.astype(np.uint8))

    overlay = create_overlay(image, region_payloads)
    overlay_path = image_output_dir / "overlay.png"
    overlay.save(overlay_path)

    np.savez_compressed(
        image_output_dir / "six_lung_zones.npz",
        right_upper_lung_zone=six_zone_stack[0],
        right_mid_lung_zone=six_zone_stack[1],
        right_lower_lung_zone=six_zone_stack[2],
        left_upper_lung_zone=six_zone_stack[3],
        left_mid_lung_zone=six_zone_stack[4],
        left_lower_lung_zone=six_zone_stack[5],
    )

    metadata = {
        "image_path": str(image_path),
        "overlay_path": str(overlay_path),
        "regions": region_payloads,
    }
    with open(image_output_dir / "regions.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Segment the COVIDx split and save six lung-zone masks only."
    )
    parser.add_argument("--data-dir", required=True, help="Directory containing COVIDx images.")
    parser.add_argument("--image-list", required=True, help="COVIDx manifest file.")
    parser.add_argument("--output-dir", required=True, help="Directory for saved masks and metadata.")
    parser.add_argument("--gpus", default="cpu", help="GPU setting for CXAS. Use 'cpu' for simple testing.")
    parser.add_argument("--limit", type=int, default=None, help="Optional limit for a quick dry run.")
    args = parser.parse_args()

    model = CXAS(model_name="UNet_ResNet50_default", gpus=args.gpus)
    label_index = build_label_index()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    records = parse_covidx_manifest(args.image_list, args.data_dir, limit=args.limit)
    manifest = []
    for record in records:
        metadata = process_single_image(model, Path(record.image_path), output_dir, label_index)
        metadata["image_id"] = record.image_id
        metadata["label"] = record.label
        metadata["source"] = record.source
        metadata["filename"] = record.filename
        manifest.append(metadata)

    with open(output_dir / "segmentation_manifest.json", "w", encoding="utf-8") as f:
        json.dump({"images": manifest}, f, indent=2)

    print(f"Processed {len(records)} images")
    print(f"Saved segmentation outputs to: {output_dir}")


if __name__ == "__main__":
    main()
