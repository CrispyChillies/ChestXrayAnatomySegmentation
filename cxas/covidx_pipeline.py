from __future__ import annotations

import importlib.util
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from cxas import CXAS
from cxas.label_mapper import id2label_dict


REGION_SPECS = [
    {
        "region_name": "right_upper_lung_zone",
        "cxas_label": "right upper zone lung",
        "side": "right",
        "level": "upper",
    },
    {
        "region_name": "right_mid_lung_zone",
        "cxas_label": "right mid zone lung",
        "side": "right",
        "level": "mid",
    },
    {
        "region_name": "right_lower_lung_zone",
        "cxas_label": "right lung base",
        "side": "right",
        "level": "lower",
    },
    {
        "region_name": "left_upper_lung_zone",
        "cxas_label": "left upper zone lung",
        "side": "left",
        "level": "upper",
    },
    {
        "region_name": "left_mid_lung_zone",
        "cxas_label": "left mid zone lung",
        "side": "left",
        "level": "mid",
    },
    {
        "region_name": "left_lower_lung_zone",
        "cxas_label": "left lung base",
        "side": "left",
        "level": "lower",
    },
]

LEVEL_ORDER = {"upper": 0, "mid": 1, "lower": 2}
IMAGE_NORMALIZE = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])


@dataclass
class CovidXRecord:
    image_id: str
    filename: str
    label: str
    source: str
    image_path: str


@dataclass
class RegionArtifact:
    region_id: str
    region_name: str
    cxas_label: str
    side: str
    level: str
    present: bool
    bbox_xyxy: list[int] | None
    area_pixels: int
    area_ratio: float
    mask_path: str | None
    crop_path: str | None
    embedding_path: str | None


def parse_covidx_manifest(
    image_list_file: str | Path,
    data_dir: str | Path,
    limit: int | None = None,
) -> list[CovidXRecord]:
    records: list[CovidXRecord] = []
    data_dir = Path(data_dir)
    with open(image_list_file, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 4:
                raise ValueError(
                    f"Expected COVIDx manifest line with 4 columns, got: {line!r}"
                )
            image_id, filename, label, source = parts[:4]
            image_path = data_dir / filename
            records.append(
                CovidXRecord(
                    image_id=image_id,
                    filename=filename,
                    label=label,
                    source=source,
                    image_path=str(image_path),
                )
            )
            if limit is not None and len(records) >= limit:
                break
    return records


def build_label_index() -> dict[str, int]:
    return {label_name: int(label_id) for label_id, label_name in id2label_dict.items()}


def sanitize_name(value: str) -> str:
    return value.lower().replace(" ", "_").replace("-", "_")


def mask_to_bbox(mask: np.ndarray) -> list[int] | None:
    ys, xs = np.where(mask)
    if xs.size == 0 or ys.size == 0:
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]


def expand_bbox(
    bbox_xyxy: list[int],
    image_width: int,
    image_height: int,
    pad_ratio: float,
) -> list[int]:
    x1, y1, x2, y2 = bbox_xyxy
    width = max(1, x2 - x1)
    height = max(1, y2 - y1)
    pad = int(max(width, height) * pad_ratio)
    return [
        max(0, x1 - pad),
        max(0, y1 - pad),
        min(image_width, x2 + pad),
        min(image_height, y2 + pad),
    ]


def save_binary_mask(mask: np.ndarray, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((mask.astype(np.uint8) * 255)).save(output_path)


def create_region_crop(
    image: Image.Image,
    mask: np.ndarray,
    bbox_xyxy: list[int],
    pad_ratio: float,
) -> Image.Image:
    image_np = np.array(image.convert("RGB"))
    x1, y1, x2, y2 = expand_bbox(bbox_xyxy, image_np.shape[1], image_np.shape[0], pad_ratio)
    masked = np.zeros_like(image_np)
    masked[mask] = image_np[mask]
    cropped = masked[y1:y2, x1:x2]
    if cropped.size == 0:
        cropped = image_np[y1:y2, x1:x2]
    return Image.fromarray(cropped)


def load_rgb_image(image_path: str | Path) -> Image.Image:
    return Image.open(image_path).convert("RGB")


def save_json(payload: Any, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


class ConvNeXtFeatureExtractor:
    def __init__(
        self,
        retrieval_repo: str | Path,
        checkpoint_path: str | Path,
        device: str = "cuda",
    ) -> None:
        self.device = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")
        model_module = self._load_model_module(Path(retrieval_repo) / "model.py")
        self.model = model_module.ConvNeXtV2(embedding_dim=None)
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        state_dict = checkpoint.get("state-dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
        cleaned_state_dict = self._clean_state_dict(state_dict)
        self.model.load_state_dict(cleaned_state_dict, strict=False)
        self.model.to(self.device)
        self.model.eval()
        self.transform = transforms.Compose(
            [
                transforms.Resize((384, 384)),
                transforms.ToTensor(),
                IMAGE_NORMALIZE,
            ]
        )

    @staticmethod
    def _load_model_module(model_py_path: Path):
        spec = importlib.util.spec_from_file_location("retrieval_model_module", model_py_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load retrieval model module from {model_py_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    @staticmethod
    def _clean_state_dict(state_dict: dict[str, Any]) -> dict[str, Any]:
        cleaned: dict[str, Any] = {}
        for key, value in state_dict.items():
            new_key = key[7:] if key.startswith("module.") else key
            cleaned[new_key] = value
        return cleaned

    @torch.no_grad()
    def embed_image(self, image: Image.Image) -> np.ndarray:
        tensor = self.transform(image).unsqueeze(0).to(self.device)
        embedding = self.model(tensor)[0].detach().cpu().numpy().astype(np.float32)
        return embedding


class GraphArtifactBuilder:
    def __init__(self) -> None:
        self.nodes: list[dict[str, Any]] = []
        self.edges: list[dict[str, Any]] = []

    def add_image_node(self, record: CovidXRecord, embedding_path: str) -> None:
        self.nodes.append(
            {
                "id": record.image_id,
                "labels": ["Image"],
                "properties": {
                    "image_id": record.image_id,
                    "filename": record.filename,
                    "label": record.label,
                    "source": record.source,
                    "image_path": record.image_path,
                    "whole_embedding_path": embedding_path,
                },
            }
        )

    def add_region_type_nodes(self) -> None:
        for spec in REGION_SPECS:
            self.nodes.append(
                {
                    "id": f"region_type::{spec['region_name']}",
                    "labels": ["RegionType"],
                    "properties": {
                        "name": spec["region_name"],
                        "cxas_label": spec["cxas_label"],
                        "side": spec["side"],
                        "level": spec["level"],
                    },
                }
            )

    def add_region_node(self, image_id: str, artifact: RegionArtifact) -> None:
        self.nodes.append(
            {
                "id": artifact.region_id,
                "labels": ["Region"],
                "properties": asdict(artifact),
            }
        )
        self.edges.append(
            {
                "type": "HAS_REGION",
                "source": image_id,
                "target": artifact.region_id,
                "properties": {
                    "present": artifact.present,
                    "area_ratio": artifact.area_ratio,
                },
            }
        )
        self.edges.append(
            {
                "type": "INSTANCE_OF",
                "source": artifact.region_id,
                "target": f"region_type::{artifact.region_name}",
                "properties": {},
            }
        )

    def add_spatial_edges(self, image_id: str, region_artifacts: list[RegionArtifact]) -> None:
        by_side: dict[str, list[RegionArtifact]] = {"left": [], "right": []}
        for artifact in region_artifacts:
            if artifact.present:
                by_side[artifact.side].append(artifact)

        for side, artifacts in by_side.items():
            ordered = sorted(artifacts, key=lambda item: LEVEL_ORDER[item.level])
            for current, nxt in zip(ordered, ordered[1:]):
                self.edges.append(
                    {
                        "type": "ABOVE",
                        "source": current.region_id,
                        "target": nxt.region_id,
                        "properties": {"image_id": image_id, "side": side},
                    }
                )

        for level in LEVEL_ORDER:
            left_match = next(
                (artifact for artifact in by_side["left"] if artifact.level == level),
                None,
            )
            right_match = next(
                (artifact for artifact in by_side["right"] if artifact.level == level),
                None,
            )
            if left_match and right_match:
                self.edges.append(
                    {
                        "type": "MIRRORS",
                        "source": left_match.region_id,
                        "target": right_match.region_id,
                        "properties": {"image_id": image_id, "level": level},
                    }
                )

    def save(self, output_dir: str | Path) -> None:
        output_dir = Path(output_dir)
        save_json({"nodes": self.nodes}, output_dir / "graph_nodes.json")
        save_json({"edges": self.edges}, output_dir / "graph_edges.json")
        save_json(
            {
                "num_nodes": len(self.nodes),
                "num_edges": len(self.edges),
            },
            output_dir / "graph_summary.json",
        )


class Neo4jGraphUploader:
    def __init__(self, uri: str, user: str, password: str, database: str = "neo4j") -> None:
        from neo4j import GraphDatabase

        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.database = database

    def close(self) -> None:
        self.driver.close()

    def upload(self, nodes: list[dict[str, Any]], edges: list[dict[str, Any]]) -> None:
        with self.driver.session(database=self.database) as session:
            session.run(
                "CREATE CONSTRAINT image_id IF NOT EXISTS FOR (n:Image) REQUIRE n.image_id IS UNIQUE"
            )
            session.run(
                "CREATE CONSTRAINT region_type_name IF NOT EXISTS FOR (n:RegionType) REQUIRE n.name IS UNIQUE"
            )
            session.run(
                "CREATE CONSTRAINT region_id IF NOT EXISTS FOR (n:Region) REQUIRE n.region_id IS UNIQUE"
            )

            image_rows = [node["properties"] for node in nodes if "Image" in node["labels"]]
            region_type_rows = [node["properties"] for node in nodes if "RegionType" in node["labels"]]
            region_rows = [node["properties"] for node in nodes if "Region" in node["labels"]]

            if image_rows:
                session.run(
                    """
                    UNWIND $rows AS row
                    MERGE (n:Image {image_id: row.image_id})
                    SET n += row
                    """,
                    rows=image_rows,
                )
            if region_type_rows:
                session.run(
                    """
                    UNWIND $rows AS row
                    MERGE (n:RegionType {name: row.name})
                    SET n += row
                    """,
                    rows=region_type_rows,
                )
            if region_rows:
                session.run(
                    """
                    UNWIND $rows AS row
                    MERGE (n:Region {region_id: row.region_id})
                    SET n += row
                    """,
                    rows=region_rows,
                )

            self._upload_edge_type(session, edges, "HAS_REGION")
            self._upload_edge_type(session, edges, "INSTANCE_OF")
            self._upload_edge_type(session, edges, "ABOVE")
            self._upload_edge_type(session, edges, "MIRRORS")

    def _upload_edge_type(self, session, edges: list[dict[str, Any]], edge_type: str) -> None:
        rows = [edge for edge in edges if edge["type"] == edge_type]
        if not rows:
            return

        if edge_type == "HAS_REGION":
            session.run(
                """
                UNWIND $rows AS row
                MATCH (a:Image {image_id: row.source})
                MATCH (b:Region {region_id: row.target})
                MERGE (a)-[r:HAS_REGION]->(b)
                SET r += row.properties
                """,
                rows=rows,
            )
        elif edge_type == "INSTANCE_OF":
            session.run(
                """
                UNWIND $rows AS row
                MATCH (a:Region {region_id: row.source})
                MATCH (b:RegionType {name: replace(row.target, 'region_type::', '')})
                MERGE (a)-[r:INSTANCE_OF]->(b)
                SET r += row.properties
                """,
                rows=rows,
            )
        elif edge_type in {"ABOVE", "MIRRORS"}:
            session.run(
                f"""
                UNWIND $rows AS row
                MATCH (a:Region {{region_id: row.source}})
                MATCH (b:Region {{region_id: row.target}})
                MERGE (a)-[r:{edge_type}]->(b)
                SET r += row.properties
                """,
                rows=rows,
            )


class CovidXGraphPipeline:
    def __init__(
        self,
        segmentation_model: CXAS,
        feature_extractor: ConvNeXtFeatureExtractor,
        output_dir: str | Path,
        crop_pad_ratio: float = 0.08,
    ) -> None:
        self.segmentation_model = segmentation_model
        self.feature_extractor = feature_extractor
        self.output_dir = Path(output_dir)
        self.crop_pad_ratio = crop_pad_ratio
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.label_index = build_label_index()
        self.graph = GraphArtifactBuilder()
        self.graph.add_region_type_nodes()

    def process_records(
        self,
        records: list[CovidXRecord],
        skip_existing: bool = False,
    ) -> GraphArtifactBuilder:
        manifests: list[dict[str, Any]] = []
        for record in records:
            image_meta_path = self.output_dir / "metadata" / f"{record.image_id}.json"
            if skip_existing and image_meta_path.exists():
                with open(image_meta_path, "r", encoding="utf-8") as f:
                    manifest = json.load(f)
                self._hydrate_graph_from_manifest(manifest)
                manifests.append(manifest)
                continue

            manifest = self.process_single_record(record)
            manifests.append(manifest)

        save_json({"images": manifests}, self.output_dir / "pipeline_manifest.json")
        self.graph.save(self.output_dir / "graph")
        return self.graph

    def process_single_record(self, record: CovidXRecord) -> dict[str, Any]:
        predictions = self.segmentation_model.process_file(record.image_path)
        resized_mask = self.segmentation_model.resize_to_numpy(
            predictions["segmentation_preds"][0],
            predictions["file_size"][0],
        )
        image = load_rgb_image(record.image_path)

        whole_embedding = self.feature_extractor.embed_image(image)
        embedding_dir = self.output_dir / "embeddings" / record.image_id
        embedding_dir.mkdir(parents=True, exist_ok=True)
        whole_embedding_path = embedding_dir / "whole_image.npy"
        np.save(whole_embedding_path, whole_embedding)

        self.graph.add_image_node(record, str(whole_embedding_path))

        region_artifacts: list[RegionArtifact] = []
        region_embedding_payload: dict[str, np.ndarray] = {"whole_image": whole_embedding}

        for spec in REGION_SPECS:
            label_idx = self.label_index[spec["cxas_label"]]
            region_mask = resized_mask[label_idx].astype(bool)
            bbox = mask_to_bbox(region_mask)
            area_pixels = int(region_mask.sum())
            area_ratio = float(area_pixels / region_mask.size)
            mask_path: str | None = None
            crop_path: str | None = None
            embedding_path: str | None = None
            present = bbox is not None and area_pixels > 0

            if present:
                mask_output = self.output_dir / "masks" / record.image_id / f"{spec['region_name']}.png"
                save_binary_mask(region_mask, mask_output)
                mask_path = str(mask_output)

                crop_image = create_region_crop(image, region_mask, bbox, self.crop_pad_ratio)
                crop_output = self.output_dir / "crops" / record.image_id / f"{spec['region_name']}.png"
                crop_output.parent.mkdir(parents=True, exist_ok=True)
                crop_image.save(crop_output)
                crop_path = str(crop_output)

                region_embedding = self.feature_extractor.embed_image(crop_image)
                region_embedding_file = embedding_dir / f"{spec['region_name']}.npy"
                np.save(region_embedding_file, region_embedding)
                embedding_path = str(region_embedding_file)
                region_embedding_payload[spec["region_name"]] = region_embedding

            artifact = RegionArtifact(
                region_id=f"{record.image_id}::{spec['region_name']}",
                region_name=spec["region_name"],
                cxas_label=spec["cxas_label"],
                side=spec["side"],
                level=spec["level"],
                present=present,
                bbox_xyxy=bbox,
                area_pixels=area_pixels,
                area_ratio=area_ratio,
                mask_path=mask_path,
                crop_path=crop_path,
                embedding_path=embedding_path,
            )
            region_artifacts.append(artifact)
            self.graph.add_region_node(record.image_id, artifact)

        np.savez_compressed(embedding_dir / "all_embeddings.npz", **region_embedding_payload)
        self.graph.add_spatial_edges(record.image_id, region_artifacts)

        manifest = {
            "image": asdict(record),
            "whole_embedding_path": str(whole_embedding_path),
            "regions": [asdict(artifact) for artifact in region_artifacts],
        }
        save_json(manifest, self.output_dir / "metadata" / f"{record.image_id}.json")
        return manifest

    def _hydrate_graph_from_manifest(self, manifest: dict[str, Any]) -> None:
        image_payload = manifest["image"]
        record = CovidXRecord(**image_payload)
        whole_embedding_path = manifest["whole_embedding_path"]
        self.graph.add_image_node(record, whole_embedding_path)

        region_artifacts = [RegionArtifact(**region_payload) for region_payload in manifest["regions"]]
        for artifact in region_artifacts:
            self.graph.add_region_node(record.image_id, artifact)
        self.graph.add_spatial_edges(record.image_id, region_artifacts)
