from __future__ import annotations

import argparse
import os

from cxas import CXAS
from cxas.covidx_pipeline import (
    ConvNeXtFeatureExtractor,
    CovidXGraphPipeline,
    Neo4jGraphUploader,
    parse_covidx_manifest,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="COVIDx test-only pipeline: segmentation, six lung-zone crops, ConvNeXtV2 features, and graph construction."
    )
    parser.add_argument("--data-dir", required=True, help="Directory containing COVIDx test images.")
    parser.add_argument("--image-list", required=True, help="COVIDx manifest file.")
    parser.add_argument("--output-dir", required=True, help="Directory where masks, crops, embeddings, and graph files are saved.")
    parser.add_argument(
        "--retrieval-repo",
        required=True,
        help="Path to the Image-Retrieval---Thesis-2026 repo containing model.py.",
    )
    parser.add_argument(
        "--convnext-checkpoint",
        required=True,
        help="Path to the trained ConvNeXtV2 checkpoint.",
    )
    parser.add_argument(
        "--segmentor-gpus",
        default="cpu",
        help="CXAS GPU selection. Use 'cpu' on Kaggle if segmentation GPU support is unstable.",
    )
    parser.add_argument(
        "--feature-device",
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device for ConvNeXtV2 feature extraction.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Optional limit for quick testing.")
    parser.add_argument("--skip-existing", action="store_true", help="Skip images with existing metadata output.")
    parser.add_argument("--crop-pad-ratio", type=float, default=0.08, help="Padding ratio applied around region bounding boxes.")
    parser.add_argument(
        "--graph-backend",
        default="local",
        choices=["local", "neo4j", "both"],
        help="Where to store the graph artifact.",
    )
    parser.add_argument("--neo4j-uri", default=None, help="Neo4j AuraDB bolt URI. Can also be set via NEO4J_URI.")
    parser.add_argument("--neo4j-user", default=None, help="Neo4j username. Can also be set via NEO4J_USER.")
    parser.add_argument("--neo4j-password", default=None, help="Neo4j password. Can also be set via NEO4J_PASSWORD.")
    parser.add_argument("--neo4j-database", default="neo4j", help="Neo4j database name.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = parse_covidx_manifest(args.image_list, args.data_dir, limit=args.limit)

    segmentor = CXAS(model_name="UNet_ResNet50_default", gpus=args.segmentor_gpus)
    extractor = ConvNeXtFeatureExtractor(
        retrieval_repo=args.retrieval_repo,
        checkpoint_path=args.convnext_checkpoint,
        device=args.feature_device,
    )
    pipeline = CovidXGraphPipeline(
        segmentation_model=segmentor,
        feature_extractor=extractor,
        output_dir=args.output_dir,
        crop_pad_ratio=args.crop_pad_ratio,
    )
    graph = pipeline.process_records(records, skip_existing=args.skip_existing)

    if args.graph_backend in {"neo4j", "both"}:
        neo4j_uri = args.neo4j_uri or os.getenv("NEO4J_URI")
        neo4j_user = args.neo4j_user or os.getenv("NEO4J_USER")
        neo4j_password = args.neo4j_password or os.getenv("NEO4J_PASSWORD")
        if not neo4j_uri or not neo4j_user or not neo4j_password:
            raise ValueError(
                "Neo4j upload requested but credentials are missing. Set --neo4j-uri/--neo4j-user/--neo4j-password or use environment variables."
            )
        uploader = Neo4jGraphUploader(
            uri=neo4j_uri,
            user=neo4j_user,
            password=neo4j_password,
            database=args.neo4j_database,
        )
        try:
            uploader.upload(graph.nodes, graph.edges)
        finally:
            uploader.close()

    print(f"Processed {len(records)} images.")
    print(f"Artifacts saved to: {args.output_dir}")
    print("Pipeline completed through graph construction.")


if __name__ == "__main__":
    main()
