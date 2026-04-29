# COVIDx Region Graph Pipeline

This pipeline is designed for your current evaluation setup:

- dataset: COVIDx CXR
- split usage: test-only retrieval
- anatomical core: 6 lung zones
- feature extractor: trained ConvNeXtV2 from `Image-Retrieval---Thesis-2026`
- stop point: graph construction, not retrieval scoring

## Recommended tooling

Use two managed backends:

- `Neo4j AuraDB` for the knowledge graph
- `Zilliz Cloud / Milvus` later for vector retrieval

Why this pairing:

- Neo4j is strong for explicit region relationships and metadata queries.
- Zilliz is strong for dense vector search at retrieval time.
- Both work from Kaggle using environment variables and outbound network access.

For graph construction in this repo, the implementation supports:

- `local` graph artifact export as JSON
- optional upload to `Neo4j AuraDB`

## Outputs

Running the pipeline creates:

- `metadata/{image_id}.json`
- `masks/{image_id}/{region}.png`
- `crops/{image_id}/{region}.png`
- `embeddings/{image_id}/whole_image.npy`
- `embeddings/{image_id}/{region}.npy`
- `embeddings/{image_id}/all_embeddings.npz`
- `graph/graph_nodes.json`
- `graph/graph_edges.json`
- `graph/graph_summary.json`
- `pipeline_manifest.json`

## Graph schema

### Node labels

- `Image`
- `Region`
- `RegionType`

### Relationships

- `(:Image)-[:HAS_REGION]->(:Region)`
- `(:Region)-[:INSTANCE_OF]->(:RegionType)`
- `(:Region)-[:ABOVE]->(:Region)`
- `(:Region)-[:MIRRORS]->(:Region)`

The `ABOVE` edges encode upper-to-mid-to-lower ordering on each side.
The `MIRRORS` edges connect left and right regions at the same vertical level.

## Kaggle install

```bash
pip install -r requirements-covidx-pipeline.txt
```

You also need the segmentation repo requirements plus your retrieval repo dependencies.

## Kaggle example

```bash
python run_covidx_graph_pipeline.py \
  --data-dir /kaggle/input/covidx-test/images \
  --image-list /kaggle/input/covidx-test/test_COVIDx4.txt \
  --output-dir /kaggle/working/covidx_region_graph \
  --retrieval-repo /kaggle/working/Image-Retrieval---Thesis-2026 \
  --convnext-checkpoint /kaggle/working/Image-Retrieval---Thesis-2026/model.pth \
  --segmentor-gpus cpu \
  --feature-device cuda \
  --graph-backend local
```

## Neo4j AuraDB upload

Set credentials in Kaggle as secrets or environment variables:

```bash
export NEO4J_URI='neo4j+s://<your-instance>.databases.neo4j.io'
export NEO4J_USER='neo4j'
export NEO4J_PASSWORD='<your-password>'
```

Then run:

```bash
python run_covidx_graph_pipeline.py \
  --data-dir /kaggle/input/covidx-test/images \
  --image-list /kaggle/input/covidx-test/test_COVIDx4.txt \
  --output-dir /kaggle/working/covidx_region_graph \
  --retrieval-repo /kaggle/working/Image-Retrieval---Thesis-2026 \
  --convnext-checkpoint /kaggle/working/Image-Retrieval---Thesis-2026/model.pth \
  --segmentor-gpus cpu \
  --feature-device cuda \
  --graph-backend both
```

## Notes

- The graph stores metadata and file paths to embeddings, not full vectors inside Neo4j.
- This is deliberate. Later retrieval should use Milvus/Zilliz for vector search.
- The six-region design is a practical first version for COVIDx and maps well to diffuse and bilateral disease patterns.
