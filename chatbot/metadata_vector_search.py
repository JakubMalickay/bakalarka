from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

# Force transformers/sentence-transformers to avoid TensorFlow imports.
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("USE_TF", "0")

try:
    import faiss  # type: ignore
except ImportError as exc:  # pragma: no cover - depends on local env
    raise ImportError(
        "FAISS is required for the vector metadata workflow. Install 'faiss-cpu'."
    ) from exc

try:
    from sentence_transformers import SentenceTransformer
except ImportError as exc:  # pragma: no cover - depends on local env
    raise ImportError(
        "sentence-transformers is required for high-quality embeddings. Install 'sentence-transformers'."
    ) from exc


@dataclass
class VectorSearchResult:
    rank: int
    score: float
    object_type: str
    name: str
    unique_name: str | None
    payload: dict[str, Any]
    search_text: str


def _build_search_result(item: dict[str, Any], rank: int, score: float) -> VectorSearchResult:
    return VectorSearchResult(
        rank=rank,
        score=float(score),
        object_type=item["object_type"],
        name=item["name"],
        unique_name=item.get("unique_name"),
        payload=item["payload"],
        search_text=item["search_text"],
    )


class SentenceTransformerEmbedder:
    """High-quality semantic embedder backed by sentence-transformers."""

    def __init__(self, model_name: str, device: str = "auto") -> None:
        self.model_name = model_name
        self.device = device

        # Let sentence-transformers choose automatically unless explicitly pinned.
        if device == "auto":
            self.model = SentenceTransformer(model_name)
        else:
            self.model = SentenceTransformer(model_name, device=device)

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        vectors = self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return vectors.astype(np.float32)


class OLAPMetadataVectorStore:
    def __init__(
        self,
        dimensions_path: Path,
        measures_path: Path,
        db_dir: Path,
        embedding_model: str = "BAAI/bge-large-en-v1.5",
        embedding_device: str = "auto",
        vector_dimensions: int = 512,
    ) -> None:
        self.dimensions_path = dimensions_path
        self.measures_path = measures_path
        self.db_dir = db_dir
        self.vector_dimensions = vector_dimensions
        self.embedding_model = embedding_model
        self.embedding_device = embedding_device
        self.embedder = SentenceTransformerEmbedder(model_name=embedding_model, device=embedding_device)
        self.index_path = db_dir / "olap_all_objects.faiss"
        self.meta_path = db_dir / "olap_all_objects_metadata.json"
        self.manifest_path = db_dir / "olap_all_objects_manifest.json"

        self.dimension_index_path = db_dir / "olap_dimensions.faiss"
        self.dimension_meta_path = db_dir / "olap_dimensions_metadata.json"
        self.dimension_manifest_path = db_dir / "olap_dimensions_manifest.json"

        self.child_index_path = db_dir / "olap_dimension_children.faiss"
        self.child_meta_path = db_dir / "olap_dimension_children_metadata.json"
        self.child_manifest_path = db_dir / "olap_dimension_children_manifest.json"

        self.dim_child_index_path = db_dir / "olap_dimensions_and_children.faiss"
        self.dim_child_meta_path = db_dir / "olap_dimensions_and_children_metadata.json"
        self.dim_child_manifest_path = db_dir / "olap_dimensions_and_children_manifest.json"

        self.measure_index_path = db_dir / "olap_measures.faiss"
        self.measure_meta_path = db_dir / "olap_measures_metadata.json"
        self.measure_manifest_path = db_dir / "olap_measures_manifest.json"

        self.db_dir.mkdir(parents=True, exist_ok=True)
        self.index = None
        self.metadata: list[dict[str, Any]] = []
        self.dimension_index = None
        self.dimension_metadata: list[dict[str, Any]] = []
        self.child_index = None
        self.child_metadata: list[dict[str, Any]] = []
        self.dim_child_index = None
        self.dim_child_metadata: list[dict[str, Any]] = []
        self.measure_index = None
        self.measure_metadata: list[dict[str, Any]] = []
        self._load_or_build()

    def search(self, query: str, top_k: int = 20) -> list[VectorSearchResult]:
        return self._search_index(self.index, self.metadata, query, top_k, "Vector index")

    def search_dimensions(self, query: str, top_k: int = 5) -> list[VectorSearchResult]:
        return self._search_index(
            self.dimension_index,
            self.dimension_metadata,
            query,
            top_k,
            "Dimension vector index",
        )

    def search_children_in_dimensions(
        self,
        query: str,
        dimension_unique_names: list[str],
        top_k_per_dimension: int = 5,
    ) -> list[VectorSearchResult]:
        if self.child_index is None:
            raise RuntimeError("Child vector index is not initialized")
        if not dimension_unique_names:
            return []

        query_vector = self.embedder.embed_texts([query])
        # Fetch a broader candidate set, then keep only children of selected dimensions.
        child_total = len(self.child_metadata)
        overfetch = min(child_total, max(top_k_per_dimension * max(len(dimension_unique_names), 1) * 8, 50))
        distances, indices = self.child_index.search(query_vector, overfetch)

        allowed_dims = set(dimension_unique_names)
        per_dim_counts = {dim_unique: 0 for dim_unique in allowed_dims}

        results: list[VectorSearchResult] = []
        for score, index in zip(distances[0], indices[0]):
            if index < 0:
                continue
            item = self.child_metadata[index]
            parent_unique = item.get("parent_dimension_unique_name")
            if parent_unique not in allowed_dims:
                continue
            if per_dim_counts[parent_unique] >= top_k_per_dimension:
                continue

            per_dim_counts[parent_unique] += 1
            results.append(
                VectorSearchResult(
                    rank=len(results) + 1,
                    score=float(score),
                    object_type=item["object_type"],
                    name=item["name"],
                    unique_name=item.get("unique_name"),
                    payload=item["payload"],
                    search_text=item["search_text"],
                )
            )

            if all(count >= top_k_per_dimension for count in per_dim_counts.values()):
                break

        return results

    def search_measures(self, query: str, top_k: int = 10) -> list[VectorSearchResult]:
        return self._search_index(
            self.measure_index,
            self.measure_metadata,
            query,
            top_k,
            "Measure vector index",
        )

    def search_dimensions_and_children(self, query: str, top_k: int = 15) -> list[VectorSearchResult]:
        return self._search_index(
            self.dim_child_index,
            self.dim_child_metadata,
            query,
            top_k,
            "Dimensions and children vector index",
        )

    def _search_index(
        self,
        index: faiss.Index | None,
        metadata: list[dict[str, Any]],
        query: str,
        top_k: int,
        label: str,
    ) -> list[VectorSearchResult]:
        if index is None:
            raise RuntimeError(f"{label} is not initialized")

        query_vector = self.embedder.embed_texts([query])
        distances, indices = index.search(query_vector, min(top_k, len(metadata)))

        results: list[VectorSearchResult] = []
        for rank, (score, item_index) in enumerate(zip(distances[0], indices[0]), start=1):
            if item_index < 0:
                continue
            results.append(_build_search_result(metadata[item_index], rank, float(score)))
        return results

    def _load_or_build(self) -> None:
        if self._is_current():
            print(f"[Vector DB] Loading existing index from {self.index_path}")
            self.index, self.metadata = self._load_bundle(self.index_path, self.meta_path)
            self.dimension_index, self.dimension_metadata = self._load_bundle(
                self.dimension_index_path,
                self.dimension_meta_path,
            )
            self.child_index, self.child_metadata = self._load_bundle(self.child_index_path, self.child_meta_path)
            self.dim_child_index, self.dim_child_metadata = self._load_bundle(
                self.dim_child_index_path,
                self.dim_child_meta_path,
            )
            self.measure_index, self.measure_metadata = self._load_bundle(
                self.measure_index_path,
                self.measure_meta_path,
            )
            return

        print("[Vector DB] Building index (source or embedding settings changed)")
        self._build_index()

    def _is_current(self) -> bool:
        required = [
            self.index_path,
            self.meta_path,
            self.manifest_path,
            self.dimension_index_path,
            self.dimension_meta_path,
            self.dimension_manifest_path,
            self.child_index_path,
            self.child_meta_path,
            self.child_manifest_path,
            self.dim_child_index_path,
            self.dim_child_meta_path,
            self.dim_child_manifest_path,
            self.measure_index_path,
            self.measure_meta_path,
            self.measure_manifest_path,
        ]
        if any(not path.exists() for path in required):
            return False

        manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        dimension_manifest = json.loads(self.dimension_manifest_path.read_text(encoding="utf-8"))
        child_manifest = json.loads(self.child_manifest_path.read_text(encoding="utf-8"))
        dim_child_manifest = json.loads(self.dim_child_manifest_path.read_text(encoding="utf-8"))
        measure_manifest = json.loads(self.measure_manifest_path.read_text(encoding="utf-8"))
        expected = self._source_manifest()
        return (
            all(manifest.get(key) == value for key, value in expected.items())
            and all(dimension_manifest.get(key) == value for key, value in expected.items())
            and all(child_manifest.get(key) == value for key, value in expected.items())
            and all(dim_child_manifest.get(key) == value for key, value in expected.items())
            and all(measure_manifest.get(key) == value for key, value in expected.items())
        )

    def _build_index(self) -> None:
        self.metadata = self._collect_objects()
        self.dimension_metadata = [item for item in self.metadata if item["object_type"] == "dimension"]
        self.child_metadata = [
            item for item in self.metadata if item["object_type"] in {"level", "attribute"}
        ]
        self.dim_child_metadata = self.dimension_metadata + self.child_metadata
        self.measure_metadata = [item for item in self.metadata if item["object_type"] == "measure"]

        self.index = self._build_and_store_bundle(
            self.index_path,
            self.meta_path,
            self.manifest_path,
            self.metadata,
            update_vector_dimensions=True,
        )
        self.dimension_index = self._build_and_store_bundle(
            self.dimension_index_path,
            self.dimension_meta_path,
            self.dimension_manifest_path,
            self.dimension_metadata,
        )
        self.child_index = self._build_and_store_bundle(
            self.child_index_path,
            self.child_meta_path,
            self.child_manifest_path,
            self.child_metadata,
        )
        self.dim_child_index = self._build_and_store_bundle(
            self.dim_child_index_path,
            self.dim_child_meta_path,
            self.dim_child_manifest_path,
            self.dim_child_metadata,
        )
        self.measure_index = self._build_and_store_bundle(
            self.measure_index_path,
            self.measure_meta_path,
            self.measure_manifest_path,
            self.measure_metadata,
        )

    def _load_bundle(self, index_path: Path, meta_path: Path) -> tuple[faiss.Index, list[dict[str, Any]]]:
        return faiss.read_index(str(index_path)), json.loads(meta_path.read_text(encoding="utf-8"))

    def _build_and_store_bundle(
        self,
        index_path: Path,
        meta_path: Path,
        manifest_path: Path,
        metadata: list[dict[str, Any]],
        update_vector_dimensions: bool = False,
    ) -> faiss.Index:
        vectors = self.embedder.embed_texts([item["search_text"] for item in metadata])
        index = faiss.IndexFlatIP(int(vectors.shape[1]))
        index.add(vectors)
        faiss.write_index(index, str(index_path))
        meta_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
        manifest_path.write_text(json.dumps(self._source_manifest(), indent=2), encoding="utf-8")
        if update_vector_dimensions:
            self.vector_dimensions = int(vectors.shape[1])
        return index

    def _source_manifest(self) -> dict[str, Any]:
        return {
            "dimensions_path": str(self.dimensions_path),
            "measures_path": str(self.measures_path),
            "dimensions_mtime": self.dimensions_path.stat().st_mtime,
            "measures_mtime": self.measures_path.stat().st_mtime,
            "embedding_model": self.embedding_model,
            "embedding_device": self.embedding_device,
        }

    def _collect_objects(self) -> list[dict[str, Any]]:
        dimension_data = json.loads(self.dimensions_path.read_text(encoding="utf-8"))[0]
        measure_data = json.loads(self.measures_path.read_text(encoding="utf-8"))[0]

        objects: list[dict[str, Any]] = []

        for dimension in dimension_data.get("dimensions", []):
            dim_unique_name = dimension.get("unique_name")
            objects.append(
                self._record(
                    object_type="dimension",
                    payload=dimension,
                    extra_context=f"cube {dimension_data.get('cube', '')}",
                )
            )
            for level in dimension.get("levels", []):
                objects.append(
                    self._record(
                        object_type="level",
                        payload=level,
                        extra_context=f"parent dimension {dimension.get('name', '')}",
                        parent_dimension_unique_name=dim_unique_name,
                    )
                )
            for attribute in dimension.get("attributes", []):
                objects.append(
                    self._record(
                        object_type="attribute",
                        payload=attribute,
                        extra_context=f"parent dimension {dimension.get('name', '')}",
                        parent_dimension_unique_name=dim_unique_name,
                    )
                )

        for measure in measure_data.get("measures", []):
            objects.append(
                self._record(
                    object_type="measure",
                    payload=measure,
                    extra_context=f"measure group {measure.get('measure_group', '')}",
                )
            )

        return objects

    def _record(
        self,
        object_type: str,
        payload: dict[str, Any],
        extra_context: str,
        parent_dimension_unique_name: str | None = None,
    ) -> dict[str, Any]:
        description = payload.get("description") or ""
        search_text = " | ".join(
            part
            for part in [
                object_type,
                payload.get("name", ""),
                payload.get("unique_name", ""),
                extra_context,
                description,
            ]
            if part
        )
        return {
            "object_type": object_type,
            "name": payload.get("name", ""),
            "unique_name": payload.get("unique_name"),
            "search_text": search_text,
            "payload": payload,
            "parent_dimension_unique_name": parent_dimension_unique_name,
        }