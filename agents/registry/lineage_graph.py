"""
agents/registry/lineage_graph.py

Phase 3 — Model Lineage Graph (Section 3.2).

Builds and queries the directed acyclic graph (DAG) that represents the full
provenance chain from raw data to live inference for every model version:

    Raw Dataset (data_lineage.id)
        │ produced_features
    Feature Engineering (feature_store_events.id)
        │ consumed_by
    Training Run (MLflow run_id)
        │ produced
    Model Artifact (MLflow artifact_uri)
        │ registered_as
    Evaluation Run (MLflow run_id)
        │ approved_by_eval
    Model Version (model_name/version)
        │ packaged_as          [Phase 4 — OCI image]
    OCI Image (registry/image@sha256:digest)
        │ deployed_as          [Phase 4 — KServe]
    InferenceService (KServe CRD name)
        │ receives_traffic     [Phase 4+]
    Live Traffic (Prometheus label set)

Nodes are typed via LineageNodeType enum.  Each node stores an external_id —
the natural identifier from the source system (MLflow run ID, data_lineage row
integer, etc.) — plus a metadata JSONB blob.

Phase 3 populates nodes up to "model_version".  Phase 4 adds oci_image onward.
The graph is queryable in full from the UI without knowing the internal node IDs.

Thread safety: LineageGraphBuilder is stateful (holds model_name + model_version)
and intended to be instantiated once per pipeline execution.
"""
from __future__ import annotations

import logging
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

logger = logging.getLogger(__name__)


# Node type enum 
class LineageNodeType(str, Enum):
    """
    Ordered by position in the lineage DAG (root → leaf).
    Phase 3 nodes: DATASET through MODEL_VERSION.
    Phase 4 nodes: OCI_IMAGE, INFER_SERVICE, LIVE_TRAFFIC.
    """
    DATASET          = "dataset"
    FEAT_ENG         = "feat_eng"
    TRAINING_RUN     = "training_run"
    MODEL_ARTIFACT   = "model_artifact"
    EVAL_RUN         = "eval_run"
    MODEL_VERSION    = "model_version"
    OCI_IMAGE        = "oci_image"        # Phase 4
    INFER_SERVICE    = "infer_service"    # Phase 4
    LIVE_TRAFFIC     = "live_traffic"     # Phase 4+


# Edge relationship labels 
class EdgeRelationship(str, Enum):
    """Semantic labels for directed edges in the lineage DAG."""
    PRODUCED_FEATURES  = "produced_features"   # dataset → feat_eng
    CONSUMED_BY        = "consumed_by"          # feat_eng → training_run
    PRODUCED           = "produced"             # training_run → model_artifact
    REGISTERED_AS      = "registered_as"        # model_artifact → model_version
    EVALUATED_BY       = "evaluated_by"         # model_version → eval_run
    PACKAGED_AS        = "packaged_as"          # model_version → oci_image   (Phase 4)
    DEPLOYED_AS        = "deployed_as"          # oci_image → infer_service   (Phase 4)
    RECEIVES_TRAFFIC   = "receives_traffic"     # infer_service → live_traffic (Phase 4+)


# Typed node and edge dataclasses 
@dataclass
class LineageNode:
    """In-memory representation of a lineage DAG node."""
    model_name:    str
    model_version: str
    node_type:     LineageNodeType
    external_id:   str                     # natural ID from the external system
    display_label: str
    metadata:      dict[str, Any] = field(default_factory=dict)
    db_id:         Optional[int] = None    # set after DB INSERT


@dataclass
class LineageEdge:
    """In-memory representation of a directed lineage edge."""
    parent_node_id: int
    child_node_id:  int
    relationship:   str
    db_id:          Optional[int] = None


# Graph builder 
class LineageGraphBuilder:
    """
    Incrementally builds the lineage DAG in Neon as each pipeline stage completes.

    Instantiate once per pipeline execution.  Call record_* methods in order as
    each stage completes.  Each method returns the DB row ID of the inserted node,
    which must be passed as the parent_node_id to the next stage's record_* call.

    Args:
        pool: asyncpg.Pool connected to Neon.
        model_name: MLflow registered model name.
        model_version: MLflow model version string (e.g. "3").
    """

    def __init__(self, pool: Any, model_name: str, model_version: str) -> None:
        self._pool = pool
        self._model_name = model_name
        self._model_version = model_version

    async def add_node(
        self,
        node_type: LineageNodeType,
        external_id: str,
        display_label: str,
        metadata: Optional[dict[str, Any]] = None,
    ) -> int:
        """
        Insert a lineage node into model_lineage_nodes.

        Returns:
            The DB primary key (id) of the inserted row.

        Note:
            If a node with the same (model_name, model_version, node_type, external_id)
            already exists, the existing row ID is returned (idempotent).
        """
        import json
        meta_json = json.dumps(metadata or {})
        async with self._pool.acquire() as conn:
            row_id: int = await conn.fetchval(
                """
                INSERT INTO model_lineage_nodes
                    (model_name, model_version, node_type, external_id, display_label, metadata)
                VALUES ($1, $2, $3, $4, $5, $6::jsonb)
                ON CONFLICT DO NOTHING
                RETURNING id
                """,
                self._model_name,
                self._model_version,
                node_type.value,
                external_id,
                display_label,
                meta_json,
            )
            if row_id is None:
                # Row already existed (ON CONFLICT DO NOTHING suppressed the RETURNING)
                row_id = await conn.fetchval(
                    """
                    SELECT id FROM model_lineage_nodes
                    WHERE model_name=$1 AND model_version=$2
                      AND node_type=$3 AND external_id=$4
                    """,
                    self._model_name, self._model_version,
                    node_type.value, external_id,
                )
        logger.debug(
            "Lineage node: model=%s ver=%s type=%s external_id=%s db_id=%d",
            self._model_name, self._model_version, node_type.value, external_id, row_id,
        )
        return row_id

    async def add_edge(
        self,
        parent_node_id: int,
        child_node_id: int,
        relationship: EdgeRelationship,
    ) -> None:
        """
        Insert a directed edge.  Idempotent — duplicate edges are silently ignored.
        """
        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO model_lineage_edges (parent_node_id, child_node_id, relationship)
                VALUES ($1, $2, $3)
                ON CONFLICT (parent_node_id, child_node_id, relationship) DO NOTHING
                """,
                parent_node_id,
                child_node_id,
                relationship.value,
            )

    # Semantic record methods 
    async def record_data_ingestion(
        self,
        lineage_id: int,
        content_hash: str,
        source_uri: str,
        row_count: int,
    ) -> int:
        """
        Add a dataset node representing a Phase 1 data_lineage record.

        Args:
            lineage_id: The PK from data_lineage table.
            content_hash: SHA-256 of the raw dataset bytes.
            source_uri: R2/S3 URI or file path.
            row_count: Number of rows in the dataset.

        Returns:
            The DB node ID.
        """
        return await self.add_node(
            node_type=LineageNodeType.DATASET,
            external_id=str(lineage_id),
            display_label=f"Dataset — {source_uri.split('/')[-1]} ({row_count:,} rows)",
            metadata={
                "lineage_id":   lineage_id,
                "content_hash": content_hash,
                "source_uri":   source_uri,
                "row_count":    row_count,
            },
        )

    async def record_feature_engineering(
        self,
        feast_event_id: int,
        feature_view: str,
        rows_written: int,
        parent_dataset_node_id: int,
    ) -> int:
        """
        Add a feature engineering node after Feast materialization.

        Args:
            feast_event_id: PK from feature_store_events table.
            feature_view: Feast feature view name.
            rows_written: Features written in this materialization run.
            parent_dataset_node_id: DB id of the upstream dataset node.

        Returns:
            The DB node ID.
        """
        node_id = await self.add_node(
            node_type=LineageNodeType.FEAT_ENG,
            external_id=str(feast_event_id),
            display_label=f"Feature Engineering — {feature_view} ({rows_written:,} rows)",
            metadata={
                "feast_event_id": feast_event_id,
                "feature_view":   feature_view,
                "rows_written":   rows_written,
            },
        )
        await self.add_edge(parent_dataset_node_id, node_id, EdgeRelationship.PRODUCED_FEATURES)
        return node_id

    async def record_training_run(
        self,
        run_id: str,
        framework: str,
        hyperparams: dict[str, Any],
        parent_feat_node_id: int,
    ) -> int:
        """
        Add a training run node from an MLflow run ID.

        Args:
            run_id: MLflow run ID.
            framework: e.g. "xgboost", "pytorch".
            hyperparams: Key training hyperparameters for the lineage record.
            parent_feat_node_id: DB id of the upstream feature engineering node.

        Returns:
            The DB node ID.
        """
        node_id = await self.add_node(
            node_type=LineageNodeType.TRAINING_RUN,
            external_id=run_id,
            display_label=f"Training Run — {framework} ({run_id[:8]}…)",
            metadata={
                "run_id":      run_id,
                "framework":   framework,
                "hyperparams": hyperparams,
            },
        )
        await self.add_edge(parent_feat_node_id, node_id, EdgeRelationship.CONSUMED_BY)
        return node_id

    async def record_model_artifact(
        self,
        artifact_uri: str,
        parent_run_node_id: int,
    ) -> int:
        """
        Add a model artifact node from an MLflow artifact URI.

        Args:
            artifact_uri: e.g. "runs:/abc123/model".
            parent_run_node_id: DB id of the upstream training run node.

        Returns:
            The DB node ID.
        """
        node_id = await self.add_node(
            node_type=LineageNodeType.MODEL_ARTIFACT,
            external_id=artifact_uri,
            display_label=f"Model Artifact — {artifact_uri}",
            metadata={"artifact_uri": artifact_uri},
        )
        await self.add_edge(parent_run_node_id, node_id, EdgeRelationship.PRODUCED)
        return node_id

    async def record_evaluation_run(
        self,
        eval_run_id: str,
        metrics: dict[str, float],
        parent_artifact_node_id: int,
    ) -> int:
        """
        Add an evaluation run node from an MLflow eval run ID.

        Args:
            eval_run_id: MLflow run ID for the evaluation run.
            metrics: Dict of metric names to values (accuracy, f1, auc, etc.).
            parent_artifact_node_id: DB id of the upstream model artifact node.

        Returns:
            The DB node ID.
        """
        accuracy = metrics.get("accuracy", 0.0)
        node_id = await self.add_node(
            node_type=LineageNodeType.EVAL_RUN,
            external_id=eval_run_id,
            display_label=f"Evaluation Run — accuracy={accuracy:.4f} ({eval_run_id[:8]}…)",
            metadata={
                "eval_run_id": eval_run_id,
                "metrics":     metrics,
            },
        )
        await self.add_edge(parent_artifact_node_id, node_id, EdgeRelationship.REGISTERED_AS)
        return node_id

    async def record_model_version(
        self,
        version: str,
        parent_eval_node_id: int,
    ) -> int:
        """
        Add a model version node from the MLflow Registry.

        Args:
            version: MLflow model version string (e.g. "3").
            parent_eval_node_id: DB id of the upstream evaluation run node.

        Returns:
            The DB node ID.
        """
        external_id = f"{self._model_name}/{version}"
        node_id = await self.add_node(
            node_type=LineageNodeType.MODEL_VERSION,
            external_id=external_id,
            display_label=f"Model Version — {self._model_name} v{version}",
            metadata={
                "model_name":    self._model_name,
                "model_version": version,
            },
        )
        await self.add_edge(parent_eval_node_id, node_id, EdgeRelationship.EVALUATED_BY)
        return node_id

    # Phase 4 stub methods (implemented in Phase 4) 
    async def record_oci_image(
        self,
        image_uri: str,
        digest: str,
        parent_version_node_id: int,
    ) -> int:
        """
        Phase 4 stub.  Records the OCI image node after Kaniko build completes.
        Called from DockerfileAgent (Phase 4).
        """
        node_id = await self.add_node(
            node_type=LineageNodeType.OCI_IMAGE,
            external_id=f"{image_uri}@{digest}",
            display_label=f"OCI Image — {image_uri.split('/')[-1]}",
            metadata={"image_uri": image_uri, "digest": digest},
        )
        await self.add_edge(parent_version_node_id, node_id, EdgeRelationship.PACKAGED_AS)
        return node_id


# Graph querier 
class LineageGraphQuerier:
    """
    Queries the lineage DAG for display in the UI (Phase 6) and the REST API.

    Provides:
        get_full_lineage()  — full node + edge lists for a given model version
        get_lineage_path()  — ordered node list from root dataset to latest node
        find_by_run_id()    — find which model version a given MLflow run belongs to
    """

    def __init__(self, pool: Any) -> None:
        self._pool = pool

    async def get_full_lineage(
        self,
        model_name: str,
        model_version: str,
    ) -> dict[str, Any]:
        """
        Return the full DAG (nodes + edges) for a model version.

        Returns:
            {
                "model_name":    str,
                "model_version": str,
                "nodes": [{"id":int, "node_type":str, "external_id":str,
                            "display_label":str, "metadata":dict, "created_at":str}, ...],
                "edges": [{"id":int, "parent_node_id":int, "child_node_id":int,
                            "relationship":str}, ...],
            }
        """
        async with self._pool.acquire() as conn:
            node_rows = await conn.fetch(
                """
                SELECT id, node_type, external_id, display_label, metadata, created_at
                FROM model_lineage_nodes
                WHERE model_name=$1 AND model_version=$2
                ORDER BY id
                """,
                model_name, model_version,
            )
            if not node_rows:
                return {
                    "model_name": model_name,
                    "model_version": model_version,
                    "nodes": [],
                    "edges": [],
                }

            node_ids = [r["id"] for r in node_rows]
            edge_rows = await conn.fetch(
                """
                SELECT id, parent_node_id, child_node_id, relationship
                FROM model_lineage_edges
                WHERE parent_node_id = ANY($1::bigint[])
                   OR child_node_id  = ANY($1::bigint[])
                ORDER BY id
                """,
                node_ids,
            )

        nodes = [
            {
                "id":            r["id"],
                "node_type":     r["node_type"],
                "external_id":   r["external_id"],
                "display_label": r["display_label"],
                "metadata":      dict(r["metadata"]) if r["metadata"] else {},
                "created_at":    r["created_at"].isoformat() if r["created_at"] else None,
            }
            for r in node_rows
        ]
        edges = [
            {
                "id":             r["id"],
                "parent_node_id": r["parent_node_id"],
                "child_node_id":  r["child_node_id"],
                "relationship":   r["relationship"],
            }
            for r in edge_rows
        ]
        return {
            "model_name":    model_name,
            "model_version": model_version,
            "nodes":         nodes,
            "edges":         edges,
        }

    async def get_lineage_path(
        self,
        model_name: str,
        model_version: str,
    ) -> list[dict[str, Any]]:
        """
        Return nodes in topological order from root dataset to the latest node.

        Uses BFS starting from nodes with no incoming edges (roots) and follows
        outgoing edges in insertion order.  Returns an empty list if no lineage
        has been recorded for this version.
        """
        graph = await self.get_full_lineage(model_name, model_version)
        if not graph["nodes"]:
            return []

        # Build adjacency and in-degree maps
        nodes_by_id = {n["id"]: n for n in graph["nodes"]}
        in_degree: dict[int, int] = defaultdict(int)
        adjacency: dict[int, list[int]] = defaultdict(list)

        for edge in graph["edges"]:
            adjacency[edge["parent_node_id"]].append(edge["child_node_id"])
            in_degree[edge["child_node_id"]] += 1

        # All node IDs
        all_ids = set(nodes_by_id.keys())

        # BFS topological sort (Kahn's algorithm)
        queue: deque[int] = deque(nid for nid in all_ids if in_degree[nid] == 0)
        ordered: list[dict[str, Any]] = []

        while queue:
            nid = queue.popleft()
            if nid in nodes_by_id:
                ordered.append(nodes_by_id[nid])
            for child_id in adjacency[nid]:
                in_degree[child_id] -= 1
                if in_degree[child_id] == 0:
                    queue.append(child_id)

        return ordered

    async def find_by_run_id(self, run_id: str) -> Optional[dict[str, str]]:
        """
        Find which (model_name, model_version) a given MLflow run_id belongs to.

        Searches both training_run and eval_run nodes.  Returns None if not found.
        """
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT model_name, model_version
                FROM model_lineage_nodes
                WHERE external_id=$1
                  AND node_type IN ('training_run', 'eval_run')
                LIMIT 1
                """,
                run_id,
            )
        if row is None:
            return None
        return {"model_name": row["model_name"], "model_version": row["model_version"]}

    async def get_versions_with_lineage(self, model_name: str) -> list[str]:
        """
        Return all model versions that have at least one lineage node.
        """
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT DISTINCT model_version
                FROM model_lineage_nodes
                WHERE model_name=$1
                ORDER BY model_version::int DESC
                """,
                model_name,
            )
        return [r["model_version"] for r in rows]