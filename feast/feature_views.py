"""
feast/feature_views.py
════════════════════════
Defines entities and feature views registered by the Data Agent.

Apply with:  feast apply
Materialize: feast materialize-incremental $(date -u +"%Y-%m-%dT%H:%M:%S")
"""
from datetime import timedelta
from pathlib import Path

import pandas as pd
from feast import (
    Entity,
    FeatureStore,
    FeatureView,
    Field,
    FileSource,
)
from feast.types import Float32, Int64, String

# ── Entity: each row is keyed by a record_id ─────────────────────────────────
record = Entity(name="record", join_keys=["record_id"])

# ── Offline source: Parquet files written by the Data Agent ──────────────────
data_source = FileSource(
    path=str(Path(__file__).parent / "data" / "features.parquet"),
    timestamp_field="event_timestamp",
)

# ── Feature view: numeric features used by all models ────────────────────────
numeric_features_fv = FeatureView(
    name="numeric_features",
    entities=[record],
    ttl=timedelta(days=1),
    schema=[
        Field(name="f0",  dtype=Float32),
        Field(name="f1",  dtype=Float32),
        Field(name="f2",  dtype=Float32),
        Field(name="f3",  dtype=Float32),
        Field(name="f4",  dtype=Float32),
        Field(name="f5",  dtype=Float32),
        Field(name="f6",  dtype=Float32),
        Field(name="f7",  dtype=Float32),
        Field(name="f8",  dtype=Float32),
        Field(name="f9",  dtype=Float32),
    ],
    source=data_source,
    tags={"team": "mlops", "domain": "generic"},
)

# ── Feature view: categorical / entity-level metadata ────────────────────────
entity_metadata_fv = FeatureView(
    name="entity_metadata",
    entities=[record],
    ttl=timedelta(days=7),
    schema=[
        Field(name="category",  dtype=String),
        Field(name="source",    dtype=String),
        Field(name="row_count", dtype=Int64),
    ],
    source=data_source,
)


def generate_sample_features(output_path: str = "feast/data/features.parquet") -> None:
    """
    Generate a sample feature dataset for local development.
    In production the Data Agent writes this file after ingestion.
    """
    import numpy as np
    from datetime import datetime, timezone

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)
    n = 1000
    df = pd.DataFrame(
        rng.standard_normal((n, 10)),
        columns=[f"f{i}" for i in range(10)],
    )
    df["record_id"]        = range(n)
    df["category"]         = rng.choice(["A", "B", "C"], size=n)
    df["source"]           = "synthetic"
    df["row_count"]        = n
    df["event_timestamp"]  = datetime.now(timezone.utc)
    df.to_parquet(output_path, index=False)
    print(f"Sample features written to {output_path}")


if __name__ == "__main__":
    generate_sample_features()
