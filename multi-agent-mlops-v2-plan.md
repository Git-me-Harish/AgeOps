# Enterprise V2 Plan — Multi-Agent MLOps Platform
## From Mock Prototype → Production Enterprise System

**Document Type**: Architecture Decision + Phased Delivery Plan
**Perspective**: MLOps Engineer + DevOps Engineer + System Design Architect
**Version**: 2.0 | Status: Planning

---

## What V1 Was (Honest Assessment)

V1 proved the architecture concept. It has the right skeleton — LangGraph agents,
MLflow integration, Kubernetes manifests, CI/CD pipeline. But the following things
are not production-grade:

| V1 Component | What's Wrong | V2 Fix |
|---|---|---|
| Data Agent | Mocks boto3, synthetic dataframes, no real connector | Real ingestion from S3/GCS/ADLS with schema enforcement |
| Training Agent | Falls back to `make_classification()` synthetic data | Real Kubernetes Jobs with actual framework runners (XGBoost, PyTorch, HF) |
| Evaluation Agent | `mlflow.evaluate()` on synthetic holdout | Custom LLM-as-judge + RAGAS-style eval framework on real holdout sets |
| Deployment Agent | Stubs Prometheus metrics, sleep() for canary wait | Real Prometheus query loop, real KServe GRPC health checks |
| RL Optimizer | Trains on synthetic historical runs | Trains on real MLflow trace data, validates recommendations before applying |
| Model "Registry" | Just MLflow stage transitions | Full model lineage, packaging as OCI images, signed provenance |
| Security Agent | Regex pattern matching only | OWASP-compliant input validation, Trivy in CI (not simulated) |
| UI | Static mock data everywhere | Live WebSocket feeds, real API calls, no hardcoded arrays |
| MCP Servers | Thin FastAPI wrappers, no tool schemas | Fully typed MCP tools, version negotiation, capability discovery |
| Monitoring | Prometheus gauges with stub values | Real Evidently drift reports, live inference traffic analysis |

---

## The Question You Asked — Model as Docker Image

This is the right instinct and it is **exactly how production MLOps works**.
Here is the complete answer:

### The Model-as-OCI-Image Pattern

When a model completes training and evaluation, instead of just registering
it in the MLflow Model Registry as a directory of files, you package it as
a self-contained OCI container image. The image contains:

```
┌─────────────────────────────────────────────────────┐
│  mlops-model:v1.2.3 (OCI Image)                    │
│                                                     │
│  /model/                                            │
│    artifacts/          ← weights, encoders, scalers │
│    MLmodel             ← MLflow model spec          │
│    python_env.yaml     ← exact Python env           │
│    requirements.txt    ← pinned deps                │
│                                                     │
│  /serving/                                          │
│    server.py           ← FastAPI inference server   │
│    health.py           ← liveness + readiness       │
│    metrics.py          ← Prometheus exporter        │
│                                                     │
│  LABEL org.opencontainers.image.revision=<git-sha>  │
│  LABEL mlops.model.accuracy=0.923                   │
│  LABEL mlops.model.framework=xgboost                │
│  LABEL mlops.model.trained-at=2025-08-09            │
│  LABEL mlops.model.dataset-hash=sha256:abc123       │
│  LABEL mlops.eval.f1=0.901                          │
│                                                     │
│  ENTRYPOINT ["python", "serving/server.py"]         │
└─────────────────────────────────────────────────────┘
```

An AI engineer pulls this image and gets:
- A working inference server at port 8080 on `docker run`
- A `/predict` endpoint with documented input/output schema
- A `/health` endpoint for Kubernetes probes
- `/metrics` for Prometheus
- All model dependencies already baked in — no pip install surprises
- Cryptographic provenance: they can verify what dataset trained it,
  what accuracy it achieved, which git commit built it

This is better than raw MLflow artifacts because:
1. Zero environment setup — works identically on any machine
2. The image IS the deployment unit — same artifact in dev and production
3. Immutable + versioned — `docker pull mlops-model:v1.2.3` is deterministic
4. Security scanned before release (Trivy in CI blocks CRITICAL CVEs)
5. KServe can serve it directly without knowing anything about the framework

This becomes **Phase 4** of the V2 plan below.

---

## V2 Architecture — Enterprise Production

```
┌──────────────────────────────────────────────────────────────────────────┐
│                         INGESTION PLANE                                  │
│   S3 / GCS / ADLS / Kafka / REST / Database CDC                         │
│   → Schema Registry (Avro/Protobuf) → Feature Validation → Feast        │
└─────────────────────────────┬────────────────────────────────────────────┘
                              │ validated, versioned dataset
┌─────────────────────────────▼────────────────────────────────────────────┐
│                        AGENT ORCHESTRATION PLANE                         │
│                                                                          │
│  LangGraph Supervisor                                                    │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  │
│  │ Planner  │  │  Data    │  │ Training │  │  Eval    │  │  Deploy  │  │
│  │  Agent   │→ │  Agent   │→ │  Agent   │→ │  Agent   │→ │  Agent   │  │
│  │ (ReAct)  │  │(Tool node│  │(K8s Job) │  │(LLM judge│  │(KServe)  │  │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘  └──────────┘  │
│        ↑                                                       │         │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐       │         │
│  │    RL    │  │ Monitor  │  │Governance│  │ Security │       │         │
│  │  Agent   │  │  Agent   │  │  Agent   │  │  Agent   │       │         │
│  │  (PPO)   │  │(Evidently│  │(OPA+SBOM)│  │(Trivy)   │       │         │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘       │         │
└─────────────────────────────────────────────────────────────────┼────────┘
                                                                   │
┌─────────────────────────────────────────────────────────────────▼────────┐
│                        MODEL PACKAGING PLANE                             │
│                                                                          │
│   Trained weights + MLmodel spec                                         │
│   → Dockerfile Generator Agent                                           │
│   → OCI Image Build (Kaniko in-cluster)                                  │
│   → Trivy Scan → SBOM Generation → Cosign Sign                          │
│   → Push to Registry (Docker Hub / GHCR)                                │
│   → MLflow Registry: model_uri = oci://registry/mlops-model:v1.2.3      │
│   → Image LABELS = accuracy, f1, dataset-hash, git-sha                  │
└─────────────────────────────┬────────────────────────────────────────────┘
                              │ signed, scanned OCI image
┌─────────────────────────────▼────────────────────────────────────────────┐
│                        SERVING PLANE                                     │
│                                                                          │
│   KServe InferenceService → pulls OCI image                             │
│   → Canary: 10% → 50% → 100% (real Prometheus-gated)                   │
│   → Shadow mode (A/B traffic mirroring for eval)                        │
│   → Auto-rollback on error-rate breach                                  │
│   → gRPC + REST dual-protocol serving                                   │
└─────────────────────────────┬────────────────────────────────────────────┘
                              │ live inference
┌─────────────────────────────▼────────────────────────────────────────────┐
│                       OBSERVABILITY PLANE                                │
│                                                                          │
│   OpenTelemetry Collector                                                │
│   → Traces: Tempo (self-hosted)                                         │
│   → Metrics: Prometheus → Grafana                                       │
│   → Logs: Loki ← Promtail                                               │
│   → Drift reports: Evidently → Grafana dashboard                        │
│   → Agent traces: MLflow Tracing (native @mlflow.trace)                 │
│   → Cost tracking: token usage per agent per workflow                   │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## The 6 Phases — Enterprise V2 Delivery Plan

---

### PHASE 1 — Real Data Infrastructure & Feature Platform

**Goal**: Replace all synthetic data and mock connectors with real, production-grade
data ingestion, validation, and feature management.

**Duration estimate**: Build sprint — once started, do not proceed to Phase 2 until
every item here is real (no mocks).

#### 1.1 Multi-Source Data Connectors (Data Agent)

Replace the `boto3.client` stub with a proper connector framework:

```
Supported sources in V2:
  ├── Object Storage:  S3, Cloudflare R2, GCS, Azure ADLS (all S3-compatible)
  ├── Databases:       PostgreSQL CDC (Debezium), MySQL, SQLite export
  ├── Streaming:       Apache Kafka topics (real-time feature ingestion)
  ├── REST APIs:       Generic HTTP connector with auth (Bearer, API Key, OAuth2)
  └── File Uploads:    CSV, Parquet, JSON, Avro via signed upload URL
```

Each connector implements a strict `DataConnector` interface:
- `connect()` → validates credentials, tests connectivity
- `read(schema: DataSchema) → DataFrame` → typed, schema-enforced read
- `sample(n: int) → DataFrame` → head sample for fast validation
- `stream() → AsyncIterator[Batch]` → for Kafka / streaming sources

**Schema Registry**: Every dataset gets a versioned Avro or JSON Schema registered
in the system before ingestion proceeds. The Data Agent rejects datasets that do
not conform to their registered schema. Schema evolution is tracked.

#### 1.2 Real Data Validation (Great Expectations → Soda Core)

V1 used Great Expectations with a `try/except` fallback to null-ratio calculation.
V2 runs real validation suites with:
- Auto-generated expectations from a profiling run on the first dataset version
- Configurable threshold overrides per dataset
- Validation results stored in Neon as structured records (not just MLflow tags)
- Blocking: validation failure halts the pipeline and creates a GitHub Issue
  automatically via the Governance Agent

#### 1.3 Real Drift Detection (Evidently)

V1 returned `0.04` as a stub drift score. V2:
- Stores reference dataset statistics (mean, std, histogram) per feature at
  training time in Neon
- Runs Evidently DataDriftPreset against real current vs reference data
- Generates HTML + JSON drift reports stored in R2
- Publishes per-feature drift scores as Prometheus metrics
- Sends structured alert to Monitoring Agent if any feature exceeds threshold

#### 1.4 Feast Feature Store — Real Materialization

V1 called `store.materialize()` and caught the exception silently. V2:
- Defines real feature views with proper entity keys and TTLs
- Online store: Redis (self-hosted on K3s node)
- Offline store: Parquet files in R2
- Feature retrieval at training time uses `store.get_historical_features()`
- Feature retrieval at serving time uses `store.get_online_features()`
- Training Agent reads from Feast, not directly from raw files

#### 1.5 Data Lineage Tracking

Every dataset that enters the system gets:
- A content hash (SHA-256 of the raw bytes)
- A lineage record in Neon: `{source_uri, content_hash, row_count, schema_version,
  ingested_at, workflow_id}`
- MLflow dataset tracking via `mlflow.log_input()`
- Governance Agent signs the lineage record for tamper evidence

**Completion criteria for Phase 1**:
- [ ] At least 2 real data connectors working (R2 + PostgreSQL)
- [ ] Real Great Expectations suite auto-generated and running
- [ ] Real Evidently drift report generated and stored
- [ ] Feast materialization writing to Redis and R2 Parquet
- [ ] Data lineage record created in Neon for every ingestion
- [ ] Zero synthetic dataframes (`make_classification`) anywhere in production code

---

### PHASE 2 — Real Agent Intelligence (LLM-Backed ReAct Loops)

**Goal**: Replace every agent's stub logic with a genuine LLM-backed reasoning loop.
Agents must use real tool calls, not hardcoded decision trees.

**This is the core of what makes the system "agentic" vs "scripted".**

#### 2.1 LLM Gateway (Single Point of LLM Access)

All agents go through one gateway — never call LLM APIs directly:

```
LLM Gateway responsibilities:
  ├── Model routing: primary (gpt-4o-mini / claude-haiku) + fallback (local Ollama)
  ├── Prompt template versioning: stored in Neon, retrieved by version ID
  ├── Token budget enforcement: per-agent, per-workflow limits
  ├── Semantic cache: Redis-backed, cosine-similarity hit detection
  ├── Retry with exponential backoff + jitter
  ├── Structured output enforcement: instructor library + Pydantic schemas
  ├── Cost tracking: tokens in/out per agent per run → MLflow metric
  └── Fallback to Ollama (llama3.1:8b) when API quota exhausted
```

Local Ollama on Oracle Cloud ARM VM gives a $0 fallback model for
non-critical reasoning steps (planning, classification, summarisation).
Critical steps (final deployment decision, security assessment) always use
the cloud model.

#### 2.2 Real Planner Agent (Plan-and-Execute Pattern)

V1 Planner returned a hardcoded dict. V2 uses Plan-and-Execute:

```
Input: {dataset_uri, target_metric, deadline, resource_constraints}

LLM reasoning loop:
  1. Query MLflow for last 20 similar workflows (same dataset family)
  2. Query RL agent for current hyperparameter recommendations
  3. Reason about: which frameworks to try, whether to run parallel experiments,
     whether the dataset needs augmentation, estimated training time
  4. Output: ExecutionPlan (typed Pydantic model):
     - ordered list of PlanSteps with estimated durations
     - parallel_experiments: list of hyperparameter grids to try concurrently
     - data_augmentation_required: bool with reasoning
     - estimated_accuracy: float with confidence interval
     - human_approval_required: bool (True if changes are high-impact)

The plan is stored in Neon and shown in the UI for human review
before execution starts (HITL gate on every new plan).
```

#### 2.3 Real Training Agent (Multi-Framework Kubernetes Jobs)

V1 launched a single hardcoded GBM Job. V2 supports:

```
Framework runners (each is a separate container image):
  ├── mlops-runner-sklearn:latest    → scikit-learn pipelines
  ├── mlops-runner-xgboost:latest    → XGBoost with GPU support
  ├── mlops-runner-pytorch:latest    → PyTorch with Distributed Data Parallel
  ├── mlops-runner-huggingface:latest → HuggingFace Trainer API fine-tuning
  └── mlops-runner-custom:latest     → User-provided training script

Training Agent workflow:
  1. Select framework runner based on Planner recommendation
  2. Generate Kubernetes Job manifest (not hardcoded — generated from template)
  3. Inject: dataset URI, Feast feature view, MLflow run ID, hyperparams
  4. Launch Job, stream logs back via Kubernetes log API (real-time in UI)
  5. Poll Job status — handle: RUNNING, SUCCEEDED, FAILED, OOMKilled
  6. On OOMKilled: reduce batch_size, retry (RL agent records this)
  7. On SUCCEEDED: retrieve metrics from MLflow run (not from Job stdout)
  8. Support parallel experiments: launch N Jobs concurrently, pick best
```

MLflow autologging is enabled per framework. Training containers also log:
- System metrics (CPU, RAM, GPU utilisation) every 30 seconds
- Per-epoch loss curves
- Feature importance scores
- Training data hash (for lineage)

#### 2.4 Real Evaluation Agent (LLM-as-Judge + Golden Dataset)

V1 used `mlflow.evaluate()` on synthetic holdout. V2:

```
Evaluation pipeline:
  1. Load holdout dataset from Feast (never from the training dataset split)
  2. Run mlflow.evaluate() with default metrics (accuracy, F1, AUC, confusion matrix)
  3. Run custom evaluators:
     a. Calibration check: are predicted probabilities well-calibrated?
     b. Slice evaluation: run metrics on known-important subgroups
        (if metadata available: by category, by time period, by data source)
     c. Regression testing: compare against current production model
        → block promotion if new model is worse on ANY slice
  4. LLM-as-Judge evaluation:
     - Sample 50 predictions where model is most uncertain (near decision boundary)
     - Send to LLM with original features and ask: "Given these features, is this
       prediction reasonable? What evidence supports/contradicts it?"
     - LLM judge produces structured critique: {verdict, confidence, reasoning}
     - Low-confidence predictions are flagged for human review
  5. Bias check (real, not stub):
     - If protected attribute columns present: compute demographic parity,
       equalized odds, individual fairness metrics
     - Block promotion if any fairness metric fails
  6. Threshold gate: all metrics must pass configurable thresholds
  7. Comparison report stored as HTML artifact in R2, linked from MLflow
```

#### 2.5 Real Governance Agent (OPA Policy Engine Integration)

V1 `check_policy()` returned `True` when OPA was unreachable (fail-open in prod).
V2:
- OPA runs as a sidecar in the Orchestrator pod (not a separate network call)
- Policies loaded from Rego files at startup, not fetched on every call
- `check_policy()` is synchronous and never network-dependent in the hot path
- Production: fail-closed (returns False immediately if OPA sidecar crashes)
- Every policy decision is logged with: policy name, input hash, decision,
  evaluation time — stored in `agent_audit_trail` table
- Compliance reports generated weekly and stored as PDF artifacts in R2

#### 2.6 Real Security Agent

V1 did regex injection detection. V2 adds:

```
Security Agent tools (real, not stubs):
  ├── Trivy integration: scan every Docker image before deployment
  │     (not CI simulation — actual Trivy binary in a Job)
  ├── Semgrep SAST: scan training scripts before execution
  ├── Supply chain: verify pip dependency hashes against known-good lockfile
  ├── Secret scanning: detect AWS keys / tokens in datasets and model artifacts
  ├── Input sanitization: enforce input schema + type constraints at API boundary
  ├── Output filtering: PII detection on inference outputs before returning to caller
  └── Rate limiting: per-client request budgets at the API gateway level
```

**Completion criteria for Phase 2**:
- [ ] LLM Gateway running with real primary model + Ollama fallback
- [ ] Planner produces real ExecutionPlan using LLM reasoning
- [ ] Training Agent launches real K8s Jobs with real framework runner images
- [ ] Evaluation Agent runs real mlflow.evaluate() on real holdout data
- [ ] Governance Agent OPA runs as sidecar, fail-closed in production
- [ ] Security Agent runs real Trivy scan via Kubernetes Job (not simulation)
- [ ] Token cost tracked per agent per workflow in MLflow metrics
- [ ] Zero hardcoded decisions in any agent (all reasoning via LLM tool calls)

---

### PHASE 3 — Model Registry as Source of Truth

**Goal**: The MLflow Model Registry becomes the single source of truth for every
model artifact, its lineage, its evaluation results, its approvals, and its
deployment history. No model reaches production without a complete registry record.

#### 3.1 Structured Model Metadata Schema

Every model registered in the MLflow Registry must have these tags:

```
Required tags (pipeline enforces — missing tag = registration rejected):
  mlops.dataset.uri           ← R2 path of training dataset
  mlops.dataset.hash          ← SHA-256 of training data
  mlops.dataset.row_count     ← numeric
  mlops.framework             ← xgboost | pytorch | sklearn | huggingface | custom
  mlops.framework.version     ← e.g., 2.1.0
  mlops.python.version        ← e.g., 3.11.6
  mlops.eval.accuracy         ← numeric
  mlops.eval.f1               ← numeric
  mlops.eval.auc              ← numeric
  mlops.eval.holdout_hash     ← SHA-256 of holdout dataset
  mlops.eval.bias_passed      ← true | false
  mlops.security.trivy_scan   ← passed | failed
  mlops.security.cve_critical ← numeric
  mlops.git.commit            ← full git SHA
  mlops.git.repo              ← repo URL
  mlops.approved_by           ← reviewer GitHub username
  mlops.approved_at           ← ISO 8601 timestamp
  mlops.governance.audit_id   ← SHA-256 audit trail signature
```

These tags are what get embedded as OCI image LABELS in Phase 4.

#### 3.2 Model Lineage Graph

The system builds a directed acyclic graph:

```
Raw Dataset (version hash)
    │
Feature Engineering (Feast view version)
    │
Training Run (MLflow run ID)
    │
Model Artifact (MLflow URI)
    │
Evaluation Run (MLflow run ID)
    │
Model Version (Registry: name + version)
    │
OCI Image (registry URL + digest)
    │
InferenceService (KServe CRD name)
    │
Live Traffic (Prometheus metric labels)
```

This full lineage is queryable via the UI and the REST API. An AI engineer
can take any live prediction and trace it back to the exact row of training
data that most influenced it (if SHAP or similar explanability is enabled).

#### 3.3 Promotion Workflow (Enforced State Machine)

Model lifecycle states with enforced transitions:

```
None → Staging → Production → Archived
              ↘ Rejected

Transition rules (enforced by Governance Agent, not optional):
  None → Staging:      Training Agent registers automatically
                       Requires: all mandatory tags present
  Staging → Production: Requires:
                         1. Evaluation Agent approval (threshold gate passed)
                         2. Human approval (HITL gate — always required)
                         3. Security Agent clearance (Trivy scan passed)
                         4. Governance Agent sign-off (OPA policies passed)
                         5. OCI image built and signed (Phase 4)
  Production → Archived: Automatic after new version goes Production
                          Or manual via UI with reason required
  Staging → Rejected:  Automatic on evaluation failure
                       Cannot be overridden without re-evaluation
```

The UI shows this state machine visually. Every transition is a timestamped
event in `agent_audit_trail`.

#### 3.4 Model Comparison View

The UI's Model Registry page (currently a simple card list) becomes a full
comparison view:

```
Side-by-side comparison of any 2-4 model versions:
  ├── Metric comparison: accuracy, F1, AUC, latency P99, memory footprint
  ├── Training data comparison: dataset hash, row count, feature counts
  ├── Slice performance: how does each model perform on each data segment?
  ├── Prediction distribution: probability histogram overlay
  ├── SHAP feature importance: per-model feature ranking
  └── Decision: "Use v3 for production" button → triggers approval workflow
```

**Completion criteria for Phase 3**:
- [ ] All 18 mandatory tags enforced at registration (missing tag = rejection)
- [ ] Full lineage graph queryable from UI for any production model
- [ ] Promotion state machine enforced — no model bypasses evaluation+security
- [ ] Model comparison view shows real metrics for any 2 selected versions
- [ ] Every promotion event stored in audit trail with signatures
- [ ] Historical promotion timeline visible per model in the UI

---

### PHASE 4 — Model Packaging as OCI Images

**Goal**: Every production-approved model becomes a signed, scanned,
self-contained OCI container image. This is the core innovation over
standard MLflow deployments.

#### 4.1 Dockerfile Generator Agent

A new specialist agent: `DockerfileAgent`. Triggered after Evaluation Agent approval.

The agent:
1. Reads the model's framework tag from the registry
2. Selects the appropriate base image:
   ```
   sklearn/xgboost  → python:3.11-slim (12 MB base)
   pytorch          → pytorch/pytorch:2.5.1-cuda12.1-cudnn8-runtime
   huggingface      → huggingface/transformers-pytorch-gpu:latest
   custom           → user-specified base from model tags
   ```
3. Generates a Dockerfile using the MLflow model's `python_env.yaml` for
   exact dependency pinning
4. Injects a standardised inference server:
   ```python
   # Embedded in every model image
   # Provides: POST /predict, GET /health, GET /metrics, GET /info
   ```
5. Adds all mandatory OCI labels (from Phase 3 tags)
6. Submits the Dockerfile to Kaniko (in-cluster build — no Docker daemon needed)

The generated Dockerfile is stored as an artifact in the MLflow run so it
can be reproduced exactly.

#### 4.2 In-Cluster Build with Kaniko

Kaniko runs as a Kubernetes Job inside the cluster. No Docker daemon, no
privileged containers. Build process:

```
DockerfileAgent creates Kaniko Job → 
  Kaniko pulls base image → 
  Copies MLflow artifacts from R2 →
  Installs pinned dependencies →
  Builds image →
  Pushes to Docker Hub / GHCR →
  Reports digest back to DockerfileAgent
```

Build logs are streamed to the Training Agent's log view in the UI.

#### 4.3 Supply Chain Security (Cosign + SBOM)

After build:
1. **Trivy scan**: scans the built image for CVEs
   - CRITICAL CVE count > 0 → image rejected, Deployment Agent blocked
   - HIGH CVE count > threshold → human approval required before deployment
2. **SBOM generation**: Syft generates a Software Bill of Materials in SPDX format
   - Stored in R2 as `{model_name}-{version}-sbom.spdx.json`
   - Attached to the OCI image as an attestation
3. **Cosign signing**: image is cryptographically signed with a keyless
   Sigstore signature (OIDC-based, no key management needed)
   - Verification command embedded in model's README
4. **Image digest recorded**: the `sha256:` digest (not the tag) is stored
   in MLflow Registry as the canonical reference. Tags are mutable; digests are not.

#### 4.4 Standardised Inference Server (Embedded in Every Image)

Every model image includes the same inference server implementation:

```
Endpoints:
  POST /predict
    Input:  {"instances": [...]} or {"inputs": {...}} (both formats supported)
    Output: {"predictions": [...], "model_version": "v1.2.3",
             "inference_time_ms": 12.4, "request_id": "uuid"}
    
  POST /predict/batch
    Input:  {"instances": [...], "batch_size": 32}
    Streams results as JSON lines

  GET  /health/live   → {"status": "ok"}
  GET  /health/ready  → {"status": "ready", "model_loaded": true, "warmup_done": true}
  GET  /metrics       → Prometheus text format
  GET  /info          → full model metadata (all OCI labels as JSON)
  GET  /schema        → input/output JSON Schema for this model
  GET  /docs          → auto-generated OpenAPI docs
```

The inference server handles:
- Model warmup (first prediction pre-run to avoid cold-start latency spike)
- Input validation against the registered schema
- Output PII filtering (Security Agent policy applied at serving time)
- Per-request trace context propagation (OpenTelemetry)
- Graceful shutdown on SIGTERM (drain in-flight requests)

#### 4.5 How an AI Engineer Uses the Model Image

This is the workflow for someone consuming a model built by this system:

```bash
# Pull the model
docker pull your-dockerhub-user/mlops-model-fraud-detector:v2.1.0

# Verify signature (supply chain security)
cosign verify your-dockerhub-user/mlops-model-fraud-detector:v2.1.0

# Check what's in it
docker run --rm your-dockerhub-user/mlops-model-fraud-detector:v2.1.0 cat /info
# Returns: accuracy, F1, training dataset hash, git commit, approved_by, etc.

# Run it
docker run -p 8080:8080 your-dockerhub-user/mlops-model-fraud-detector:v2.1.0

# Predict
curl -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d '{"instances": [{"amount": 1500.0, "merchant_category": "electronics", ...}]}'

# In production: Kubernetes deployment
# (KServe InferenceService uses this exact image — same artifact)
kubectl apply -f - <<EOF
apiVersion: serving.kserve.io/v1beta1
kind: InferenceService
metadata:
  name: fraud-detector
spec:
  predictor:
    containers:
      - image: your-dockerhub-user/mlops-model-fraud-detector:v2.1.0@sha256:abc123
        ports:
          - containerPort: 8080
EOF
```

This is superior to the current approach because the image IS the model.
No MLflow server needed at inference time. No Python environment setup.
The exact same artifact used in testing is deployed to production.

**Completion criteria for Phase 4**:
- [ ] DockerfileAgent generates real Dockerfiles for sklearn, XGBoost, PyTorch
- [ ] Kaniko builds successfully in-cluster (no privileged containers)
- [ ] Trivy scan blocks images with CRITICAL CVEs before push
- [ ] SBOM generated and stored in R2 for every image
- [ ] Cosign signature verifiable with public key
- [ ] Inference server endpoints all functional with real model loaded
- [ ] `docker run` on a built image serves predictions in < 5 seconds
- [ ] KServe InferenceService serves the same image used in docker run

---

### PHASE 5 — Production Observability & Intelligent Monitoring

**Goal**: The monitoring system proactively detects model degradation, data drift,
and system anomalies. It closes the loop back to training automatically.

#### 5.1 Three-Layer Observability Stack

```
Layer 1 — Infrastructure (Kubernetes, pods, nodes)
  Prometheus Node Exporter + kube-state-metrics
  Grafana dashboard: pod restarts, memory pressure, CPU throttling

Layer 2 — Application (agents, API, model serving)
  OpenTelemetry SDK in every agent and MCP server
  Traces → Grafana Tempo (self-hosted)
  Metrics → Prometheus
  Logs → Loki
  Grafana dashboard: request rate, error rate, latency P50/P95/P99

Layer 3 — ML Model Health (the novel layer)
  Evidently running on real inference traffic samples
  Prometheus custom metrics: drift score, prediction distribution shift,
    feature value ranges vs training baseline
  Grafana dashboard: per-feature drift, prediction confidence distribution,
    business metric correlation (if available)
```

#### 5.2 Real Monitoring Agent Loop

V1 monitoring returned stub values. V2 runs a real loop:

```
Every 60 seconds (configurable):
  1. Sample last N inference requests from the serving layer
     (KServe logs → Loki → Monitoring Agent query)
  2. Run Evidently DataDriftPreset on sampled inputs vs reference stats
  3. Run Evidently ClassificationPreset on sampled outputs (if ground truth available)
  4. Publish per-feature drift scores to Prometheus
  5. Check against thresholds:
     - Drift score > 0.15 on any feature → WARNING alert
     - Drift score > 0.30 on any feature → CRITICAL alert + trigger retraining
     - Accuracy drop > 5% vs baseline → CRITICAL + trigger retraining
     - P99 latency > 500ms → page the Deployment Agent
  6. Store Evidently report as HTML artifact in R2
  7. Write monitoring event to Neon for historical trend analysis
```

Ground truth labelling pipeline: if the system has access to delayed ground
truth labels (common in fraud detection, churn prediction), a separate
Labelling Agent collects them and computes real accuracy metrics on
production predictions.

#### 5.3 Retraining Loop (Closed-Loop MLOps)

This is what separates V2 from a static deployment:

```
Monitoring Agent detects drift →
  → Creates WorkflowTrigger record in Neon
  → Notifies Orchestrator via internal event queue
  → Orchestrator creates new workflow: source=monitoring, trigger=drift_detected
  → New workflow runs: Data Agent (with updated dataset) →
    Training Agent → Evaluation Agent → human approval → Deployment Agent
  → If new model passes evaluation AND beats current production model:
    → Canary rollout begins automatically
  → If new model does NOT beat current:
    → Monitoring Agent logs degradation, alerts sent, no deployment
    → RL Agent records this as negative reward
```

The loop is fully automatic except for the human approval gate before
any production deployment.

#### 5.4 RL Agent — Real Learning Loop

V1 RL agent trained on synthetic historical runs. V2:

```
Training data: real MLflow runs from the last 90 days
  Features: dataset size, feature count, framework, hyperparams,
            training duration, evaluation metrics, deployment outcome,
            drift score at 7/14/30 days post-deployment
  
Reward function (learned, not hardcoded):
  R = w1 * accuracy_gain
    + w2 * (1 / training_time)
    + w3 * (1 / drift_at_30_days)
    - w4 * retraining_frequency
    - w5 * cost_per_prediction

The weights w1-w5 are themselves tuned by a meta-learner that observes
which reward weightings led to the best 30-day production outcomes.

RL recommendations consumed by:
  - Planner Agent: framework selection, experiment parallelism
  - Training Agent: initial hyperparameter ranges, batch size
  - Deployment Agent: canary step sizes, rollback thresholds
  - Monitoring Agent: drift check frequency, alert thresholds

RL model retrained weekly (not daily — needs sufficient new data).
Recommendations shown in UI as "Optimizer Suggestions" with confidence scores.
Human can accept, reject, or modify before they take effect.
```

**Completion criteria for Phase 5**:
- [ ] All three observability layers instrumented and visible in Grafana
- [ ] Monitoring Agent runs real Evidently reports on real inference traffic
- [ ] Drift detection triggers automated retraining workflow end-to-end
- [ ] RL agent trains on real MLflow run data (not synthetic)
- [ ] RL recommendations shown in UI with accept/reject controls
- [ ] Complete workflow trace visible in Grafana Tempo (agent → tool → result)
- [ ] Zero stub metrics anywhere in the monitoring pipeline

---

### PHASE 6 — Enterprise UI & Developer Experience

**Goal**: Replace the Mantine/mock-data UI with a purpose-built, production-quality
interface that is genuinely useful to MLOps engineers, data scientists, and
team leads.

#### 6.1 UI Architecture Change

V1 used Mantine with static mock arrays. V2:

```
Tech stack:
  Framework:    Next.js 14 (App Router) — SSR for fast initial load
  UI system:    Custom design tokens + Radix UI primitives (not a component library)
  State:        React Query v5 (server state) + Zustand (local UI state)
  Real-time:    WebSocket connection to API (workflow status, agent logs, alerts)
  Charts:       Observable Plot (D3-based, more expressive than Recharts)
  Workflow canvas: React Flow (kept — it's the right choice)
  Auth:         NextAuth.js with GitHub OAuth
  API calls:    Type-safe with Zod schemas shared between frontend and FastAPI
```

#### 6.2 The 8 Pages (V2, Fully Real Data)

**Page 1 — Command Center (replaces Dashboard)**
- Live workflow status stream (WebSocket)
- Real agent health from Kubernetes pod status API
- Real alerts from Prometheus AlertManager webhook
- Real metric charts from Prometheus query API (not hardcoded Recharts data)
- "Start Pipeline" opens a multi-step wizard (not a single text input)

**Page 2 — Pipeline Builder (replaces Workflow Designer)**
- React Flow canvas (kept, enhanced)
- Drag-and-drop agent nodes with real capability descriptions from A2A registry
- Edge types: sequential, parallel, conditional, fallback
- Export as: JSON (for API), YAML (for GitOps), Python (for direct execution)
- "Validate" button: checks the graph for cycles, disconnected nodes,
  missing required agents
- "Dry run" button: Planner Agent estimates duration and cost before committing

**Page 3 — Experiment Lab (replaces Experiments)**
- Real data from MLflow Tracking API (not mocked)
- Parallel coordinates chart for hyperparameter exploration
- Metric comparison: select N runs, compare any metric side-by-side
- Artifact browser: list and preview artifacts stored in R2
- Full agent trace view: expandable span tree showing every LLM call,
  tool invocation, and decision made during the workflow
- Run diff: what changed between two runs?

**Page 4 — Model Registry (V2)**
- Lineage graph: visualise the full DAG from dataset → model → deployment
- Model comparison table: real metrics from MLflow, not hardcoded
- Promotion workflow: the state machine shown visually with current state
- "Pull command" for every image: `docker pull registry/model:version`
- SBOM viewer: list of all packages in the model image
- Signature status: verified / unverified with Cosign

**Page 5 — Live Serving Dashboard**
- Per-model: request rate, error rate, latency P50/P95/P99 (real Prometheus)
- Canary progress indicator: current traffic split with real metrics
- Rollback button: available at any time, requires reason input
- Shadow mode toggle: route 10% of traffic to candidate model without
  affecting production responses (for shadow evaluation)
- Per-endpoint breakdown: which clients are calling which models

**Page 6 — Drift & Health Center (replaces and expands Monitoring)**
- Per-feature drift timeline (real Evidently data from Neon)
- Prediction distribution shift: probability histogram evolution over time
- Business metric correlation: if configured, show how model accuracy
  correlates with business KPIs
- "Trigger retraining" button: manual override independent of thresholds
- Historical drift reports: browse and open stored Evidently HTML reports

**Page 7 — Agent Intelligence (replaces Agents)**
- Real agent status from Kubernetes pod API
- Per-agent: last 10 tasks, success rate, average duration, token cost
- RL recommendations panel: current suggestions with confidence scores,
  accept/reject controls
- Agent audit log: every decision with full OPA evaluation context
- "Configure agent" panel: real config that writes to Kubernetes ConfigMap

**Page 8 — Security & Compliance (kept, made real)**
- Real Trivy results from the most recent scan Job
- SBOM browser: search packages across all deployed model images
- OPA policy editor: edit Rego policies with syntax highlighting,
  validate before saving
- Compliance report generator: produce PDF covering GDPR, SOC2 controls
- Secret scanner: trigger a scan of all R2 buckets for credential leakage

#### 6.3 Real-Time Updates (WebSocket Architecture)

```
Backend publishes events to Redis Pub/Sub:
  workflow.{id}.status_changed
  workflow.{id}.agent.{agent_id}.log_line
  alert.{severity}.{source}
  model.{id}.deployment_status_changed

Frontend subscribes on page load:
  WebSocket → /ws/events?workflows=active&alerts=all
  React Query invalidates relevant queries on each event
  Toast notifications for alerts
  In-place status updates for running workflows
```

**Completion criteria for Phase 6**:
- [ ] Zero hardcoded mock arrays anywhere in the UI
- [ ] All charts pull from real Prometheus / MLflow / Neon APIs
- [ ] WebSocket real-time updates working for workflow status
- [ ] Model lineage graph rendered from real data
- [ ] Docker pull command generated from real image digest
- [ ] SBOM viewer showing real Syft output
- [ ] OPA policy editor saves to real ConfigMap
- [ ] All 8 pages functional with real backend data

---

## Cross-Cutting Architecture Decisions (ADRs)

### ADR-001: Model Images as the Deployment Unit

**Status**: Accepted

**Context**: V1 stored models as MLflow artifact directories and had KServe
load them via the MLflow model format plugin. This requires MLflow to be
running at inference time and ties serving to the MLflow ecosystem.

**Decision**: Package every production model as a signed OCI container image.
MLflow stores the image digest (not just the model path) as the canonical
model reference. KServe serves the image directly.

**Consequences**:
- Positive: No MLflow dependency at inference time. Image is self-contained.
  AI engineers can use models without any MLflow knowledge.
- Positive: Supply chain security (Cosign, SBOM, Trivy) is natural for images.
- Negative: Build time added to pipeline (2-5 minutes for Kaniko build).
- Negative: Image sizes are larger than raw model artifacts.
- Accepted trade-off: build time is worth the portability and security gains.

### ADR-002: OPA as Sidecar (Not Network Call)

**Status**: Accepted

**Context**: V1 called OPA over the network and failed open when unreachable.
This is a security vulnerability in production.

**Decision**: OPA runs as a sidecar container in the Orchestrator pod. Policy
evaluation is an in-process call. Fail-closed in production (deny if OPA
sidecar is unhealthy).

**Consequences**:
- Positive: Policy evaluation latency < 1ms (no network hop).
- Positive: Fail-closed is enforceable without distributed system coordination.
- Negative: OPA sidecar crash takes down the Orchestrator pod — this is intentional.
- Negative: Policy updates require pod restart — mitigated by ConfigMap hot-reload.

### ADR-003: Ollama as LLM Fallback

**Status**: Accepted

**Context**: Budget constraint of ₹300/month limits OpenAI/Anthropic API usage.
Agents must work even when API quota is exhausted.

**Decision**: Ollama running on the Oracle Cloud ARM VM provides `llama3.1:8b`
as a $0 fallback. The LLM Gateway routes to Ollama when the primary API is
unavailable or the token budget is exceeded.

**Consequences**:
- Positive: System functions at $0 API cost for non-critical operations.
- Negative: Ollama on ARM CPU is slow (~5 tokens/second for 8B model).
  Acceptable for planning/governance but not for latency-sensitive steps.
- Mitigation: Only planning, logging summarisation, and non-critical
  governance steps use Ollama fallback. Security and deployment decisions
  always use the cloud model.

### ADR-004: Neon PostgreSQL + pgvector for All Persistent State

**Status**: Accepted

**Context**: V1 used Neon for relational data. V2 needs vector similarity
search for semantic caching, feature similarity, and memory retrieval.

**Decision**: Enable pgvector extension on Neon. Use it for:
- LLM Gateway semantic cache (query embedding → cached response lookup)
- Agent long-term memory (facts about past workflows, dataset characteristics)
- Similarity search on historical experiment runs

**Consequences**:
- Positive: No additional vector database service needed (stays within $0 budget).
- Positive: Transactional consistency between relational and vector data.
- Negative: Neon free tier has 0.5 GB limit. Vector data grows quickly.
  Mitigation: embeddings stored for last 90 days only, older data pruned.

### ADR-005: Kaniko for In-Cluster Image Builds

**Status**: Accepted

**Context**: Building Docker images in Kubernetes typically requires a Docker
daemon (privileged container — security risk) or an external CI system.

**Decision**: Kaniko builds images inside Kubernetes Jobs without a Docker
daemon. No privileged containers. Image context pulled from R2.

**Consequences**:
- Positive: Zero privileged containers in the cluster — OPA policy enforces this.
- Positive: Build output goes directly to registry — no intermediate storage.
- Negative: Kaniko is slower than Docker BuildKit for large images.
  Mitigation: layer caching via R2 (Kaniko supports S3-compatible cache).

---

## Delivery Sequence

```
Phase 1 (Data Infrastructure)    → complete before Phase 2
Phase 2 (Real Agent Intelligence) → complete before Phase 3
Phase 3 (Model Registry)          → Phase 3 and Phase 4 can run in parallel
Phase 4 (OCI Image Packaging)     → depends on Phase 3 tags being in place
Phase 5 (Monitoring)              → depends on Phase 4 (needs real inference traffic)
Phase 6 (UI)                      → can start at Phase 2, finishes with Phase 5
```

Do not start a phase until the previous one's completion criteria are all checked.
Each phase is independently verifiable and independently shippable.

---

## What We Are NOT Building (Explicit Scope Boundary)

- Multi-tenancy: this is a single-team system. Multi-tenant SaaS is a separate product.
- Custom LLM training: we use pre-trained models. Fine-tuning is in scope; pre-training is not.
- Paid infrastructure: every component must remain on free tier or self-hosted.
- A general-purpose ML platform: this system is opinionated about the workflow.
  Teams that need full flexibility should use Kubeflow or SageMaker.

---

*Plan version 2.0 — ready for implementation once you confirm this sits right.*
*Begin with Phase 1 when ready.*
