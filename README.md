# Multi-Agent MLOps Platform

> Production-grade, end-to-end ML operations platform powered by a 10-agent LangGraph pipeline — built entirely on free-tier infrastructure.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                        React UI (Mantine + ReactFlow)               │
│   Dashboard │ Workflow Designer │ Experiments │ Models │ Security   │
└──────────────────────────┬──────────────────────────────────────────┘
                           │ REST /api/*
┌──────────────────────────▼──────────────────────────────────────────┐
│                    FastAPI REST API Server                          │
│                   mcp_servers/api_server.py                        │
└───────┬───────────────────────────────────────────────────┬────────┘
        │ LangGraph workflow                                │ A2A Registry
┌───────▼───────────────────────────────────────────────────▼────────┐
│                    Orchestrator Agent                               │
│              agents/orchestrator.py  (LangGraph root)              │
└──┬────────┬─────────┬───────────┬────────────┬────────────┬────────┘
   │        │         │           │            │            │
Data    Training  Evaluation  Deployment  Monitoring  Security/Governance
Agent   Agent     Agent       Agent       Agent       Agents (guardrails)
   │        │         │           │            │
   ▼        ▼         ▼           ▼            ▼
 R2     K8s Job    MLflow      KServe      Prometheus
(S3)    (Batch)   evaluate()  Canary       + Loki
   │        │         │           │            │
   └────────┴─────────┴───────────┴────────────┘
                        │
              MLflow Tracking (Neon PostgreSQL + R2)
                        │
                 RL Optimizer (daily CronJob, PPO)
```

### The 10 Agents

| Agent | Responsibility | Key Tools |
|-------|---------------|-----------|
| **Orchestrator** | Workflow routing, HITL gates | LangGraph StateGraph, Neon checkpointer |
| **Planner** | Execution plan + RL integration | MLflow search_runs |
| **Data Agent** | Ingest → Validate → Drift → Feast | Great Expectations, Evidently, Boto3 |
| **Training Agent** | K8s Jobs + MLflow autolog | sklearn/XGBoost, MLflow Registry |
| **Evaluation Agent** | `mlflow.evaluate()` + bias + threshold gate | MLflow evaluate, custom judges |
| **Deployment Agent** | KServe canary rollout + auto-rollback | Kubernetes CRD, Prometheus metrics |
| **Monitoring Agent** | Live accuracy + drift + Prometheus metrics | Evidently, prometheus_client |
| **Governance Agent** | OPA policies + SHA-256 audit trail | Gatekeeper, structured Loki logs |
| **Security Agent** | Injection defense + PII detection + Trivy | Regex patterns, SARIF parsing |
| **RL Optimizer** | PPO on MLflow traces → hyperparameter recommendations | Stable Baselines3, Gymnasium |

---

## Infrastructure (Zero Cost)

| Service | Provider | Free Tier |
|---------|----------|-----------|
| Compute | Oracle Cloud Always-Free | 4× A1 ARM OCPU · 24 GB RAM |
| Kubernetes | K3s (self-hosted on Oracle) | Unlimited |
| Artifact Storage | Cloudflare R2 | 10 GB · 10M reads/month |
| Database | Neon PostgreSQL | 0.5 GB · 191.9 compute-hours/month |
| Model Serving | KServe on K3s | Self-hosted |
| Monitoring | Prometheus + Grafana + Loki | Self-hosted |

---

## Quick Start (Local Dev)

### Prerequisites
- Docker + Docker Compose
- Python 3.11+
- Node.js 20+

### 1. Clone and configure
```bash
git clone https://github.com/Git-me-Harish/multi-agent-mlops
cd multi-agent-mlops
cp .env.example .env
# Fill in R2 keys, Neon DB URL, and OpenAI key in .env
```

### 2. Start all services
```bash
docker compose up -d
```

Services started:
- **UI** → http://localhost:3000
- **API** → http://localhost:8000/api/docs
- **MLflow** → http://localhost:5000
- **Grafana** → http://localhost:3001 (admin/admin)
- **Prometheus** → http://localhost:9090

### 3. Run tests
```bash
pip install -r requirements.txt
pytest tests/unit -v                   # fast, no services needed
pytest tests/integration -v            # requires MLflow
pytest tests/security -v               # policy simulation tests
```

---

## Production Deployment (Oracle Cloud K3s)

### 1. Provision VM
Create two Always-Free ARM VMs on Oracle Cloud:
- Shape: `VM.Standard.A1.Flex` (2 OCPU · 12 GB RAM each)
- OS: Ubuntu 22.04

### 2. Bootstrap cluster
```bash
# SSH into your VM
scp -r . ubuntu@<VM_IP>:~/multi-agent-mlops
ssh ubuntu@<VM_IP>
cd ~/multi-agent-mlops
cp .env.example .env && nano .env   # fill in production values
chmod +x scripts/bootstrap_cluster.sh
./scripts/bootstrap_cluster.sh
```

This installs K3s, Helm, Prometheus stack, Loki, OPA Gatekeeper, and deploys all agents.

### 3. Set GitHub Secrets
In your GitHub repo → Settings → Secrets → Actions:

```
DOCKERHUB_USERNAME        your Docker Hub username
DOCKERHUB_TOKEN           Docker Hub access token
KUBECONFIG_STAGING        base64-encoded staging kubeconfig
KUBECONFIG_PRODUCTION     base64-encoded production kubeconfig
MLFLOW_STAGING_URI        http://<staging-node-ip>:5000
```

Push to `main` → CI/CD pipeline runs automatically.

---

## Development Workflow

### Run a pipeline manually
```bash
# Via REST API
curl -X POST http://localhost:8000/api/workflows \
  -H "Content-Type: application/json" \
  -d '{"dataset_uri": "s3://mlflow-artifacts/datasets/sample.csv"}'

# Response: {"workflow_id": "...", "status": "running"}
```

### Approve a human-in-the-loop gate
```bash
curl -X POST http://localhost:8000/api/workflows/<id>/approve \
  -H "Content-Type: application/json" \
  -d '{"approved": true, "reviewer": "harish", "reason": "metrics look good"}'
```

### Train the RL optimizer
```bash
python -m rl_agent.rl_optimizer   # trains and saves model to /models/rl_optimizer.zip
```

### Check budget usage
```bash
python scripts/budget_monitor.py
```

### Run DB migrations
```bash
alembic upgrade head
```

---

## Configuration Reference

All configuration is via environment variables (see `.env.example`).

Key agent limits:

| Variable | Default | Description |
|----------|---------|-------------|
| `AGENT_MAX_ITERATIONS` | 20 | Max LangGraph loop iterations |
| `AGENT_TASK_TIMEOUT_SECONDS` | 120 | Per-agent task timeout |
| `AGENT_MAX_CONCURRENT` | 10 | Max parallel agent tasks |
| `R2_STORAGE_ALERT_GB` | 8.0 | Alert at 80% of 10 GB free tier |
| `NEON_STORAGE_ALERT_MB` | 400.0 | Alert at 80% of 0.5 GB free tier |

---

## Project Structure

```
multi-agent-mlops/
├── agents/                  # All 10 LangGraph agents
│   ├── __init__.py          # AgentState TypedDict + AgentTaskResult
│   ├── orchestrator.py      # Root workflow graph
│   ├── data_agent.py        # Ingest → validate → drift → Feast
│   ├── training_agent.py    # K8s Jobs + MLflow autolog
│   ├── evaluation_agent.py  # mlflow.evaluate() + threshold gate
│   ├── deployment_agent.py  # KServe canary + rollback
│   ├── monitoring_agent.py  # Live metrics + Prometheus
│   ├── governance_agent.py  # OPA + audit trail
│   └── security_agent.py    # Injection + PII + Trivy
├── mcp_servers/
│   ├── api_server.py        # Main REST API (consumed by UI)
│   └── mlflow_server.py     # MLflow MCP Server (agent tools)
├── rl_agent/
│   └── rl_optimizer.py      # PPO optimizer (daily CronJob)
├── configs/
│   ├── settings.py          # Pydantic Settings (all env vars)
│   ├── a2a_registry/        # Agent cards + registry
│   ├── kubernetes/          # K8s manifests + Helm values
│   └── monitoring/          # Prometheus + Grafana config
├── policies/
│   └── opa_rego/            # Gatekeeper policy rules
├── feast/                   # Feature store repo
├── scripts/
│   ├── bootstrap_cluster.sh # One-shot K3s bootstrap
│   ├── budget_monitor.py    # Daily R2 + Neon usage check
│   ├── run_ai_eval.py       # CI eval gate
│   ├── check_eval_thresholds.py
│   └── migrations/          # Alembic DB migrations
├── tests/
│   ├── unit/                # No-service tests
│   ├── integration/         # Require MLflow
│   └── security/            # OPA + injection tests
├── ui/                      # React + Mantine frontend
│   └── src/
│       ├── pages/           # 6 pages
│       ├── components/      # Reusable UI components
│       ├── store/           # Zustand state
│       └── types/           # TypeScript types
├── Dockerfile               # Orchestrator + API
├── Dockerfile.mcp           # MLflow MCP Server
├── docker-compose.yaml      # Local dev stack
└── .github/workflows/
    └── ci-cd.yaml           # 7-job CI/CD pipeline
```

---

## License

MIT — built by [Harish](https://github.com/Git-me-Harish) as a production-grade MLOps reference implementation.
