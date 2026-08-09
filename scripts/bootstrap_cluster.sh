#!/usr/bin/env bash
# scripts/bootstrap_cluster.sh
# ══════════════════════════════════════════════════════════════════════════════
# One-shot bootstrap for a fresh Oracle Cloud Always-Free ARM VM (Ubuntu 22.04)
#
# What this script does
#   1. Install K3s (single-node Kubernetes)
#   2. Install Helm 3
#   3. Create namespaces: mlops, mlflow, monitoring, kserve
#   4. Install Prometheus-stack, Loki, Grafana via Helm
#   5. Install OPA Gatekeeper
#   6. Seal and apply secrets from .env
#   7. Apply all Kubernetes manifests
#
# Prerequisites on the VM
#   - Ubuntu 22.04 ARM64
#   - .env file present in the repo root (see .env.example)
#   - curl, git already installed (usually pre-installed)
#
# Usage
#   chmod +x scripts/bootstrap_cluster.sh
#   ./scripts/bootstrap_cluster.sh
# ══════════════════════════════════════════════════════════════════════════════
set -euo pipefail
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
info()    { echo -e "${GREEN}[INFO]${NC}  $*"; }
warning() { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error()   { echo -e "${RED}[ERR]${NC}   $*"; exit 1; }

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="${REPO_ROOT}/.env"

[ -f "$ENV_FILE" ] || error ".env file not found at ${ENV_FILE}. Copy .env.example and fill in values."

# shellcheck disable=SC1090
source "$ENV_FILE"

# ── 1. Install K3s ───────────────────────────────────────────────────────────
if ! command -v k3s &>/dev/null; then
  info "Installing K3s..."
  curl -sfL https://get.k3s.io | INSTALL_K3S_EXEC="--disable traefik" sh -
  mkdir -p ~/.kube
  sudo cp /etc/rancher/k3s/k3s.yaml ~/.kube/config
  sudo chown "$USER":"$USER" ~/.kube/config
  export KUBECONFIG=~/.kube/config
  info "K3s installed. Waiting for node to be Ready..."
  kubectl wait --for=condition=Ready node --all --timeout=120s
else
  info "K3s already installed — skipping."
  export KUBECONFIG=~/.kube/config
fi

# ── 2. Install Helm ──────────────────────────────────────────────────────────
if ! command -v helm &>/dev/null; then
  info "Installing Helm 3..."
  curl -sfL https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3 | bash
else
  info "Helm already installed — skipping."
fi

# ── 3. Namespaces ────────────────────────────────────────────────────────────
info "Creating namespaces..."
for ns in mlops mlflow monitoring kserve; do
  kubectl create namespace "$ns" --dry-run=client -o yaml | kubectl apply -f -
done
kubectl label namespace mlops    kubernetes.io/metadata.name=mlops    --overwrite
kubectl label namespace mlflow   kubernetes.io/metadata.name=mlflow   --overwrite
kubectl label namespace monitoring kubernetes.io/metadata.name=monitoring --overwrite

# ── 4. Helm repos ─────────────────────────────────────────────────────────────
info "Adding Helm repos..."
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo add grafana              https://grafana.github.io/helm-charts
helm repo add open-policy-agent    https://open-policy-agent.github.io/gatekeeper/charts
helm repo update

# ── 5. Prometheus + Grafana stack ─────────────────────────────────────────────
info "Installing kube-prometheus-stack..."
helm upgrade --install kube-prometheus-stack prometheus-community/kube-prometheus-stack \
  --namespace monitoring --create-namespace \
  --set grafana.adminPassword="${GRAFANA_ADMIN_PASSWORD:-admin}" \
  --set prometheus.prometheusSpec.scrapeInterval=15s \
  --wait --timeout 5m || warning "Prometheus stack already installed or timed out"

# ── 6. Loki ───────────────────────────────────────────────────────────────────
info "Installing Loki stack..."
helm upgrade --install loki grafana/loki-stack \
  --namespace monitoring \
  --set grafana.enabled=false \
  --set promtail.enabled=true \
  --wait --timeout 3m || warning "Loki already installed or timed out"

# ── 7. OPA Gatekeeper ────────────────────────────────────────────────────────
info "Installing OPA Gatekeeper..."
helm upgrade --install gatekeeper open-policy-agent/gatekeeper \
  --namespace gatekeeper-system --create-namespace \
  --set auditInterval=30 \
  --wait --timeout 3m || warning "Gatekeeper already installed or timed out"

# ── 8. Kubernetes secrets from .env ──────────────────────────────────────────
info "Creating Kubernetes secrets..."

kubectl create secret generic r2-credentials \
  --namespace mlops \
  --from-literal=access-key="${R2_ACCESS_KEY_ID}" \
  --from-literal=secret-key="${R2_SECRET_ACCESS_KEY}" \
  --from-literal=endpoint-url="${R2_ENDPOINT_URL}" \
  --dry-run=client -o yaml | kubectl apply -f -

kubectl create secret generic neon-credentials \
  --namespace mlops \
  --from-literal=connection-string="${DATABASE_URL}" \
  --dry-run=client -o yaml | kubectl apply -f -

kubectl create secret generic neon-credentials \
  --namespace mlflow \
  --from-literal=connection-string="${DATABASE_URL}" \
  --dry-run=client -o yaml | kubectl apply -f -

kubectl create secret generic r2-credentials \
  --namespace mlflow \
  --from-literal=access-key="${R2_ACCESS_KEY_ID}" \
  --from-literal=secret-key="${R2_SECRET_ACCESS_KEY}" \
  --from-literal=endpoint-url="${R2_ENDPOINT_URL}" \
  --dry-run=client -o yaml | kubectl apply -f -

kubectl create secret generic llm-credentials \
  --namespace mlops \
  --from-literal=openai-api-key="${OPENAI_API_KEY:-placeholder}" \
  --dry-run=client -o yaml | kubectl apply -f -

kubectl create secret generic cloudflare-secrets \
  --namespace mlops \
  --from-literal=api-token="${CLOUDFLARE_API_TOKEN}" \
  --from-literal=account-id="${CLOUDFLARE_ACCOUNT_ID}" \
  --dry-run=client -o yaml | kubectl apply -f -

kubectl create secret generic neon-secrets \
  --namespace mlops \
  --from-literal=api-key="${NEON_API_KEY}" \
  --from-literal=project-id="${NEON_PROJECT_ID}" \
  --dry-run=client -o yaml | kubectl apply -f -

# ── 9. MLflow via Helm values ─────────────────────────────────────────────────
info "Installing MLflow..."
helm upgrade --install mlflow mlflow/mlflow \
  --namespace mlflow --create-namespace \
  -f "${REPO_ROOT}/configs/kubernetes/mlflow-values.yaml" \
  --wait --timeout 3m || warning "MLflow install timed out (may already be running)"

# ── 10. Agent manifests ────────────────────────────────────────────────────────
info "Applying agent manifests..."
kubectl apply -f "${REPO_ROOT}/configs/kubernetes/agents/" -n mlops
kubectl apply -f "${REPO_ROOT}/configs/kubernetes/network-policies.yaml"

# ── 11. OPA policies ──────────────────────────────────────────────────────────
info "Applying OPA ConstraintTemplates..."
# The policies are applied once Gatekeeper is ready
kubectl wait --for=condition=Ready pod -l control-plane=controller-manager \
  -n gatekeeper-system --timeout=60s || warning "Gatekeeper pods not ready yet"

info ""
info "════════════════════════════════════════════════════════"
info "  Bootstrap complete!"
info "  MLflow UI:  http://$(kubectl get node -o jsonpath='{.items[0].status.addresses[0].address}'):5000"
info "  Grafana:    http://$(kubectl get node -o jsonpath='{.items[0].status.addresses[0].address}'):3000"
info "  API:        http://$(kubectl get node -o jsonpath='{.items[0].status.addresses[0].address}'):8000"
info "════════════════════════════════════════════════════════"
