#!/usr/bin/env bash
# scripts/setup_k3s_cluster.sh
# ════════════════════════════════════════════════════════════════════════
# Sets up a production 4-node K3s cluster on Oracle Cloud Free Tier ARM VMs.
#
# VM Layout (as per the technical document §3.1):
#   node-1  → K3s master  (control plane + etcd)
#   node-2  → K3s worker  (MLflow + MCP servers)
#   node-3  → K3s worker  (Agent workloads)
#   node-4  → K3s worker  (Monitoring: Prometheus + Grafana + Loki)
#
# Usage (run on your local machine, not the VMs):
#   chmod +x scripts/setup_k3s_cluster.sh
#   ./scripts/setup_k3s_cluster.sh \
#       --master 10.0.0.1 \
#       --workers "10.0.0.2,10.0.0.3,10.0.0.4" \
#       --ssh-key ~/.ssh/oracle_cloud_key.pem \
#       --ssh-user ubuntu
# ════════════════════════════════════════════════════════════════════════
set -euo pipefail
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; BLUE='\033[0;34m'; NC='\033[0m'
info()    { echo -e "${GREEN}[INFO]${NC}   $*"; }
step()    { echo -e "${BLUE}[STEP]${NC}   $*"; }
warning() { echo -e "${YELLOW}[WARN]${NC}   $*"; }
error()   { echo -e "${RED}[ERR]${NC}    $*"; exit 1; }

# ── Parse arguments ───────────────────────────────────────────────────────────
MASTER_IP=""
WORKER_IPS=""
SSH_KEY="~/.ssh/id_rsa"
SSH_USER="ubuntu"
K3S_VERSION="v1.31.1+k3s1"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --master)   MASTER_IP="$2";   shift 2 ;;
    --workers)  WORKER_IPS="$2";  shift 2 ;;
    --ssh-key)  SSH_KEY="$2";     shift 2 ;;
    --ssh-user) SSH_USER="$2";    shift 2 ;;
    --k3s-version) K3S_VERSION="$2"; shift 2 ;;
    *) error "Unknown argument: $1" ;;
  esac
done

[ -z "$MASTER_IP" ]  && error "--master <IP> is required"
[ -z "$WORKER_IPS" ] && error "--workers <IP,IP,...> is required"

SSH_OPTS="-i ${SSH_KEY} -o StrictHostKeyChecking=no -o ConnectTimeout=30"
IFS=',' read -ra WORKERS <<< "$WORKER_IPS"

# ── Helper: run command on remote host ────────────────────────────────────────
remote() {
  local host="$1"; shift
  # shellcheck disable=SC2086
  ssh $SSH_OPTS "${SSH_USER}@${host}" "$@"
}

remote_sudo() {
  local host="$1"; shift
  # shellcheck disable=SC2086
  ssh $SSH_OPTS "${SSH_USER}@${host}" "sudo bash -s" <<< "$@"
}

# ── Step 1: Install K3s master ────────────────────────────────────────────────
step "1/5  Installing K3s master on ${MASTER_IP}..."
remote "$MASTER_IP" \
  "curl -sfL https://get.k3s.io | INSTALL_K3S_VERSION=${K3S_VERSION} \
   K3S_KUBECONFIG_MODE=644 sh -s - server \
   --cluster-init \
   --disable traefik \
   --disable servicelb \
   --node-taint CriticalAddonsOnly=true:NoExecute"

info "Waiting 30 s for master to initialise..."
sleep 30

# ── Step 2: Retrieve join token ───────────────────────────────────────────────
step "2/5  Retrieving K3s join token..."
K3S_TOKEN=$(remote "$MASTER_IP" "sudo cat /var/lib/rancher/k3s/server/node-token")
info "Token retrieved (length=${#K3S_TOKEN})"

# ── Step 3: Retrieve kubeconfig ───────────────────────────────────────────────
step "3/5  Downloading kubeconfig..."
mkdir -p ~/.kube
# shellcheck disable=SC2086
scp $SSH_OPTS "${SSH_USER}@${MASTER_IP}:/etc/rancher/k3s/k3s.yaml" /tmp/k3s.yaml
# Replace 127.0.0.1 with the actual master IP
sed "s/127.0.0.1/${MASTER_IP}/g" /tmp/k3s.yaml > ~/.kube/config
chmod 600 ~/.kube/config
export KUBECONFIG=~/.kube/config
info "kubeconfig saved to ~/.kube/config"

# ── Step 4: Join worker nodes ─────────────────────────────────────────────────
step "4/5  Joining worker nodes..."
for WORKER_IP in "${WORKERS[@]}"; do
  info "Joining worker ${WORKER_IP}..."
  remote "$WORKER_IP" \
    "curl -sfL https://get.k3s.io | INSTALL_K3S_VERSION=${K3S_VERSION} \
     K3S_URL=https://${MASTER_IP}:6443 \
     K3S_TOKEN=${K3S_TOKEN} \
     sh -s - agent"
done

info "Waiting 30 s for workers to join..."
sleep 30

# ── Step 5: Label nodes by workload ──────────────────────────────────────────
step "5/5  Labelling nodes for workload placement..."
WORKER_ARRAY=("${WORKERS[@]}")

# Get node names from K8s (strip control-plane node)
ALL_NODES=$(kubectl get nodes -o jsonpath='{.items[*].metadata.name}')
read -ra NODE_NAMES <<< "$ALL_NODES"

# Label each worker node
if [ "${#NODE_NAMES[@]}" -ge 4 ]; then
  kubectl label node "${NODE_NAMES[1]}" workload=mlflow   --overwrite
  kubectl label node "${NODE_NAMES[2]}" workload=agents   --overwrite
  kubectl label node "${NODE_NAMES[3]}" workload=monitoring --overwrite
  info "Node labels applied:"
  kubectl get nodes --show-labels
fi

# ── Verify cluster ─────────────────────────────────────────────────────────────
info ""
info "════════════════════════════════════════════════════════"
kubectl get nodes -o wide
info "════════════════════════════════════════════════════════"
info "✅ K3s cluster ready!"
info "   Master:   ${MASTER_IP}"
info "   Workers:  ${WORKER_IPS}"
info "   Version:  ${K3S_VERSION}"
info ""
info "Next steps:"
info "  1. Run: ./scripts/bootstrap_cluster.sh   (installs Helm, MLflow, monitoring)"
info "  2. Run: kubectl apply -f configs/kubernetes/agents/  (deploys agents)"
info "════════════════════════════════════════════════════════"
