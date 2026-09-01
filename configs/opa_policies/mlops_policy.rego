# configs/opa_policies/mlops_policy.rego
# ─────────────────────────────────────────────────────────────────────────────
# MLOps Governance Policy — OPA Rego v1
#
# Evaluated by GovernanceAgent via OPA sidecar at 127.0.0.1:8181.
# Mounted into the Orchestrator pod from a Kubernetes ConfigMap:
#
#   kubectl create configmap mlops-opa-policy \
#     --from-file=mlops_policy.rego=configs/opa_policies/mlops_policy.rego \
#     -n mlops
#
# Policy packages:
#   data.mlops.allow       → main gate (true = allow action, false = deny)
#   data.mlops.deny_reasons → list of human-readable denial reasons
#
# Actions covered:
#   promote_to_staging      → Staging model registration
#   promote_to_production   → Production model deployment
#   deploy_inference_service → KServe InferenceService creation
#   run_training_job        → Kubernetes training Job launch
#   trigger_retraining      → Automated retraining from Monitoring Agent
#
# Test with:
#   opa eval --data mlops_policy.rego \
#            --input input.json \
#            "data.mlops.allow"
#
# Lint with:
#   opa check mlops_policy.rego
# ─────────────────────────────────────────────────────────────────────────────

package mlops

import rego.v1

# ─────────────────────────────────────────────────────────────────────────────
# Top-level decision
# allow is true only when there are NO deny_reasons.
# ─────────────────────────────────────────────────────────────────────────────

default allow := false

allow if {
    count(deny_reasons) == 0
}

# ─────────────────────────────────────────────────────────────────────────────
# Deny reasons — collected as a set, returned as an array
# Each rule contributes one string to the deny set.
# allow = false when any deny rule fires.
# ─────────────────────────────────────────────────────────────────────────────

deny_reasons contains msg if {
    input.action in {"promote_to_staging", "promote_to_production", "deploy_inference_service"}
    missing := required_tags_missing
    count(missing) > 0
    msg := sprintf("Required model tags missing: %v", [missing])
}

deny_reasons contains msg if {
    input.action in {"promote_to_staging", "promote_to_production"}
    input.eval_metrics.overall_passed != true
    msg := "Evaluation gate not passed: eval_metrics.overall_passed must be true"
}

deny_reasons contains msg if {
    input.action in {"promote_to_staging", "promote_to_production"}
    input.eval_metrics.bias_passed != true
    msg := "Bias check failed: model cannot be promoted with unmitigated fairness issues"
}

deny_reasons contains msg if {
    input.action in {"promote_to_staging", "promote_to_production", "deploy_inference_service"}
    input.security_scan.trivy_passed != true
    msg := sprintf(
        "Trivy CVE scan failed: critical=%v high=%v — remediate before promoting",
        [
            object.get(input.security_scan, "critical_cves", 0),
            object.get(input.security_scan, "high_cves", 0),
        ]
    )
}

deny_reasons contains msg if {
    input.action in {"promote_to_staging", "promote_to_production", "deploy_inference_service"}
    input.security_scan.secrets_detected == true
    msg := "Secrets detected in dataset or model artifacts — pipeline halted for security review"
}

deny_reasons contains msg if {
    input.action in {"promote_to_staging", "promote_to_production"}
    input.security_scan.semgrep_passed != true
    msg := "Semgrep SAST found ERROR severity findings in training script — review before promoting"
}

deny_reasons contains msg if {
    input.action == "promote_to_production"
    not approved_by_human
    msg := "Production promotion requires human approval — no approval record found in model tags"
}

deny_reasons contains msg if {
    input.action == "promote_to_production"
    lineage_missing
    msg := "Model lineage not verified — dataset hash missing from model tags"
}

deny_reasons contains msg if {
    input.action == "run_training_job"
    input.security_scan.secrets_detected == true
    msg := "Cannot launch training job: secrets detected in dataset"
}

deny_reasons contains msg if {
    input.action == "deploy_inference_service"
    not input.security_scan.trivy_passed
    msg := "Cannot deploy: container image has not passed Trivy CVE scan"
}

# ─────────────────────────────────────────────────────────────────────────────
# Helper rules
# ─────────────────────────────────────────────────────────────────────────────

# Required tags for every model that enters the promotion pipeline
required_tags := {
    "mlops.dataset.uri",
    "mlops.dataset.hash",
    "mlops.dataset.row_count",
    "mlops.framework",
    "mlops.eval.accuracy",
    "mlops.eval.f1",
    "mlops.eval.bias_passed",
    "mlops.security.trivy_scan",
}

required_tags_missing contains tag if {
    some tag in required_tags
    not input.model_tags[tag]
}

# Human approval: a GitHub username must be present in the approved_by tag
approved_by_human if {
    approver := input.model_tags["mlops.approved_by"]
    approver != ""
    approver != "system"
    approver != "data-agent"
}

# Lineage: dataset hash must be present and non-empty
lineage_missing if {
    not input.model_tags["mlops.dataset.hash"]
}

lineage_missing if {
    input.model_tags["mlops.dataset.hash"] == ""
}

# ─────────────────────────────────────────────────────────────────────────────
# Action-level permission matrix
# Which agent roles are allowed to trigger which actions.
# ─────────────────────────────────────────────────────────────────────────────

allowed_actions := {
    "planner":    {"run_training_job", "trigger_retraining"},
    "training":   {"run_training_job"},
    "evaluation": {"promote_to_staging"},
    "governance": {"promote_to_staging", "promote_to_production", "deploy_inference_service"},
    "monitoring": {"trigger_retraining"},
    "security":   {"run_training_job"},
    "system":     {"run_training_job", "trigger_retraining", "promote_to_staging"},
}

deny_reasons contains msg if {
    permitted := allowed_actions[input.agent_role]
    not input.action in permitted
    msg := sprintf(
        "Agent role '%v' is not permitted to perform action '%v'",
        [input.agent_role, input.action]
    )
}

# Catch unknown agent roles
deny_reasons contains msg if {
    not input.agent_role in object.keys(allowed_actions)
    msg := sprintf("Unknown agent role: '%v'", [input.agent_role])
}

# ─────────────────────────────────────────────────────────────────────────────
# Metric thresholds by action (more permissive for staging than production)
# Override the default 0.65/0.70 thresholds for staging promotion.
# ─────────────────────────────────────────────────────────────────────────────

staging_f1_threshold    := 0.60
production_f1_threshold := 0.65
staging_acc_threshold   := 0.65
production_acc_threshold := 0.70

deny_reasons contains msg if {
    input.action == "promote_to_staging"
    input.eval_metrics.f1 < staging_f1_threshold
    msg := sprintf(
        "F1 score %.3f below staging threshold %.2f",
        [input.eval_metrics.f1, staging_f1_threshold]
    )
}

deny_reasons contains msg if {
    input.action == "promote_to_staging"
    input.eval_metrics.accuracy < staging_acc_threshold
    msg := sprintf(
        "Accuracy %.3f below staging threshold %.2f",
        [input.eval_metrics.accuracy, staging_acc_threshold]
    )
}

deny_reasons contains msg if {
    input.action == "promote_to_production"
    input.eval_metrics.f1 < production_f1_threshold
    msg := sprintf(
        "F1 score %.3f below production threshold %.2f",
        [input.eval_metrics.f1, production_f1_threshold]
    )
}

deny_reasons contains msg if {
    input.action == "promote_to_production"
    input.eval_metrics.accuracy < production_acc_threshold
    msg := sprintf(
        "Accuracy %.3f below production threshold %.2f",
        [input.eval_metrics.accuracy, production_acc_threshold]
    )
}