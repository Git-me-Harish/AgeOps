# policies/opa_rego/block-unsafe-deployments.rego

package kubernetes.admission

import future.keywords.in

# ── Rule 1: Containers must not use 'latest' tag ──────────────────────────────
deny[msg] {
  input.request.kind.kind == "Deployment"
  container := input.request.object.spec.template.spec.containers[_]
  endswith(container.image, ":latest")
  msg := sprintf(
    "Container '%v' uses ':latest' tag. Use a specific semantic version (e.g. v1.2.3).",
    [container.name]
  )
}

# ── Rule 2: All containers must run as non-root ───────────────────────────────
deny[msg] {
  input.request.kind.kind == "Deployment"
  container := input.request.object.spec.template.spec.containers[_]
  not container.securityContext.runAsNonRoot
  msg := sprintf(
    "Container '%v' must set securityContext.runAsNonRoot: true",
    [container.name]
  )
}

# ── Rule 3: Deployment Agent specifically enforced ────────────────────────────
deny[msg] {
  input.request.kind.kind == "Deployment"
  input.request.object.metadata.name == "deployment-agent"
  not input.request.object.spec.template.spec.serviceAccountName
  msg := "deployment-agent Deployment must specify a serviceAccountName"
}

# ── Rule 4: Memory limits required ───────────────────────────────────────────
deny[msg] {
  input.request.kind.kind in {"Deployment", "Job"}
  container := input.request.object.spec.template.spec.containers[_]
  not container.resources.limits.memory
  msg := sprintf(
    "Container '%v' must declare resources.limits.memory",
    [container.name]
  )
}

# ── Rule 5: Privilege escalation forbidden ────────────────────────────────────
deny[msg] {
  input.request.kind.kind == "Deployment"
  container := input.request.object.spec.template.spec.containers[_]
  container.securityContext.allowPrivilegeEscalation == true
  msg := sprintf(
    "Container '%v' must not allow privilege escalation (allowPrivilegeEscalation: false)",
    [container.name]
  )
}

# ── Rule 6: KServe InferenceService must have resource limits ─────────────────
deny[msg] {
  input.request.kind.kind == "InferenceService"
  not input.request.object.spec.predictor.model
  msg := "InferenceService must specify spec.predictor.model"
}

# ── Rule 7: Agent pods must carry an 'agent' label ───────────────────────────
deny[msg] {
  input.request.kind.kind == "Pod"
  not input.request.object.metadata.labels.agent
  input.request.namespace == "mlops"
  msg := "All pods in the mlops namespace must carry an 'agent' label for audit traceability"
}
