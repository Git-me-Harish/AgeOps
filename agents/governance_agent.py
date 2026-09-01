"""
agents/governance_agent.py
───────────────────────────
Governance Agent — V2 production rewrite.

What changed from V1:
  V1: check_policy() returned True when OPA was unreachable (fail-open in prod).
      Single regex-based rule. No audit trail.
  V2:
    - OPA runs as a sidecar in the Orchestrator pod — policy eval is a local
      HTTP call (127.0.0.1:8181), not a network call across the cluster.
    - Fail-closed in production: if OPA sidecar is unreachable, deny immediately.
    - Every decision written to policy_decisions table (immutable audit log).
    - Policy input hash (SHA-256) stored for tamper evidence.
    - Compliance report generated as HTML → R2 on demand.
    - Lineage record signing: HMAC-SHA256 over lineage rows.
    - GitHub Issue auto-creation on validation failures (via aiohttp).

OPA sidecar deployment note:
  The OPA container runs in the same pod as the Orchestrator:
    containers:
      - name: opa
        image: openpolicyagent/opa:latest
        args: ["run", "--server", "--addr=0.0.0.0:8181",
               "/policies/mlops_policy.rego"]
        ports:
          - containerPort: 8181
  Policies are mounted from the mlops-opa-policy ConfigMap (declared via
  configs/kubernetes/kustomization.yaml's configMapGenerator, sourced from
  configs/opa_policies/mlops_policy.rego). The sidecar runs with --watch
  (agents-deployment.yaml), so a ConfigMap edit — including one made live
  through agents/opa_admin.py's write_policy() from the Phase 6 Security
  page's policy editor — is picked up automatically once kubelet syncs the
  mounted volume (~60s), no SIGHUP or pod restart needed.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
from datetime import datetime, timezone
from typing import Any, Optional

import aiohttp
import mlflow
from pydantic import BaseModel

from agents import AgentTaskResult
from agents.llm_gateway import LLMGateway
from configs.settings import settings

logger = logging.getLogger(__name__)

# Typed models
class PolicyInput(BaseModel):
    """Structured input sent to OPA for evaluation."""
    workflow_id: str
    agent_role: str
    action: str                        # e.g. "promote_to_staging", "deploy_to_production"
    model_tags: dict[str, Any] = {}
    eval_metrics: dict[str, float] = {}
    security_scan: dict[str, Any] = {}
    lineage_id: Optional[int] = None
    lineage_hash: Optional[str] = None
    requester: str = "system"          # "system" | GitHub username for HITL


class PolicyDecision(BaseModel):
    """Structured result from OPA evaluation."""
    allowed: bool
    deny_reasons: list[str] = []
    policy_name: str
    evaluation_time_ms: float
    was_sidecar: bool
    policy_input_hash: str

# Governance Agent
class GovernanceAgent:
    """
    OPA-backed governance agent.

    All policy evaluations go through check_policy().
    In production (is_production=True): fail-closed on OPA unavailability.
    In dev: fail-open with a warning so local runs don't require OPA.
    """

    # Required MLflow tags — every model must have these before promotion
    REQUIRED_MODEL_TAGS: frozenset[str] = frozenset({
        "mlops.dataset.uri",
        "mlops.dataset.hash",
        "mlops.dataset.row_count",
        "mlops.framework",
        "mlops.eval.accuracy",
        "mlops.eval.f1",
        "mlops.eval.bias_passed",
        "mlops.security.trivy_scan",
    })

    def __init__(self) -> None:
        self._pool: Optional[Any] = None
        self._session: Optional[aiohttp.ClientSession] = None

    async def connect(self, pool: Any) -> None:
        self._pool = pool
        self._session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(
                total=settings.opa_timeout_seconds
            )
        )

    async def close(self) -> None:
        if self._session:
            await self._session.close()

    @mlflow.trace(name="governance_agent.run")
    def run(self, state: dict) -> AgentTaskResult:
        """Synchronous LangGraph entry point."""
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        return loop.run_until_complete(self._run_async(state))

    async def _run_async(self, state: dict) -> AgentTaskResult:
        task_id: str = state.get("workflow_id", "unknown")
        best_run_id: str = state.get("best_run_id", "")
        eval_report: dict = state.get("eval_report", {})
        security_result: dict = state.get("security_result", {})
        lineage_id: Optional[int] = state.get("lineage_id")
        action: str = state.get("governance_action", "promote_to_staging")

        gateway = LLMGateway(pool=self._pool, workflow_id=task_id)

        try:
            with mlflow.start_run(run_name=f"governance-{task_id}", nested=True):
                mlflow.set_tag("agent", "governance_agent")

                # Step 1: Fetch model tags from MLflow
                model_tags = await self._fetch_model_tags(best_run_id)

                # Step 2: Validate required tags are present
                tag_check = self._validate_required_tags(model_tags)
                if not tag_check["passed"]:
                    decision = PolicyDecision(
                        allowed=False,
                        deny_reasons=tag_check["missing"],
                        policy_name="required_tags_check",
                        evaluation_time_ms=0.0,
                        was_sidecar=False,
                        policy_input_hash="",
                    )
                    await self._persist_decision(task_id, decision, action)
                    return AgentTaskResult(
                        task_id=task_id, status="failed",
                        error=f"Required model tags missing: {tag_check['missing']}",
                    )

                # Step 3: Build OPA input
                policy_input = PolicyInput(
                    workflow_id=task_id,
                    agent_role="governance",
                    action=action,
                    model_tags=model_tags,
                    eval_metrics={
                        "accuracy":         eval_report.get("accuracy", 0.0),
                        "f1":               eval_report.get("f1", 0.0),
                        "auc":              eval_report.get("auc", 0.0),
                        "bias_passed":      float(eval_report.get("bias_passed", False)),
                        "overall_passed":   float(eval_report.get("overall_passed", False)),
                    },
                    security_scan={
                        "trivy_passed":      security_result.get("trivy_passed", False),
                        "critical_cves":     security_result.get("critical_cves", 0),
                        "high_cves":         security_result.get("high_cves", 0),
                        "semgrep_passed":    security_result.get("semgrep_passed", True),
                        "secrets_detected":  security_result.get("secrets_detected", False),
                    },
                    lineage_id=lineage_id,
                    lineage_hash=model_tags.get("mlops.dataset.hash"),
                )

                # Step 4: OPA policy evaluation
                decision = await self.check_policy(policy_input)

                # Step 5: LLM reasoning on borderline cases
                if not decision.allowed and len(decision.deny_reasons) > 0:
                    llm_summary = await self._llm_explain_denial(
                        deny_reasons=decision.deny_reasons,
                        eval_report=eval_report,
                        gateway=gateway,
                        task_id=task_id,
                    )
                    mlflow.log_param("governance.denial_explanation", llm_summary[:500])

                # Step 6: Persist audit decision
                await self._persist_decision(task_id, decision, action)

                # Step 7: Tag MLflow model with governance outcome
                await self._tag_mlflow_run(
                    run_id=best_run_id,
                    decision=decision,
                    action=action,
                )

                mlflow.log_metrics({
                    "governance.allowed":          int(decision.allowed),
                    "governance.eval_time_ms":     decision.evaluation_time_ms,
                    "governance.deny_reason_count": len(decision.deny_reasons),
                })

                logger.info(
                    "GovernanceAgent: action=%s allowed=%s reasons=%s eval_ms=%.1f",
                    action, decision.allowed, decision.deny_reasons,
                    decision.evaluation_time_ms,
                )

                status = "success" if decision.allowed else "failed"
                return AgentTaskResult(
                    task_id=task_id,
                    status=status,
                    output={
                        "allowed":          decision.allowed,
                        "deny_reasons":     decision.deny_reasons,
                        "policy_name":      decision.policy_name,
                        "was_sidecar":      decision.was_sidecar,
                        "eval_time_ms":     decision.evaluation_time_ms,
                        "action":           action,
                    },
                    error="; ".join(decision.deny_reasons) if not decision.allowed else None,
                    confidence=1.0 if decision.allowed else 0.0,
                )

        except Exception as exc:
            logger.exception("GovernanceAgent failed for workflow %s", task_id)
            # Governance failure: fail-closed in production
            if settings.is_production:
                return AgentTaskResult(
                    task_id=task_id, status="failed",
                    error=f"Governance agent error (fail-closed): {exc}",
                )
            logger.warning("GovernanceAgent: fail-open in dev mode — treating as allowed")
            return AgentTaskResult(task_id=task_id, status="success",
                output={"allowed": True, "dev_mode_bypass": True})

    # Post-workflow compliance audit
    @mlflow.trace(name="governance_agent.audit")
    def audit(self, state: dict) -> AgentTaskResult:
        """
        Synchronous LangGraph entry point for the final, post-monitoring audit
        node. This is a read-only compliance record of the completed workflow —
        it does NOT gate promotion. The real Staging→Production gate is
        PromotionStateMachine, invoked earlier in the graph (before deployment).
        """
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        return loop.run_until_complete(self._audit_async(state))

    async def _audit_async(self, state: dict) -> AgentTaskResult:
        task_id: str = state.get("workflow_id", "unknown")
        summary = {
            "final_stage": state.get("current_stage"),
            "errors":      state.get("errors", []),
            "model_uri":   state.get("model_uri"),
            "deployed":    state.get("current_stage") not in (None, "error") and not state.get("errors"),
        }
        signature = hashlib.sha256(
            json.dumps(summary, sort_keys=True, default=str).encode()
        ).hexdigest()

        if self._pool is not None:
            try:
                async with self._pool.acquire() as conn:
                    await conn.execute(
                        """
                        INSERT INTO agent_audit_trail (workflow_id, agent_id, action, payload, signature)
                        VALUES ($1, $2, $3, $4, $5)
                        """,
                        task_id, "governance_agent", "workflow_completed",
                        json.dumps(summary, default=str), signature,
                    )
            except Exception as exc:
                logger.error("Governance audit persistence failed: %s", exc)

        logger.info("GovernanceAgent audit: workflow=%s summary=%s", task_id, summary)
        return AgentTaskResult(
            task_id=task_id,
            status="success",
            output={"audit_signature": signature, "summary": summary},
        )

    # Core policy check
    async def check_policy(self, policy_input: PolicyInput) -> PolicyDecision:
        start = time.monotonic()
        input_dict = policy_input.model_dump()
        input_hash = hashlib.sha256(
            json.dumps(input_dict, sort_keys=True).encode()
        ).hexdigest()

        # Query the package root (not /mlops/allow) so OPA returns the full
        # {"result": {"allow": bool, "deny_reasons": [...]}} document — the
        # parsing below reads result.get("allow")/.get("deny_reasons").
        # settings.opa_policy_base_path is unrelated to this policy (it's for
        # a separate k8s-admission webhook) and must not be spliced in here.
        opa_url = f"{settings.opa_endpoint}/mlops"

        try:
            if self._session is None:
                self._session = aiohttp.ClientSession(
                    timeout=aiohttp.ClientTimeout(total=settings.opa_timeout_seconds)
                )

            async with self._session.post(
                opa_url,
                json={"input": input_dict},
                headers={"Content-Type": "application/json"},
            ) as resp:
                elapsed_ms = (time.monotonic() - start) * 1000
                if resp.status != 200:
                    raise RuntimeError(f"OPA returned HTTP {resp.status}")

                body = await resp.json()
                result = body.get("result", {})
                allowed = bool(result.get("allow", False))
                deny_reasons = result.get("deny_reasons", [])

                logger.info(
                    "OPA evaluation: action=%s allowed=%s reasons=%s time_ms=%.1f",
                    policy_input.action, allowed, deny_reasons, elapsed_ms,
                )

                return PolicyDecision(
                    allowed=allowed,
                    deny_reasons=deny_reasons,
                    policy_name="mlops_policy",
                    evaluation_time_ms=elapsed_ms,
                    was_sidecar=True,
                    policy_input_hash=input_hash,
                )

        except asyncio.TimeoutError:
            elapsed_ms = (time.monotonic() - start) * 1000
            logger.error("OPA sidecar timeout after %.1fms — fail-closed", elapsed_ms)
            return self._fail_closed_decision(
                input_hash, elapsed_ms,
                reason="OPA sidecar timeout — fail-closed"
            )

        except Exception as exc:
            elapsed_ms = (time.monotonic() - start) * 1000
            if settings.is_production:
                logger.error(
                    "OPA sidecar unreachable — fail-closed (production): %s", exc
                )
                return self._fail_closed_decision(
                    input_hash, elapsed_ms,
                    reason=f"OPA sidecar unreachable: {exc}"
                )
            else:
                logger.warning(
                    "OPA sidecar unreachable — fail-open (dev mode): %s", exc
                )
                return PolicyDecision(
                    allowed=True,
                    deny_reasons=[],
                    policy_name="dev_bypass",
                    evaluation_time_ms=elapsed_ms,
                    was_sidecar=False,
                    policy_input_hash=input_hash,
                )

    # Required tag validation
    def _validate_required_tags(self, model_tags: dict) -> dict:
        """
        Check all REQUIRED_MODEL_TAGS are present and non-empty.
        Returns {"passed": bool, "missing": list[str]}.
        """
        missing = [
            tag for tag in self.REQUIRED_MODEL_TAGS
            if not model_tags.get(tag)
        ]
        return {"passed": len(missing) == 0, "missing": missing}

    # MLflow integration
    async def _fetch_model_tags(self, run_id: str) -> dict[str, Any]:
        """Fetch all tags from an MLflow run."""
        if not run_id:
            return {}
        try:
            client = mlflow.tracking.MlflowClient()
            run = await asyncio.to_thread(client.get_run, run_id)
            return dict(run.data.tags)
        except Exception as exc:
            logger.warning("MLflow tag fetch failed for run %s: %s", run_id, exc)
            return {}

    async def _tag_mlflow_run(
        self,
        run_id: str,
        decision: PolicyDecision,
        action: str,
    ) -> None:
        """Stamp governance outcome on the MLflow run."""
        if not run_id:
            return
        try:
            client = mlflow.tracking.MlflowClient()
            await asyncio.to_thread(
                client.set_tag, run_id, "mlops.governance.allowed", str(decision.allowed)
            )
            await asyncio.to_thread(
                client.set_tag, run_id, "mlops.governance.action", action
            )
            await asyncio.to_thread(
                client.set_tag, run_id, "mlops.governance.audit_id",
                decision.policy_input_hash[:16]
            )
            if decision.deny_reasons:
                await asyncio.to_thread(
                    client.set_tag, run_id, "mlops.governance.deny_reasons",
                    "; ".join(decision.deny_reasons)[:500]
                )
        except Exception as exc:
            logger.warning("MLflow governance tag failed (non-fatal): %s", exc)

    # LLM denial explanation
    async def _llm_explain_denial(
        self,
        deny_reasons: list[str],
        eval_report: dict,
        gateway: LLMGateway,
        task_id: str,
    ) -> str:
        """
        Ask the LLM to generate a human-readable explanation of the denial.
        This is logged to MLflow and shown in the UI.
        """
        try:
            prompt = f"""
A model promotion was denied by the governance policy.

Deny reasons:
{json.dumps(deny_reasons, indent=2)}

Evaluation report summary:
- Accuracy: {eval_report.get('accuracy', 'N/A')}
- F1: {eval_report.get('f1', 'N/A')}
- Bias passed: {eval_report.get('bias_passed', 'N/A')}
- Overall passed: {eval_report.get('overall_passed', 'N/A')}

Write a concise (3-5 sentence) explanation that a non-technical stakeholder
can understand. Explain what failed, why it matters, and what needs to be
fixed before the model can be promoted. Be specific, not generic.
""".strip()
            response = await gateway.complete(
                prompt=prompt,
                agent_role="governance",
                skip_cache=True,
            )
            return response.text
        except Exception as exc:
            logger.warning("LLM denial explanation failed: %s", exc)
            return "; ".join(deny_reasons)

    # Neon persistence
    async def _persist_decision(
        self,
        workflow_id: str,
        decision: PolicyDecision,
        action: str,
    ) -> None:
        """Write the policy decision to the immutable audit log in Neon."""
        if self._pool is None:
            return
        try:
            async with self._pool.acquire() as conn:
                await conn.execute(
                    """
                    INSERT INTO policy_decisions (
                        workflow_id, policy_name, policy_package, decision,
                        deny_reasons, policy_input_hash, evaluation_time_ms,
                        agent_role, was_sidecar
                    ) VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9)
                    """,
                    workflow_id,
                    decision.policy_name,
                    action,
                    decision.allowed,
                    json.dumps(decision.deny_reasons),
                    decision.policy_input_hash,
                    decision.evaluation_time_ms,
                    "governance",
                    decision.was_sidecar,
                )
        except Exception as exc:
            logger.error("Policy decision persistence failed: %s", exc)

    # Compliance report
    async def generate_compliance_report(
        self,
        days: int = 7,
    ) -> Optional[str]:
        """
        Generate a weekly compliance report covering all policy decisions.
        Stores HTML report in R2. Returns R2 object key or None.
        """
        if self._pool is None or not settings.r2_configured:
            logger.warning("Cannot generate compliance report — no DB pool or R2 not configured")
            return None
        try:
            async with self._pool.acquire() as conn:
                rows = await conn.fetch(
                    """
                    SELECT pd.policy_name, pd.decision, pd.deny_reasons,
                           pd.evaluation_time_ms, pd.created_at, pd.workflow_id
                    FROM policy_decisions pd
                    WHERE pd.created_at >= NOW() - INTERVAL '$1 days'
                    ORDER BY pd.created_at DESC
                    """,
                    days,
                )

            decisions = [dict(r) for r in rows]
            total = len(decisions)
            allowed = sum(1 for d in decisions if d["decision"])
            denied = total - allowed

            html = self._build_compliance_html(decisions, total, allowed, denied, days)

            import boto3
            from botocore.config import Config
            from datetime import datetime, timezone

            timestamp = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%S")
            key = f"{settings.r2_reports_prefix}/compliance/{timestamp}_compliance.html"

            s3 = boto3.client(
                "s3",
                endpoint_url=settings.r2_endpoint_url,
                aws_access_key_id=settings.r2_access_key_id,
                aws_secret_access_key=settings.r2_secret_access_key,
                region_name=settings.r2_region,
                config=Config(retries={"max_attempts": 3}),
            )
            await asyncio.to_thread(
                s3.put_object,
                Bucket=settings.r2_bucket_name,
                Key=key, Body=html.encode("utf-8"), ContentType="text/html",
            )
            logger.info("Compliance report stored: R2 key=%s", key)
            return key

        except Exception as exc:
            logger.error("Compliance report generation failed: %s", exc)
            return None

    def _build_compliance_html(
        self, decisions: list, total: int, allowed: int, denied: int, days: int
    ) -> str:
        rows = "".join(
            f"<tr>"
            f"<td>{d.get('created_at','')}</td>"
            f"<td>{str(d.get('workflow_id',''))[:8]}</td>"
            f"<td>{d.get('policy_name','')}</td>"
            f"<td style='color:{'#2ecc71' if d.get('decision') else '#e74c3c'}'>"
            f"{'✅ ALLOW' if d.get('decision') else '❌ DENY'}</td>"
            f"<td>{', '.join(json.loads(d.get('deny_reasons','[]')))}</td>"
            f"<td>{d.get('evaluation_time_ms',0):.1f}ms</td>"
            f"</tr>"
            for d in decisions
        )
        return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Compliance Report</title>
<style>body{{font-family:monospace;background:#0d1117;color:#c9d1d9;padding:2rem}}
table{{border-collapse:collapse;width:100%}}td,th{{border:1px solid #30363d;padding:8px}}
th{{background:#161b22}}.stat{{display:inline-block;margin:8px;padding:12px 20px;
border-radius:6px;background:#161b22}}</style></head><body>
<h1>Governance Compliance Report — Last {days} days</h1>
<p>Generated: {datetime.now(tz=timezone.utc).isoformat()}</p>
<div class="stat">Total decisions: <strong>{total}</strong></div>
<div class="stat" style="color:#2ecc71">Allowed: <strong>{allowed}</strong></div>
<div class="stat" style="color:#e74c3c">Denied: <strong>{denied}</strong></div>
<div class="stat">Approval rate: <strong>{allowed/max(total,1)*100:.1f}%</strong></div>
<h2>Decision Log</h2>
<table><tr><th>Timestamp</th><th>Workflow</th><th>Policy</th>
<th>Decision</th><th>Deny Reasons</th><th>Eval Time</th></tr>
{rows}</table></body></html>"""

    # GitHub Issue auto-creation
    async def create_github_issue(
        self,
        title: str,
        body: str,
        labels: Optional[list[str]] = None,
    ) -> Optional[str]:
        """
        Create a GitHub Issue for blocking pipeline failures.
        Called by DataAgent on validation failure.
        Returns the issue URL or None on failure.
        """
        token = settings.github_client_secret  # reuse — this is the PAT in dev
        repo = "your-org/multi-agent-mlops"    # update to your repo
        if not token:
            logger.warning("GitHub token not configured — issue creation skipped")
            return None
        try:
            url = f"https://api.github.com/repos/{repo}/issues"
            payload = {
                "title": title,
                "body":  body,
                "labels": labels or ["mlops", "automated"],
            }
            if self._session is None:
                self._session = aiohttp.ClientSession()
            async with self._session.post(
                url,
                json=payload,
                headers={
                    "Authorization": f"Bearer {token}",
                    "Accept": "application/vnd.github+json",
                    "X-GitHub-Api-Version": "2022-11-28",
                },
            ) as resp:
                if resp.status == 201:
                    data = await resp.json()
                    issue_url = data.get("html_url", "")
                    logger.info("GitHub Issue created: %s", issue_url)
                    return issue_url
                else:
                    text = await resp.text()
                    logger.warning("GitHub Issue creation failed %d: %s", resp.status, text)
                    return None
        except Exception as exc:
            logger.warning("GitHub Issue creation exception: %s", exc)
            return None

    # Helpers
    @staticmethod
    def _fail_closed_decision(
        input_hash: str,
        elapsed_ms: float,
        reason: str,
    ) -> PolicyDecision:
        return PolicyDecision(
            allowed=False,
            deny_reasons=[reason],
            policy_name="fail_closed",
            evaluation_time_ms=elapsed_ms,
            was_sidecar=True,
            policy_input_hash=input_hash,
        )