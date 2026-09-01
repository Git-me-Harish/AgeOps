# tests/unit/test_api_server_realtime.py
"""
Unit tests for the Phase 6 additions to mcp_servers/api_server.py:
  - GET /api/agents/status (real pod health, replacing the V1 UI's
    hardcoded-'online' AgentStatusGrid)
  - /ws/events (WebSocket relay of agents/events.py publishes)

Kubernetes and Redis are mocked at their client boundaries, matching the
Kubernetes-mocking convention already used across this test suite
(training_agent/deployment_agent tests) — the request-handling and
data-shaping logic in api_server.py itself runs for real.
"""
from __future__ import annotations

import json
import threading
import time
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

import mcp_servers.api_server as api_server

client = TestClient(api_server.app)


def _make_agent_card(agent_id: str):
    card = MagicMock()
    card.agent_id = agent_id
    card.model_dump.return_value = {"agent_id": agent_id}
    return card


def _make_pod(app_label: str, phase: str, ready: bool, restarts: int = 0):
    pod = MagicMock()
    pod.metadata.labels = {"app": app_label}
    pod.status.phase = phase
    cs = MagicMock()
    cs.ready = ready
    cs.restart_count = restarts
    pod.status.container_statuses = [cs]
    return pod


class TestAgentPodStatus:
    def test_no_k8s_client_reports_unknown(self):
        with patch.object(api_server, "_registry") as mock_registry, \
             patch.object(api_server, "_init_k8s_client", return_value=None):
            mock_registry.list_all.return_value = [_make_agent_card("orchestrator_agent")]
            resp = client.get("/api/agents/status")
        assert resp.status_code == 200
        body = resp.json()
        assert body["agents"][0]["status"] == "unknown"
        assert "kubernetes client unavailable" in body["agents"][0]["reason"]

    def test_healthy_pod_reports_online(self):
        mock_pods = MagicMock()
        mock_pods.items = [_make_pod("orchestrator-agent", "Running", ready=True)]
        mock_core_v1 = MagicMock()
        mock_core_v1.list_namespaced_pod.return_value = mock_pods
        mock_k8s = MagicMock()
        mock_k8s.CoreV1Api.return_value = mock_core_v1

        with patch.object(api_server, "_registry") as mock_registry, \
             patch.object(api_server, "_init_k8s_client", return_value=mock_k8s):
            mock_registry.list_all.return_value = [_make_agent_card("orchestrator_agent")]
            resp = client.get("/api/agents/status")

        body = resp.json()
        assert body["agents"][0]["status"] == "online"
        assert body["agents"][0]["pod_count"] == 1

    def test_no_matching_pod_reports_unknown_not_online(self):
        """
        Regression guard for the exact V1 bug this replaces: AgentStatusGrid
        defaulted every agent to 'online' with no real data behind it. A
        missing pod must never look identical to a healthy one.
        """
        mock_pods = MagicMock()
        mock_pods.items = []
        mock_core_v1 = MagicMock()
        mock_core_v1.list_namespaced_pod.return_value = mock_pods
        mock_k8s = MagicMock()
        mock_k8s.CoreV1Api.return_value = mock_core_v1

        with patch.object(api_server, "_registry") as mock_registry, \
             patch.object(api_server, "_init_k8s_client", return_value=mock_k8s):
            mock_registry.list_all.return_value = [_make_agent_card("training_agent")]
            resp = client.get("/api/agents/status")

        body = resp.json()
        assert body["agents"][0]["status"] == "unknown"
        assert body["agents"][0]["reason"] == "no matching pod found"

    def test_not_ready_container_reports_degraded(self):
        mock_pods = MagicMock()
        mock_pods.items = [_make_pod("deployment-agent", "Running", ready=False)]
        mock_core_v1 = MagicMock()
        mock_core_v1.list_namespaced_pod.return_value = mock_pods
        mock_k8s = MagicMock()
        mock_k8s.CoreV1Api.return_value = mock_core_v1

        with patch.object(api_server, "_registry") as mock_registry, \
             patch.object(api_server, "_init_k8s_client", return_value=mock_k8s):
            mock_registry.list_all.return_value = [_make_agent_card("deployment_agent")]
            resp = client.get("/api/agents/status")

        assert resp.json()["agents"][0]["status"] == "degraded"


class TestWebSocketEventsRelay:
    def test_ws_relays_a_real_redis_publish(self, redis_container_url):
        """
        Exercises the actual /ws/events handler against a real disposable
        Redis instance (see the redis_container_url fixture) — not mocked —
        the same discipline used for Phase 5's Postgres/Prometheus/Loki
        verification. Publishes via agents.events' real sync publish
        helper (the code path deployment_agent.py uses) and asserts the
        WebSocket client actually receives the relayed frame.
        """
        import configs.settings as settings_module
        import agents.events as events_mod

        original_redis_url = settings_module.settings.redis_url
        settings_module.settings.redis_url = redis_container_url
        events_mod.settings = settings_module.settings
        events_mod._sync_redis_client = None
        try:
            def _publish():
                time.sleep(0.5)
                events_mod.publish_deployment_status_sync(
                    "ws-test-model", model_version="1", phase="canary_step", status="in_progress",
                )

            threading.Thread(target=_publish, daemon=True).start()

            with client.websocket_connect("/ws/events") as ws:
                received = None
                for _ in range(15):
                    data = ws.receive_json()
                    if data.get("channel", "").startswith("model.ws-test-model."):
                        received = data
                        break
                assert received is not None
                assert received["model_name"] == "ws-test-model"
                assert received["phase"] == "canary_step"
        finally:
            settings_module.settings.redis_url = original_redis_url
            events_mod._sync_redis_client = None


class TestWebSocketConnectionCap:
    """
    Regression coverage for a real stub found while wiring up rate
    limiting: settings.ws_max_connections_per_client has existed since it
    was first stubbed in during the Phase 6 planning pass, but /ws/events
    never actually enforced it — any client could open unlimited
    connections. Now backed by a real Redis counter (verified against a
    real disposable Redis instance, not mocked), correct across replicas
    since every orchestrator-agent pod shares the same Redis.
    """

    def test_connections_beyond_the_limit_are_rejected(self, redis_container_url):
        import configs.settings as settings_module

        original_redis_url = settings_module.settings.redis_url
        original_limit = settings_module.settings.ws_max_connections_per_client
        settings_module.settings.redis_url = redis_container_url
        settings_module.settings.ws_max_connections_per_client = 2
        try:
            with client.websocket_connect("/ws/events") as ws1:
                with client.websocket_connect("/ws/events") as ws2:
                    # Third connection from the same client (TestClient
                    # shares one host/IP) must be rejected — the cap is 2.
                    with client.websocket_connect("/ws/events") as ws3:
                        with pytest.raises(Exception):
                            # Starlette's TestClient raises when the server
                            # closes the socket during/just after connect.
                            ws3.receive_json()
        finally:
            settings_module.settings.redis_url = original_redis_url
            settings_module.settings.ws_max_connections_per_client = original_limit

    def test_connection_count_is_released_on_disconnect(self, redis_container_url):
        import configs.settings as settings_module
        import redis as sync_redis

        original_redis_url = settings_module.settings.redis_url
        original_limit = settings_module.settings.ws_max_connections_per_client
        settings_module.settings.redis_url = redis_container_url
        settings_module.settings.ws_max_connections_per_client = 1
        try:
            with client.websocket_connect("/ws/events"):
                pass  # connect then immediately disconnect

            # A fresh connection after the first fully closed must succeed
            # — proving the counter was decremented, not left stuck at 1.
            with client.websocket_connect("/ws/events") as ws:
                ws.close()
        finally:
            settings_module.settings.redis_url = original_redis_url
            settings_module.settings.ws_max_connections_per_client = original_limit
