"""Webhook route-level model/provider runtime overrides."""

from types import SimpleNamespace

from gateway.run import GatewayRunner


def test_route_agent_overrides_resolve_provider_with_route_env(monkeypatch):
    runner = GatewayRunner.__new__(GatewayRunner)
    runner._session_model_overrides = {}

    captured = {}

    def fake_resolve_runtime_provider(**kwargs):
        captured.update(kwargs)
        return {
            "provider": "zai",
            "api_mode": "chat_completions",
            "base_url": kwargs["explicit_base_url"],
            "api_key": kwargs["explicit_api_key"],
            "command": None,
            "args": [],
            "credential_pool": None,
        }

    monkeypatch.setattr(
        "hermes_cli.runtime_provider.resolve_runtime_provider",
        fake_resolve_runtime_provider,
    )

    model, runtime = runner._resolve_session_agent_runtime(
        session_key="webhook:review:delivery-1",
        user_config={"model": {"default": "global-model", "provider": "openrouter"}},
        agent_overrides={
            "provider": "zai",
            "model": "glm-5.1",
            "base_url": "https://api.z.ai/api/coding/paas/v4",
        },
        environment={"ZAI_API_KEY": "route-zai-key"},
    )

    assert model == "glm-5.1"
    assert captured == {
        "requested": "zai",
        "explicit_api_key": "route-zai-key",
        "explicit_base_url": "https://api.z.ai/api/coding/paas/v4",
        "target_model": "glm-5.1",
    }
    assert runtime["provider"] == "zai"
    assert runtime["api_key"] == "route-zai-key"
    assert runtime["base_url"] == "https://api.z.ai/api/coding/paas/v4"


def test_route_model_only_override_keeps_global_runtime(monkeypatch):
    runner = GatewayRunner.__new__(GatewayRunner)
    runner._session_model_overrides = {}

    monkeypatch.setattr(
        "gateway.run._resolve_runtime_agent_kwargs",
        lambda: {
            "provider": "openrouter",
            "api_key": "global-key",
            "base_url": "https://openrouter.ai/api/v1",
            "api_mode": "chat_completions",
            "model": "runtime-supplied-model",
        },
    )

    model, runtime = runner._resolve_session_agent_runtime(
        source=SimpleNamespace(),
        session_key="webhook:review:delivery-2",
        user_config={"model": {"default": "global-model"}},
        agent_overrides={"model": "route-model"},
    )

    assert model == "route-model"
    assert runtime["provider"] == "openrouter"
    assert "model" not in runtime


def test_route_environment_is_context_local_passthrough_to_tools():
    from tools.code_execution_tool import _scrub_child_env
    from tools.env_passthrough import reset_env_overrides, set_env_overrides
    from tools.environments.local import _sanitize_subprocess_env

    token = set_env_overrides({"GITHUB_TOKEN": "route-gh-token"})
    try:
        assert _sanitize_subprocess_env({})["GITHUB_TOKEN"] == "route-gh-token"
        assert _scrub_child_env({})["GITHUB_TOKEN"] == "route-gh-token"
    finally:
        reset_env_overrides(token)

    assert "GITHUB_TOKEN" not in _sanitize_subprocess_env({})
