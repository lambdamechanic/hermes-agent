import json
import sys
import types
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
TOOLS_DIR = REPO_ROOT / "tools"
ENVIRONMENTS_DIR = REPO_ROOT / "environments"


def _load_module(module_name: str, path: Path):
    spec = spec_from_file_location(module_name, path)
    assert spec and spec.loader
    module = module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _reset_modules(prefixes: tuple[str, ...]):
    for name in list(sys.modules):
        if name.startswith(prefixes):
            sys.modules.pop(name, None)


def _install_modal_test_modules(tmp_path: Path, swerex_environment_cls) -> Path:
    _reset_modules(("tools", "hermes_cli", "minisweagent", "environments"))

    hermes_cli = types.ModuleType("hermes_cli")
    hermes_cli.__path__ = []  # type: ignore[attr-defined]
    sys.modules["hermes_cli"] = hermes_cli
    hermes_home = tmp_path / "hermes-home"
    sys.modules["hermes_cli.config"] = types.SimpleNamespace(
        get_hermes_home=lambda: hermes_home,
    )

    tools_package = types.ModuleType("tools")
    tools_package.__path__ = [str(TOOLS_DIR)]  # type: ignore[attr-defined]
    sys.modules["tools"] = tools_package

    env_package = types.ModuleType("tools.environments")
    env_package.__path__ = [str(TOOLS_DIR / "environments")]  # type: ignore[attr-defined]
    sys.modules["tools.environments"] = env_package

    class _DummyBaseEnvironment:
        def __init__(self, cwd: str, timeout: int, env=None):
            self.cwd = cwd
            self.timeout = timeout
            self.env = env or {}

        def _prepare_command(self, command: str):
            return command, None

    sys.modules["tools.environments.base"] = types.SimpleNamespace(BaseEnvironment=_DummyBaseEnvironment)
    sys.modules["tools.interrupt"] = types.SimpleNamespace(is_interrupted=lambda: False)

    environments_package = types.ModuleType("environments")
    environments_package.__path__ = [str(ENVIRONMENTS_DIR)]  # type: ignore[attr-defined]
    sys.modules["environments"] = environments_package
    sys.modules["environments.patches"] = types.SimpleNamespace(apply_patches=lambda: None)

    minisweagent = types.ModuleType("minisweagent")
    minisweagent.__path__ = []  # type: ignore[attr-defined]
    sys.modules["minisweagent"] = minisweagent
    minisweagent_environments = types.ModuleType("minisweagent.environments")
    minisweagent_environments.__path__ = []  # type: ignore[attr-defined]
    sys.modules["minisweagent.environments"] = minisweagent_environments
    minisweagent_extra = types.ModuleType("minisweagent.environments.extra")
    minisweagent_extra.__path__ = []  # type: ignore[attr-defined]
    sys.modules["minisweagent.environments.extra"] = minisweagent_extra
    sys.modules["minisweagent.environments.extra.swerex_modal"] = types.SimpleNamespace(
        SwerexModalEnvironment=swerex_environment_cls,
    )

    return hermes_home / "modal_snapshots.json"


def _make_fake_swerex_env(calls: list[dict], *, fail_on_images: set[str] | None = None, snapshot_id: str = "im-fresh") -> type:
    class _FakeSwerexModalEnvironment:
        def __init__(self, **kwargs):
            calls.append(dict(kwargs))
            if fail_on_images and kwargs.get("image") in fail_on_images:
                raise RuntimeError(f"cannot restore {kwargs['image']}")

            async def _snapshot_aio():
                return types.SimpleNamespace(object_id=snapshot_id)

            self.kwargs = dict(kwargs)
            self.deployment = types.SimpleNamespace(
                _sandbox=types.SimpleNamespace(
                    snapshot_filesystem=types.SimpleNamespace(aio=_snapshot_aio),
                )
            )
            self.stop_called = False

        def stop(self):
            self.stop_called = True

    return _FakeSwerexModalEnvironment


def test_modal_environment_migrates_legacy_snapshot_key_and_uses_snapshot_id_string(tmp_path):
    calls: list[dict] = []
    snapshot_store = _install_modal_test_modules(tmp_path, _make_fake_swerex_env(calls))
    snapshot_store.parent.mkdir(parents=True, exist_ok=True)
    snapshot_store.write_text(json.dumps({"task-legacy": "im-legacy123"}))

    modal_module = _load_module("tools.environments.modal", TOOLS_DIR / "environments" / "modal.py")
    modal_module.ModalEnvironment(image="python:3.11", task_id="task-legacy")

    assert calls[0]["image"] == "im-legacy123"
    assert json.loads(snapshot_store.read_text()) == {"direct:task-legacy": "im-legacy123"}


def test_modal_environment_prunes_stale_direct_snapshot_and_retries_base_image(tmp_path):
    calls: list[dict] = []
    snapshot_store = _install_modal_test_modules(
        tmp_path,
        _make_fake_swerex_env(calls, fail_on_images={"im-stale123"}),
    )
    snapshot_store.parent.mkdir(parents=True, exist_ok=True)
    snapshot_store.write_text(json.dumps({"direct:task-stale": "im-stale123"}))

    modal_module = _load_module("tools.environments.modal", TOOLS_DIR / "environments" / "modal.py")
    env = modal_module.ModalEnvironment(image="python:3.11", task_id="task-stale")

    assert [call["image"] for call in calls] == ["im-stale123", "python:3.11"]
    assert env._inner.kwargs["image"] == "python:3.11"
    assert json.loads(snapshot_store.read_text()) == {}


def test_modal_environment_cleanup_writes_namespaced_snapshot_key(tmp_path):
    calls: list[dict] = []
    snapshot_store = _install_modal_test_modules(
        tmp_path,
        _make_fake_swerex_env(calls, snapshot_id="im-cleanup456"),
    )

    modal_module = _load_module("tools.environments.modal", TOOLS_DIR / "environments" / "modal.py")
    env = modal_module.ModalEnvironment(image="python:3.11", task_id="task-cleanup")
    env.cleanup()

    assert json.loads(snapshot_store.read_text()) == {"direct:task-cleanup": "im-cleanup456"}


def test_patched_swerex_modal_resolves_snapshot_ids_with_from_id():
    _reset_modules(("environments", "minisweagent", "swerex", "modal"))

    deployment_calls: list[dict] = []
    registry_calls: list[tuple[str, list[str] | None]] = []
    from_id_calls: list[str] = []

    class _FakeConfig:
        def __init__(self, **kwargs):
            self.image = kwargs["image"]
            self.cwd = kwargs.get("cwd", "/")
            self.timeout = kwargs.get("timeout", 30)
            self.env = kwargs.get("env", {})
            self.startup_timeout = kwargs.get("startup_timeout", 60.0)
            self.runtime_timeout = kwargs.get("runtime_timeout", 3600.0)
            self.deployment_timeout = kwargs.get("deployment_timeout", 3600.0)
            self.install_pipx = kwargs.get("install_pipx", True)
            self.modal_sandbox_kwargs = kwargs.get("modal_sandbox_kwargs", {})

    class _FakeSwerexModalEnvironment:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class _FakeModalDeployment:
        def __init__(self, **kwargs):
            deployment_calls.append(dict(kwargs))
            self.kwargs = kwargs

        async def start(self):
            return None

        async def stop(self):
            return None

    class _FakeRexCommand:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class _FakeImage:
        @staticmethod
        def from_registry(image: str, setup_dockerfile_commands=None):
            registry_calls.append((image, setup_dockerfile_commands))
            return {"kind": "registry", "image": image}

        @staticmethod
        def from_id(image_id: str):
            from_id_calls.append(image_id)
            return {"kind": "snapshot", "image_id": image_id}

    environments_package = types.ModuleType("environments")
    environments_package.__path__ = [str(ENVIRONMENTS_DIR)]  # type: ignore[attr-defined]
    sys.modules["environments"] = environments_package

    minisweagent = types.ModuleType("minisweagent")
    minisweagent.__path__ = []  # type: ignore[attr-defined]
    sys.modules["minisweagent"] = minisweagent
    minisweagent_environments = types.ModuleType("minisweagent.environments")
    minisweagent_environments.__path__ = []  # type: ignore[attr-defined]
    sys.modules["minisweagent.environments"] = minisweagent_environments
    minisweagent_extra = types.ModuleType("minisweagent.environments.extra")
    minisweagent_extra.__path__ = []  # type: ignore[attr-defined]
    sys.modules["minisweagent.environments.extra"] = minisweagent_extra
    sys.modules["minisweagent.environments.extra.swerex_modal"] = types.SimpleNamespace(
        SwerexModalEnvironment=_FakeSwerexModalEnvironment,
        SwerexModalEnvironmentConfig=_FakeConfig,
    )

    swerex = types.ModuleType("swerex")
    swerex.__path__ = []  # type: ignore[attr-defined]
    sys.modules["swerex"] = swerex
    swerex_deployment = types.ModuleType("swerex.deployment")
    swerex_deployment.__path__ = []  # type: ignore[attr-defined]
    sys.modules["swerex.deployment"] = swerex_deployment
    sys.modules["swerex.deployment.modal"] = types.SimpleNamespace(ModalDeployment=_FakeModalDeployment)
    swerex_runtime = types.ModuleType("swerex.runtime")
    swerex_runtime.__path__ = []  # type: ignore[attr-defined]
    sys.modules["swerex.runtime"] = swerex_runtime
    sys.modules["swerex.runtime.abstract"] = types.SimpleNamespace(Command=_FakeRexCommand)
    sys.modules["modal"] = types.SimpleNamespace(Image=_FakeImage)

    patches_module = _load_module("environments.patches", ENVIRONMENTS_DIR / "patches.py")
    patches_module._patch_swerex_modal()

    snapshot_env = _FakeSwerexModalEnvironment(image="im-snapshot123")
    registry_env = _FakeSwerexModalEnvironment(image="python:3.11")

    try:
        assert from_id_calls == ["im-snapshot123"]
        assert deployment_calls[0]["image"] == {"kind": "snapshot", "image_id": "im-snapshot123"}
        assert registry_calls[0][0] == "python:3.11"
        assert "ensurepip" in registry_calls[0][1][0]
        assert deployment_calls[1]["image"] == {"kind": "registry", "image": "python:3.11"}
    finally:
        snapshot_env.stop()
        registry_env.stop()
