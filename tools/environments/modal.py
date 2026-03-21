"""Modal cloud execution environment wrapping mini-swe-agent's SwerexModalEnvironment.

Supports persistent filesystem snapshots: when enabled, the sandbox's filesystem
is snapshotted on cleanup and restored on next creation, so installed packages,
project files, and config changes survive across sessions.
"""

import json
import logging
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

from hermes_cli.config import get_hermes_home
from tools.environments.base import BaseEnvironment
from tools.interrupt import is_interrupted

logger = logging.getLogger(__name__)

_SNAPSHOT_STORE = get_hermes_home() / "modal_snapshots.json"
_DIRECT_SNAPSHOT_NAMESPACE = "direct"


def _load_snapshots() -> Dict[str, str]:
    """Load snapshot ID mapping from disk."""
    if _SNAPSHOT_STORE.exists():
        try:
            return json.loads(_SNAPSHOT_STORE.read_text())
        except Exception:
            pass
    return {}


def _save_snapshots(data: Dict[str, str]) -> None:
    """Persist snapshot ID mapping to disk."""
    _SNAPSHOT_STORE.parent.mkdir(parents=True, exist_ok=True)
    _SNAPSHOT_STORE.write_text(json.dumps(data, indent=2))


def _direct_snapshot_key(task_id: str) -> str:
    return f"{_DIRECT_SNAPSHOT_NAMESPACE}:{task_id}"


def _get_snapshot_restore_candidate(task_id: str) -> tuple[str | None, bool]:
    """Return a snapshot id for direct Modal restore and whether the key is legacy."""
    snapshots = _load_snapshots()

    namespaced_key = _direct_snapshot_key(task_id)
    snapshot_id = snapshots.get(namespaced_key)
    if isinstance(snapshot_id, str) and snapshot_id:
        return snapshot_id, False

    legacy_snapshot_id = snapshots.get(task_id)
    if isinstance(legacy_snapshot_id, str) and legacy_snapshot_id:
        return legacy_snapshot_id, True

    return None, False


def _store_direct_snapshot(task_id: str, snapshot_id: str) -> None:
    """Persist the direct Modal snapshot id under the direct namespace."""
    snapshots = _load_snapshots()
    snapshots[_direct_snapshot_key(task_id)] = snapshot_id
    snapshots.pop(task_id, None)
    _save_snapshots(snapshots)


def _delete_direct_snapshot(task_id: str, snapshot_id: str | None = None) -> None:
    """Remove direct Modal snapshot entries for a task, including legacy keys."""
    snapshots = _load_snapshots()
    updated = False

    for key in (_direct_snapshot_key(task_id), task_id):
        value = snapshots.get(key)
        if value is None:
            continue
        if snapshot_id is None or value == snapshot_id:
            snapshots.pop(key, None)
            updated = True

    if updated:
        _save_snapshots(snapshots)


class ModalEnvironment(BaseEnvironment):
    """Modal cloud execution via mini-swe-agent.

    Wraps SwerexModalEnvironment and adds sudo -S support, configurable
    resources (CPU, memory, disk), and optional filesystem persistence
    via Modal's snapshot_filesystem() API.
    """

    _patches_applied = False

    def __init__(
        self,
        image: str,
        cwd: str = "/root",
        timeout: int = 60,
        modal_sandbox_kwargs: Optional[Dict[str, Any]] = None,
        persistent_filesystem: bool = True,
        task_id: str = "default",
    ):
        super().__init__(cwd=cwd, timeout=timeout)

        if not ModalEnvironment._patches_applied:
            try:
                from environments.patches import apply_patches
                apply_patches()
            except ImportError:
                pass
            ModalEnvironment._patches_applied = True

        self._persistent = persistent_filesystem
        self._task_id = task_id
        self._base_image = image

        sandbox_kwargs = dict(modal_sandbox_kwargs or {})

        # If persistent, try to restore from a previous snapshot
        restored_snapshot_id = None
        restored_from_legacy_key = False
        if self._persistent:
            restored_snapshot_id, restored_from_legacy_key = _get_snapshot_restore_candidate(self._task_id)
            if restored_snapshot_id:
                logger.info("Modal: restoring from snapshot %s", restored_snapshot_id[:20])

        effective_image = restored_snapshot_id or image

        from minisweagent.environments.extra.swerex_modal import SwerexModalEnvironment
        try:
            self._inner = SwerexModalEnvironment(
                image=effective_image,
                cwd=cwd,
                timeout=timeout,
                startup_timeout=180.0,
                runtime_timeout=3600.0,
                modal_sandbox_kwargs=sandbox_kwargs,
                install_pipx=True,  # Required: installs pipx + swe-rex runtime (swerex-remote)
            )
        except Exception as exc:
            if not restored_snapshot_id:
                raise

            logger.warning(
                "Modal: failed to restore snapshot %s, retrying with base image: %s",
                restored_snapshot_id[:20],
                exc,
            )
            _delete_direct_snapshot(self._task_id, restored_snapshot_id)
            self._inner = SwerexModalEnvironment(
                image=image,
                cwd=cwd,
                timeout=timeout,
                startup_timeout=180.0,
                runtime_timeout=3600.0,
                modal_sandbox_kwargs=sandbox_kwargs,
                install_pipx=True,
            )
        else:
            if restored_snapshot_id and restored_from_legacy_key:
                _store_direct_snapshot(self._task_id, restored_snapshot_id)
                logger.info("Modal: migrated legacy snapshot entry for task %s", self._task_id)

    def execute(self, command: str, cwd: str = "", *,
                timeout: int | None = None,
                stdin_data: str | None = None) -> dict:
        if stdin_data is not None:
            marker = f"HERMES_EOF_{uuid.uuid4().hex[:8]}"
            while marker in stdin_data:
                marker = f"HERMES_EOF_{uuid.uuid4().hex[:8]}"
            command = f"{command} << '{marker}'\n{stdin_data}\n{marker}"

        exec_command, sudo_stdin = self._prepare_command(command)

        # Modal sandboxes execute commands via the Modal SDK and cannot pipe
        # subprocess stdin directly the way a local Popen can.  When a sudo
        # password is present, use a shell-level pipe from printf so that the
        # password feeds sudo -S without appearing as an echo argument embedded
        # in the shell string.  The password is still visible in the remote
        # sandbox's command line, but it is not exposed on the user's local
        # machine — which is the primary threat being mitigated.
        if sudo_stdin is not None:
            import shlex
            exec_command = (
                f"printf '%s\\n' {shlex.quote(sudo_stdin.rstrip())} | {exec_command}"
            )

        # Run in a background thread so we can poll for interrupts
        result_holder = {"value": None, "error": None}

        def _run():
            try:
                result_holder["value"] = self._inner.execute(exec_command, cwd=cwd, timeout=timeout)
            except Exception as e:
                result_holder["error"] = e

        t = threading.Thread(target=_run, daemon=True)
        t.start()
        while t.is_alive():
            t.join(timeout=0.2)
            if is_interrupted():
                try:
                    self._inner.stop()
                except Exception:
                    pass
                return {
                    "output": "[Command interrupted - Modal sandbox terminated]",
                    "returncode": 130,
                }

        if result_holder["error"]:
            return {"output": f"Modal execution error: {result_holder['error']}", "returncode": 1}
        return result_holder["value"]

    def cleanup(self):
        """Snapshot the filesystem (if persistent) then stop the sandbox."""
        # Check if _inner was ever set (init may have failed)
        if not hasattr(self, '_inner') or self._inner is None:
            return

        if self._persistent:
            try:
                sandbox = getattr(self._inner, 'deployment', None)
                sandbox = getattr(sandbox, '_sandbox', None) if sandbox else None
                if sandbox:
                    import asyncio
                    async def _snapshot():
                        img = await sandbox.snapshot_filesystem.aio()
                        return img.object_id
                    try:
                        snapshot_id = asyncio.run(_snapshot())
                    except RuntimeError:
                        import concurrent.futures
                        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                            snapshot_id = pool.submit(
                                asyncio.run, _snapshot()
                            ).result(timeout=60)

                    _store_direct_snapshot(self._task_id, snapshot_id)
                    logger.info("Modal: saved filesystem snapshot %s for task %s",
                                snapshot_id[:20], self._task_id)
            except Exception as e:
                logger.warning("Modal: filesystem snapshot failed: %s", e)

        if hasattr(self._inner, 'stop'):
            self._inner.stop()
