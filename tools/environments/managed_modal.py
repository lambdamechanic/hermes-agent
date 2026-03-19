"""Managed Modal environment backed by tool-gateway."""

from __future__ import annotations

import json
import logging
import requests
import uuid
from typing import Any, Dict, Optional

from tools.environments.base import BaseEnvironment
from tools.managed_tool_gateway import resolve_managed_tool_gateway

logger = logging.getLogger(__name__)


class ManagedModalEnvironment(BaseEnvironment):
    """Gateway-owned Modal sandbox with Hermes-compatible execute/cleanup."""

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

        gateway = resolve_managed_tool_gateway("modal")
        if gateway is None:
            raise ValueError("Managed Modal requires a configured tool gateway and Nous user token")

        self._gateway_origin = gateway.gateway_origin.rstrip("/")
        self._nous_user_token = gateway.nous_user_token
        self._task_id = task_id
        self._persistent = persistent_filesystem
        self._image = image
        self._sandbox_kwargs = dict(modal_sandbox_kwargs or {})
        self._sandbox_id = self._create_sandbox()

    def execute(self, command: str, cwd: str = "", *,
                timeout: int | None = None,
                stdin_data: str | None = None) -> dict:
        exec_command, sudo_stdin = self._prepare_command(command)

        # When a sudo password is present, inject it via a shell-level pipe
        # (same approach as the direct ModalEnvironment) since the gateway
        # cannot pipe subprocess stdin directly.
        if sudo_stdin is not None:
            import shlex
            exec_command = (
                f"printf '%s\\n' {shlex.quote(sudo_stdin.rstrip())} | {exec_command}"
            )

        exec_cwd = cwd or self.cwd
        payload: Dict[str, Any] = {
            "command": exec_command,
            "cwd": exec_cwd,
            "timeoutMs": int((timeout or self.timeout) * 1000),
        }
        if stdin_data is not None:
            payload["stdinData"] = stdin_data

        response = self._request(
            "POST",
            f"/v1/sandboxes/{self._sandbox_id}/exec",
            json=payload,
        )
        if response.status_code >= 400:
            return {
                "output": self._format_error("Managed Modal exec failed", response),
                "returncode": 1 if response.status_code != 504 else 124,
            }

        body = response.json()
        return {
            "output": body.get("output", ""),
            "returncode": body.get("returncode", 1),
        }

    def cleanup(self):
        if not getattr(self, "_sandbox_id", None):
            return

        try:
            self._request(
                "POST",
                f"/v1/sandboxes/{self._sandbox_id}/terminate",
                json={
                    "snapshotBeforeTerminate": self._persistent,
                },
                timeout=60,
            )
        except Exception as exc:
            logger.warning("Managed Modal cleanup failed: %s", exc)
        finally:
            self._sandbox_id = None

    def _create_sandbox(self) -> str:
        cpu = self._coerce_number(self._sandbox_kwargs.get("cpu"), 1)
        memory = self._coerce_number(
            self._sandbox_kwargs.get("memoryMiB", self._sandbox_kwargs.get("memory")),
            5120,
        )
        disk = self._coerce_number(
            self._sandbox_kwargs.get("ephemeral_disk", self._sandbox_kwargs.get("diskMiB")),
            None,
        )

        create_payload = {
            "image": self._image,
            "cwd": self.cwd,
            "cpu": cpu,
            "memoryMiB": memory,
            "timeoutMs": 3_600_000,
            "idleTimeoutMs": max(300_000, int(self.timeout * 1000)),
            "persistentFilesystem": self._persistent,
            "logicalKey": self._task_id,
        }
        if disk is not None:
            create_payload["diskMiB"] = disk

        response = self._request(
            "POST",
            "/v1/sandboxes",
            json=create_payload,
            timeout=60,
        )
        if response.status_code >= 400:
            raise RuntimeError(self._format_error("Managed Modal create failed", response))

        body = response.json()
        sandbox_id = body.get("id")
        if not isinstance(sandbox_id, str) or not sandbox_id:
            raise RuntimeError("Managed Modal create did not return a sandbox id")
        return sandbox_id

    def _request(self, method: str, path: str, *,
                 json: Dict[str, Any] | None = None,
                 timeout: int = 30) -> requests.Response:
        return requests.request(
            method,
            f"{self._gateway_origin}{path}",
            headers={
                "Authorization": f"Bearer {self._nous_user_token}",
                "Content-Type": "application/json",
            },
            json=json,
            timeout=timeout,
        )

    @staticmethod
    def _coerce_number(value: Any, default: float) -> float:
        try:
            if value is None:
                return default
            return float(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _format_error(prefix: str, response: requests.Response) -> str:
        try:
            payload = response.json()
            if isinstance(payload, dict):
                message = payload.get("error") or payload.get("message") or payload.get("code")
                if isinstance(message, str) and message:
                    return f"{prefix}: {message}"
                return f"{prefix}: {json.dumps(payload, ensure_ascii=False)}"
        except Exception:
            pass

        text = response.text.strip()
        if text:
            return f"{prefix}: {text}"
        return f"{prefix}: HTTP {response.status_code}"
