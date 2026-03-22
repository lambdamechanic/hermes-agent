"""Shared slash command helpers for skills and built-in prompt-style modes.

Shared between CLI (cli.py) and gateway (gateway/run.py) so both surfaces
can invoke skills via /skill-name commands and prompt-only built-ins like
/plan.
"""

import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

_skill_commands: Dict[str, Dict[str, Any]] = {}
_PLAN_SLUG_RE = re.compile(r"[^a-z0-9]+")


def build_plan_path(
    user_instruction: str = "",
    *,
    now: datetime | None = None,
) -> Path:
    """Return the default workspace-relative markdown path for a /plan invocation.

    Relative paths are intentional: file tools are task/backend-aware and resolve
    them against the active working directory for local, docker, ssh, modal,
    daytona, and similar terminal backends. That keeps the plan with the active
    workspace instead of the Hermes host's global home directory.
    """
    slug_source = (user_instruction or "").strip().splitlines()[0] if user_instruction else ""
    slug = _PLAN_SLUG_RE.sub("-", slug_source.lower()).strip("-")
    if slug:
        slug = "-".join(part for part in slug.split("-")[:8] if part)[:48].strip("-")
    slug = slug or "conversation-plan"
    timestamp = (now or datetime.now()).strftime("%Y-%m-%d_%H%M%S")
    return Path(".hermes") / "plans" / f"{timestamp}-{slug}.md"


def _load_skill_payload(skill_identifier: str, task_id: str | None = None) -> tuple[dict[str, Any], Path | None, str] | None:
    """Load a skill by name/path and return (loaded_payload, skill_dir, display_name)."""
    raw_identifier = (skill_identifier or "").strip()
    if not raw_identifier:
        return None

    try:
        from tools.skills_tool import SKILLS_DIR, skill_view

        identifier_path = Path(raw_identifier).expanduser()
        if identifier_path.is_absolute():
            try:
                normalized = str(identifier_path.resolve().relative_to(SKILLS_DIR.resolve()))
            except Exception:
                normalized = raw_identifier
        else:
            normalized = raw_identifier.lstrip("/")

        loaded_skill = json.loads(skill_view(normalized, task_id=task_id))
    except Exception:
        return None

    if not loaded_skill.get("success"):
        return None

    skill_name = str(loaded_skill.get("name") or normalized)
    skill_dir = None
    skill_dir_value = str(loaded_skill.get("skill_dir") or "")
    if skill_dir_value:
        try:
            skill_dir = Path(skill_dir_value)
        except Exception:
            skill_dir = None
    elif loaded_skill.get("path"):
        skill_path = str(loaded_skill.get("path") or "")
        try:
            skill_dir = SKILLS_DIR / Path(skill_path).parent
        except Exception:
            skill_dir = None

    return loaded_skill, skill_dir, skill_name


def _build_skill_message(
    loaded_skill: dict[str, Any],
    skill_dir: Path | None,
    activation_note: str,
    user_instruction: str = "",
    runtime_note: str = "",
) -> str:
    """Format a loaded skill into a user/system message payload."""
    content = str(loaded_skill.get("content") or "")

    parts = [activation_note, "", content.strip()]

    if loaded_skill.get("setup_skipped"):
        parts.extend(
            [
                "",
                "[Skill setup note: Required environment setup was skipped. Continue loading the skill and explain any reduced functionality if it matters.]",
            ]
        )
    elif loaded_skill.get("gateway_setup_hint"):
        parts.extend(
            [
                "",
                f"[Skill setup note: {loaded_skill['gateway_setup_hint']}]",
            ]
        )
    elif loaded_skill.get("setup_needed") and loaded_skill.get("setup_note"):
        parts.extend(
            [
                "",
                f"[Skill setup note: {loaded_skill['setup_note']}]",
            ]
        )

    supporting = []
    linked_files = loaded_skill.get("linked_files") or {}
    for entries in linked_files.values():
        if isinstance(entries, list):
            supporting.extend(entries)

    if not supporting and skill_dir:
        for subdir in ("references", "templates", "scripts", "assets"):
            subdir_path = skill_dir / subdir
            if subdir_path.exists():
                for f in sorted(subdir_path.rglob("*")):
                    if f.is_file():
                        rel = str(f.relative_to(skill_dir))
                        supporting.append(rel)

    if supporting and skill_dir:
        skill_view_target = str(loaded_skill.get("skill_dir") or skill_dir)
        parts.append("")
        parts.append("[This skill has supporting files you can load with the skill_view tool:]")
        for sf in supporting:
            parts.append(f"- {sf}")
        parts.append(
            f'\nTo view any of these, use: skill_view(name="{skill_view_target}", file_path="<path>")'
        )

    if user_instruction:
        parts.append("")
        parts.append(f"The user has provided the following instruction alongside the skill invocation: {user_instruction}")

    if runtime_note:
        parts.append("")
        parts.append(f"[Runtime note: {runtime_note}]")

    return "\n".join(parts)


def scan_skill_commands() -> Dict[str, Dict[str, Any]]:
    """Scan ~/.hermes/skills/ and return a mapping of /command -> skill info.

    Returns:
        Dict mapping "/skill-name" to {name, description, skill_md_path, skill_dir}.
    """
    global _skill_commands
    _skill_commands = {}
    try:
        from tools.skills_tool import (
            _get_disabled_skill_names,
            _get_skill_catalog,
        )

        disabled = _get_disabled_skill_names()
        catalog = _get_skill_catalog()
        for entry in catalog.visible_entries:
            name = entry.name
            if name in disabled:
                continue
            description = entry.description[:80] if entry.description else ""
            cmd_name = name.lower().replace(" ", "-").replace("_", "-")
            cmd_key = f"/{cmd_name}"
            if cmd_key in _skill_commands:
                continue
            _skill_commands[cmd_key] = {
                "name": name,
                "description": description or f"Invoke the {name} skill",
                "skill_md_path": str(entry.skill_md),
                "skill_dir": str(entry.skill_dir) if entry.skill_dir else "",
            }
    except Exception:
        pass
    return _skill_commands


def get_skill_commands() -> Dict[str, Dict[str, Any]]:
    """Return the current skill commands mapping (scan first if empty)."""
    if not _skill_commands:
        scan_skill_commands()
    return _skill_commands


def build_skill_invocation_message(
    cmd_key: str,
    user_instruction: str = "",
    task_id: str | None = None,
    runtime_note: str = "",
) -> Optional[str]:
    """Build the user message content for a skill slash command invocation.

    Args:
        cmd_key: The command key including leading slash (e.g., "/gif-search").
        user_instruction: Optional text the user typed after the command.

    Returns:
        The formatted message string, or None if the skill wasn't found.
    """
    commands = get_skill_commands()
    skill_info = commands.get(cmd_key)
    if not skill_info:
        return None

    loaded = _load_skill_payload(skill_info["skill_dir"], task_id=task_id)
    if not loaded:
        return f"[Failed to load skill: {skill_info['name']}]"

    loaded_skill, skill_dir, skill_name = loaded
    activation_note = (
        f'[SYSTEM: The user has invoked the "{skill_name}" skill, indicating they want '
        "you to follow its instructions. The full skill content is loaded below.]"
    )
    return _build_skill_message(
        loaded_skill,
        skill_dir,
        activation_note,
        user_instruction=user_instruction,
        runtime_note=runtime_note,
    )


def build_preloaded_skills_prompt(
    skill_identifiers: list[str],
    task_id: str | None = None,
) -> tuple[str, list[str], list[str]]:
    """Load one or more skills for session-wide CLI preloading.

    Returns (prompt_text, loaded_skill_names, missing_identifiers).
    """
    prompt_parts: list[str] = []
    loaded_names: list[str] = []
    missing: list[str] = []

    seen: set[str] = set()
    for raw_identifier in skill_identifiers:
        identifier = (raw_identifier or "").strip()
        if not identifier or identifier in seen:
            continue
        seen.add(identifier)

        loaded = _load_skill_payload(identifier, task_id=task_id)
        if not loaded:
            missing.append(identifier)
            continue

        loaded_skill, skill_dir, skill_name = loaded
        activation_note = (
            f'[SYSTEM: The user launched this CLI session with the "{skill_name}" skill '
            "preloaded. Treat its instructions as active guidance for the duration of this "
            "session unless the user overrides them.]"
        )
        prompt_parts.append(
            _build_skill_message(
                loaded_skill,
                skill_dir,
                activation_note,
            )
        )
        loaded_names.append(skill_name)

    return "\n\n".join(prompt_parts), loaded_names, missing
