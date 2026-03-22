#!/usr/bin/env python3
"""
Skills Tool Module

This module provides tools for listing and viewing skill documents.
Skills are organized as directories containing a SKILL.md file (the main instructions)
and optional supporting files like references, templates, and examples.

Inspired by Anthropic's Claude Skills system with progressive disclosure architecture:
- Metadata (name ≤64 chars, description ≤1024 chars) - shown in skills_list
- Full Instructions - loaded via skill_view when needed
- Linked Files (references, templates) - loaded on demand

Directory Structure:
    skills/
    ├── my-skill/
    │   ├── SKILL.md           # Main instructions (required)
    │   ├── references/        # Supporting documentation
    │   │   ├── api.md
    │   │   └── examples.md
    │   ├── templates/         # Templates for output
    │   │   └── template.md
    │   └── assets/            # Supplementary files (agentskills.io standard)
    └── category/              # Category folder for organization
        └── another-skill/
            └── SKILL.md

SKILL.md Format (YAML Frontmatter, agentskills.io compatible):
    ---
    name: skill-name              # Required, max 64 chars
    description: Brief description # Required, max 1024 chars
    version: 1.0.0                # Optional
    license: MIT                  # Optional (agentskills.io)
    platforms: [macos]            # Optional — restrict to specific OS platforms
                                  #   Valid: macos, linux, windows
                                  #   Omit to load on all platforms (default)
    prerequisites:                # Optional — legacy runtime requirements
      env_vars: [API_KEY]         #   Legacy env var names are normalized into
                                  #   required_environment_variables on load.
      commands: [curl, jq]        #   Command checks remain advisory only.
    compatibility: Requires X     # Optional (agentskills.io)
    metadata:                     # Optional, arbitrary key-value (agentskills.io)
      hermes:
        tags: [fine-tuning, llm]
        related_skills: [peft, lora]
    ---

    # Skill Title

    Full instructions and content here...

Available tools:
- skills_list: List skills with metadata (progressive disclosure tier 1)
- skill_view: Load full skill content (progressive disclosure tier 2-3)

Usage:
    from tools.skills_tool import skills_list, skill_view, check_skills_requirements

    # List all skills (returns metadata only - token efficient)
    result = skills_list()

    # View a skill's main content (loads full instructions)
    content = skill_view("axolotl")

    # View a reference file within a skill (loads linked file)
    content = skill_view("axolotl", "references/dataset-formats.md")
"""

import json
import logging
import os
import re
import sys
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, Any, List, Optional, Set, Tuple

import yaml
from hermes_cli.config import load_env, _ENV_VAR_NAME_RE
from tools.registry import registry

logger = logging.getLogger(__name__)


# All skills live in ~/.hermes/skills/ (seeded from bundled skills/ on install).
# This is the single source of truth -- agent edits, hub installs, and bundled
# skills all coexist here without polluting the git repo.
HERMES_HOME = Path(os.getenv("HERMES_HOME", Path.home() / ".hermes"))
SKILLS_DIR = HERMES_HOME / "skills"
_DEFAULT_HERMES_HOME = HERMES_HOME
_DEFAULT_SKILLS_DIR = SKILLS_DIR

# Anthropic-recommended limits for progressive disclosure efficiency
MAX_NAME_LENGTH = 64
MAX_DESCRIPTION_LENGTH = 1024

# Platform identifiers for the 'platforms' frontmatter field.
# Maps user-friendly names to sys.platform prefixes.
_PLATFORM_MAP = {
    "macos": "darwin",
    "linux": "linux",
    "windows": "win32",
}
_EXCLUDED_SKILL_DIRS = frozenset((".git", ".github", ".hub"))
_PROJECT_LOCAL_SKILL_DIRS = (".hermes/skills", ".agents/skills")
_REMOTE_ENV_BACKENDS = frozenset({"docker", "singularity", "modal", "ssh", "daytona"})
_secret_capture_callback = None


@dataclass(frozen=True)
class SkillSearchRoot:
    path: Path
    scope: str


@dataclass(frozen=True)
class SkillCatalogEntry:
    name: str
    description: str
    skill_md: Path
    skill_dir: Path | None
    root_path: Path
    scope: str
    category: str | None
    relative_path: str
    frontmatter: Dict[str, Any]
    conditions: Dict[str, List[str]]


@dataclass(frozen=True)
class SkillCatalog:
    roots: tuple[SkillSearchRoot, ...]
    entries: tuple[SkillCatalogEntry, ...]
    visible_entries: tuple[SkillCatalogEntry, ...]
    entries_by_name: Dict[str, SkillCatalogEntry]
    entries_by_skill_md: Dict[str, SkillCatalogEntry]
    category_dirs: Dict[str, Path]


class SkillReadinessStatus(str, Enum):
    AVAILABLE = "available"
    SETUP_NEEDED = "setup_needed"
    UNSUPPORTED = "unsupported"


def set_secret_capture_callback(callback) -> None:
    global _secret_capture_callback
    _secret_capture_callback = callback


def _normalize_project_key(path: Path | str) -> str:
    if not isinstance(path, (str, Path, os.PathLike)):
        return os.path.normcase(str(path))
    candidate = Path(path).expanduser()
    try:
        resolved = candidate.resolve()
    except Exception:
        resolved = candidate.absolute()
    return os.path.normcase(str(resolved))


def _find_git_root(start: Path) -> Optional[Path]:
    try:
        current = start.resolve()
    except Exception:
        current = start.absolute()
    for parent in [current, *current.parents]:
        if (parent / ".git").exists():
            return parent
    return None


def _skill_discovery_cwd(cwd: str | Path | None = None) -> Path:
    raw = cwd or os.getenv("TERMINAL_CWD") or os.getcwd()
    return Path(raw).expanduser()


def _current_skills_dir() -> Path:
    """Return the current user-skill root.

    Tests often monkeypatch ``SKILLS_DIR`` directly. Outside of that, respect a
    late-bound ``HERMES_HOME`` env var so prompt-building and discovery stay in
    sync with the active environment rather than the import-time default.
    """
    if not isinstance(SKILLS_DIR, (Path, str, os.PathLike)):
        return SKILLS_DIR
    if _normalize_project_key(SKILLS_DIR) != _normalize_project_key(_DEFAULT_SKILLS_DIR):
        return SKILLS_DIR
    return Path(os.getenv("HERMES_HOME", Path.home() / ".hermes")) / "skills"


def _project_local_enabled(config: Dict[str, Any] | None = None) -> bool:
    if config is None:
        try:
            from hermes_cli.config import load_config

            config = load_config()
        except Exception:
            config = {}

    skills_cfg = config.get("skills", {}) if isinstance(config, dict) else {}
    project_local = skills_cfg.get("project_local", False)
    if isinstance(project_local, dict):
        return bool(project_local.get("enabled", False))
    return bool(project_local)


def _discover_project_local_roots(
    cwd: str | Path | None = None,
) -> tuple[Path, List[SkillSearchRoot]]:
    cwd_path = _skill_discovery_cwd(cwd)
    repo_root = _find_git_root(cwd_path) or cwd_path
    try:
        repo_root = repo_root.resolve()
    except Exception:
        repo_root = repo_root.absolute()

    roots: List[SkillSearchRoot] = []
    for relative_dir in _PROJECT_LOCAL_SKILL_DIRS:
        candidate = repo_root / relative_dir
        if candidate.is_dir():
            roots.append(
                SkillSearchRoot(
                    path=candidate.resolve(),
                    scope="project_local",
                )
            )
    return repo_root, roots


def _skill_search_roots(
    *,
    cwd: str | Path | None = None,
    config: Dict[str, Any] | None = None,
) -> List[SkillSearchRoot]:
    roots: List[SkillSearchRoot] = []

    _, project_roots = _discover_project_local_roots(cwd)
    if project_roots and _project_local_enabled(config):
        roots.extend(project_roots)

    roots.append(SkillSearchRoot(path=_current_skills_dir(), scope="user"))
    return roots


def _path_within_root(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except Exception:
        return False


def _path_within_any_root(path: Path, roots: List[SkillSearchRoot]) -> bool:
    return any(_path_within_root(path, root.path) for root in roots)

def _skill_conditions_from_frontmatter(frontmatter: Dict[str, Any]) -> Dict[str, List[str]]:
    metadata = frontmatter.get("metadata")
    hermes = {}
    if isinstance(metadata, dict):
        hermes = metadata.get("hermes", {}) or {}
    if not isinstance(hermes, dict):
        hermes = {}

    def _normalize(value: Any) -> List[str]:
        if not value:
            return []
        if not isinstance(value, list):
            value = [value]
        return [str(item).strip() for item in value if str(item).strip()]

    return {
        "fallback_for_toolsets": _normalize(hermes.get("fallback_for_toolsets")),
        "requires_toolsets": _normalize(hermes.get("requires_toolsets")),
        "fallback_for_tools": _normalize(hermes.get("fallback_for_tools")),
        "requires_tools": _normalize(hermes.get("requires_tools")),
    }


def _skill_description_from_frontmatter(
    frontmatter: Dict[str, Any],
    body: str,
    *,
    max_length: int = MAX_DESCRIPTION_LENGTH,
) -> str:
    description = str(frontmatter.get("description", "") or "").strip()
    if not description:
        for line in body.strip().split("\n"):
            line = line.strip()
            if line and not line.startswith("#"):
                description = line
                break

    if len(description) > max_length:
        return description[: max_length - 3] + "..."
    return description


def _scan_skill_root(root: SkillSearchRoot) -> tuple[SkillCatalogEntry, ...]:
    entries: List[SkillCatalogEntry] = []
    if root.path.exists():
        for skill_md in root.path.rglob("SKILL.md"):
            if any(part in _EXCLUDED_SKILL_DIRS for part in skill_md.parts):
                continue

            skill_dir = skill_md.parent
            try:
                raw = skill_md.read_text(encoding="utf-8")[:4000]
                frontmatter, body = _parse_frontmatter(raw)
                if not skill_matches_platform(frontmatter):
                    continue

                name = str(frontmatter.get("name", skill_dir.name)).strip()[:MAX_NAME_LENGTH]
                if not name:
                    continue

                try:
                    relative_path = str(skill_md.relative_to(root.path))
                except Exception:
                    relative_path = str(skill_md)

                entries.append(
                    SkillCatalogEntry(
                        name=name,
                        description=_skill_description_from_frontmatter(frontmatter, body),
                        skill_md=skill_md,
                        skill_dir=skill_dir,
                        root_path=root.path,
                        scope=root.scope,
                        category=_get_category_from_path(skill_md, root.path),
                        relative_path=relative_path,
                        frontmatter=frontmatter,
                        conditions=_skill_conditions_from_frontmatter(frontmatter),
                    )
                )
            except (UnicodeDecodeError, PermissionError) as e:
                logger.debug("Failed to read skill file %s: %s", skill_md, e)
                continue
            except Exception as e:
                logger.debug(
                    "Skipping skill at %s: failed to parse: %s",
                    skill_md,
                    e,
                    exc_info=True,
                )
                continue

    return tuple(entries)


def _build_skill_catalog(roots: tuple[SkillSearchRoot, ...]) -> SkillCatalog:
    entries: List[SkillCatalogEntry] = []
    visible_entries: List[SkillCatalogEntry] = []
    entries_by_name: Dict[str, SkillCatalogEntry] = {}
    entries_by_skill_md: Dict[str, SkillCatalogEntry] = {}
    category_dirs: Dict[str, Path] = {}
    seen_names: Set[str] = set()

    for root in roots:
        root_entries = _scan_skill_root(root)
        entries.extend(root_entries)
        for entry in root_entries:
            entries_by_skill_md[_normalize_project_key(entry.skill_md)] = entry
            if entry.name in seen_names:
                continue
            seen_names.add(entry.name)
            visible_entries.append(entry)
            entries_by_name[entry.name] = entry
            if entry.category and entry.category not in category_dirs:
                category_dirs[entry.category] = entry.root_path / entry.category

    return SkillCatalog(
        roots=roots,
        entries=tuple(entries),
        visible_entries=tuple(visible_entries),
        entries_by_name=entries_by_name,
        entries_by_skill_md=entries_by_skill_md,
        category_dirs=category_dirs,
    )


def _get_skill_catalog(
    *,
    cwd: str | Path | None = None,
    config: Dict[str, Any] | None = None,
) -> SkillCatalog:
    roots = tuple(
        _skill_search_roots(
            cwd=cwd,
            config=config,
        )
    )
    return _build_skill_catalog(roots)


def skill_matches_platform(frontmatter: Dict[str, Any]) -> bool:
    """Check if a skill is compatible with the current OS platform.

    Skills declare platform requirements via a top-level ``platforms`` list
    in their YAML frontmatter::

        platforms: [macos]          # macOS only
        platforms: [macos, linux]   # macOS and Linux

    Valid values: ``macos``, ``linux``, ``windows``.

    If the field is absent or empty the skill is compatible with **all**
    platforms (backward-compatible default).
    """
    platforms = frontmatter.get("platforms")
    if not platforms:
        return True  # No restriction → loads everywhere
    if not isinstance(platforms, list):
        platforms = [platforms]
    current = sys.platform
    for p in platforms:
        mapped = _PLATFORM_MAP.get(str(p).lower().strip(), str(p).lower().strip())
        if current.startswith(mapped):
            return True
    return False


def _normalize_prerequisite_values(value: Any) -> List[str]:
    if not value:
        return []
    if isinstance(value, str):
        value = [value]
    return [str(item) for item in value if str(item).strip()]


def _collect_prerequisite_values(
    frontmatter: Dict[str, Any],
) -> Tuple[List[str], List[str]]:
    prereqs = frontmatter.get("prerequisites")
    if not prereqs or not isinstance(prereqs, dict):
        return [], []
    return (
        _normalize_prerequisite_values(prereqs.get("env_vars")),
        _normalize_prerequisite_values(prereqs.get("commands")),
    )


def _normalize_setup_metadata(frontmatter: Dict[str, Any]) -> Dict[str, Any]:
    setup = frontmatter.get("setup")
    if not isinstance(setup, dict):
        return {"help": None, "collect_secrets": []}

    help_text = setup.get("help")
    normalized_help = (
        str(help_text).strip()
        if isinstance(help_text, str) and help_text.strip()
        else None
    )

    collect_secrets_raw = setup.get("collect_secrets")
    if isinstance(collect_secrets_raw, dict):
        collect_secrets_raw = [collect_secrets_raw]
    if not isinstance(collect_secrets_raw, list):
        collect_secrets_raw = []

    collect_secrets: List[Dict[str, Any]] = []
    for item in collect_secrets_raw:
        if not isinstance(item, dict):
            continue

        env_var = str(item.get("env_var") or "").strip()
        if not env_var:
            continue

        prompt = str(item.get("prompt") or f"Enter value for {env_var}").strip()
        provider_url = str(item.get("provider_url") or item.get("url") or "").strip()

        entry: Dict[str, Any] = {
            "env_var": env_var,
            "prompt": prompt,
            "secret": bool(item.get("secret", True)),
        }
        if provider_url:
            entry["provider_url"] = provider_url
        collect_secrets.append(entry)

    return {
        "help": normalized_help,
        "collect_secrets": collect_secrets,
    }


def _get_required_environment_variables(
    frontmatter: Dict[str, Any],
    legacy_env_vars: List[str] | None = None,
) -> List[Dict[str, Any]]:
    setup = _normalize_setup_metadata(frontmatter)
    required_raw = frontmatter.get("required_environment_variables")
    if isinstance(required_raw, dict):
        required_raw = [required_raw]
    if not isinstance(required_raw, list):
        required_raw = []

    required: List[Dict[str, Any]] = []
    seen: set[str] = set()

    def _append_required(entry: Dict[str, Any]) -> None:
        env_name = str(entry.get("name") or entry.get("env_var") or "").strip()
        if not env_name or env_name in seen:
            return
        if not _ENV_VAR_NAME_RE.match(env_name):
            return

        normalized: Dict[str, Any] = {
            "name": env_name,
            "prompt": str(entry.get("prompt") or f"Enter value for {env_name}").strip(),
        }

        help_text = (
            entry.get("help")
            or entry.get("provider_url")
            or entry.get("url")
            or setup.get("help")
        )
        if isinstance(help_text, str) and help_text.strip():
            normalized["help"] = help_text.strip()

        required_for = entry.get("required_for")
        if isinstance(required_for, str) and required_for.strip():
            normalized["required_for"] = required_for.strip()

        seen.add(env_name)
        required.append(normalized)

    for item in required_raw:
        if isinstance(item, str):
            _append_required({"name": item})
            continue
        if isinstance(item, dict):
            _append_required(item)

    for item in setup["collect_secrets"]:
        _append_required(
            {
                "name": item.get("env_var"),
                "prompt": item.get("prompt"),
                "help": item.get("provider_url") or setup.get("help"),
            }
        )

    if legacy_env_vars is None:
        legacy_env_vars, _ = _collect_prerequisite_values(frontmatter)
    for env_var in legacy_env_vars:
        _append_required({"name": env_var})

    return required


def _capture_required_environment_variables(
    skill_name: str,
    missing_entries: List[Dict[str, Any]],
) -> Dict[str, Any]:
    if not missing_entries:
        return {
            "missing_names": [],
            "setup_skipped": False,
            "gateway_setup_hint": None,
        }

    missing_names = [entry["name"] for entry in missing_entries]
    if _is_gateway_surface():
        return {
            "missing_names": missing_names,
            "setup_skipped": False,
            "gateway_setup_hint": _gateway_setup_hint(),
        }

    if _secret_capture_callback is None:
        return {
            "missing_names": missing_names,
            "setup_skipped": False,
            "gateway_setup_hint": None,
        }

    setup_skipped = False
    remaining_names: List[str] = []

    for entry in missing_entries:
        metadata = {"skill_name": skill_name}
        if entry.get("help"):
            metadata["help"] = entry["help"]
        if entry.get("required_for"):
            metadata["required_for"] = entry["required_for"]

        try:
            callback_result = _secret_capture_callback(
                entry["name"],
                entry["prompt"],
                metadata,
            )
        except Exception:
            logger.warning(
                f"Secret capture callback failed for {entry['name']}", exc_info=True
            )
            callback_result = {
                "success": False,
                "stored_as": entry["name"],
                "validated": False,
                "skipped": True,
            }

        success = isinstance(callback_result, dict) and bool(
            callback_result.get("success")
        )
        skipped = isinstance(callback_result, dict) and bool(
            callback_result.get("skipped")
        )
        if success and not skipped:
            continue

        setup_skipped = True
        remaining_names.append(entry["name"])

    return {
        "missing_names": remaining_names,
        "setup_skipped": setup_skipped,
        "gateway_setup_hint": None,
    }


def _is_gateway_surface() -> bool:
    if os.getenv("HERMES_GATEWAY_SESSION"):
        return True
    return bool(os.getenv("HERMES_SESSION_PLATFORM"))


def _get_terminal_backend_name() -> str:
    return str(os.getenv("TERMINAL_ENV", "local")).strip().lower() or "local"


def _is_env_var_persisted(
    var_name: str, env_snapshot: Dict[str, str] | None = None
) -> bool:
    if env_snapshot is None:
        env_snapshot = load_env()
    if var_name in env_snapshot:
        return bool(env_snapshot.get(var_name))
    return bool(os.getenv(var_name))


def _remaining_required_environment_names(
    required_env_vars: List[Dict[str, Any]],
    capture_result: Dict[str, Any],
    *,
    env_snapshot: Dict[str, str] | None = None,
    backend: str | None = None,
) -> List[str]:
    if backend is None:
        backend = _get_terminal_backend_name()
    missing_names = set(capture_result["missing_names"])
    if backend in _REMOTE_ENV_BACKENDS:
        return [entry["name"] for entry in required_env_vars]

    if env_snapshot is None:
        env_snapshot = load_env()
    remaining = []
    for entry in required_env_vars:
        name = entry["name"]
        if name in missing_names or not _is_env_var_persisted(name, env_snapshot):
            remaining.append(name)
    return remaining


def _gateway_setup_hint() -> str:
    try:
        from gateway.platforms.base import GATEWAY_SECRET_CAPTURE_UNSUPPORTED_MESSAGE

        return GATEWAY_SECRET_CAPTURE_UNSUPPORTED_MESSAGE
    except Exception:
        return "Secure secret entry is not available. Load this skill in the local CLI to be prompted, or add the key to ~/.hermes/.env manually."


def _build_setup_note(
    readiness_status: SkillReadinessStatus,
    missing: List[str],
    setup_help: str | None = None,
) -> str | None:
    if readiness_status == SkillReadinessStatus.SETUP_NEEDED:
        missing_str = ", ".join(missing) if missing else "required prerequisites"
        note = f"Setup needed before using this skill: missing {missing_str}."
        if setup_help:
            return f"{note} {setup_help}"
        return note
    return None


def check_skills_requirements() -> bool:
    """Skills are always available -- the directory is created on first use if needed."""
    return True


def _parse_frontmatter(content: str) -> Tuple[Dict[str, Any], str]:
    """
    Parse YAML frontmatter from markdown content.

    Uses yaml.safe_load for full YAML support (nested metadata, lists, etc.)
    with a fallback to simple key:value splitting for robustness.

    Args:
        content: Full markdown file content

    Returns:
        Tuple of (frontmatter dict, remaining content)
    """
    frontmatter = {}
    body = content

    if content.startswith("---"):
        end_match = re.search(r"\n---\s*\n", content[3:])
        if end_match:
            yaml_content = content[3 : end_match.start() + 3]
            body = content[end_match.end() + 3 :]

            try:
                parsed = yaml.safe_load(yaml_content)
                if isinstance(parsed, dict):
                    frontmatter = parsed
                # yaml.safe_load returns None for empty frontmatter
            except yaml.YAMLError:
                # Fallback: simple key:value parsing for malformed YAML
                for line in yaml_content.strip().split("\n"):
                    if ":" in line:
                        key, value = line.split(":", 1)
                        frontmatter[key.strip()] = value.strip()

    return frontmatter, body


def _get_category_from_path(
    skill_path: Path,
    root_dir: Path | None = None,
) -> Optional[str]:
    """
    Extract category from skill path based on directory structure.

    For paths like: ~/.hermes/skills/mlops/axolotl/SKILL.md -> "mlops"
    """
    try:
        if root_dir is None:
            root_dir = _current_skills_dir()
        rel_path = skill_path.relative_to(root_dir)
        parts = rel_path.parts
        if len(parts) >= 3:
            return parts[0]
        return None
    except Exception:
        return None


def _estimate_tokens(content: str) -> int:
    """
    Rough token estimate (4 chars per token average).

    Args:
        content: Text content

    Returns:
        Estimated token count
    """
    return len(content) // 4


def _parse_tags(tags_value) -> List[str]:
    """
    Parse tags from frontmatter value.

    Handles:
    - Already-parsed list (from yaml.safe_load): [tag1, tag2]
    - String with brackets: "[tag1, tag2]"
    - Comma-separated string: "tag1, tag2"

    Args:
        tags_value: Raw tags value — may be a list or string

    Returns:
        List of tag strings
    """
    if not tags_value:
        return []

    # yaml.safe_load already returns a list for [tag1, tag2]
    if isinstance(tags_value, list):
        return [str(t).strip() for t in tags_value if t]

    # String fallback — handle bracket-wrapped or comma-separated
    tags_value = str(tags_value).strip()
    if tags_value.startswith("[") and tags_value.endswith("]"):
        tags_value = tags_value[1:-1]

    return [t.strip().strip("\"'") for t in tags_value.split(",") if t.strip()]



def _get_disabled_skill_names() -> Set[str]:
    """Load disabled skill names from config (once per call).

    Resolves platform from ``HERMES_PLATFORM`` env var, falls back to
    the global disabled list.
    """
    import os
    try:
        from hermes_cli.config import load_config
        config = load_config()
        skills_cfg = config.get("skills", {})
        resolved_platform = os.getenv("HERMES_PLATFORM")
        if resolved_platform:
            platform_disabled = skills_cfg.get("platform_disabled", {}).get(resolved_platform)
            if platform_disabled is not None:
                return set(platform_disabled)
        return set(skills_cfg.get("disabled", []))
    except Exception:
        return set()


def _is_skill_disabled(name: str, platform: str = None) -> bool:
    """Check if a skill is disabled in config."""
    import os
    try:
        from hermes_cli.config import load_config
        config = load_config()
        skills_cfg = config.get("skills", {})
        resolved_platform = platform or os.getenv("HERMES_PLATFORM")
        if resolved_platform:
            platform_disabled = skills_cfg.get("platform_disabled", {}).get(resolved_platform)
            if platform_disabled is not None:
                return name in platform_disabled
        return name in skills_cfg.get("disabled", [])
    except Exception:
        return False


def _find_all_skills(
    *,
    skip_disabled: bool = False,
) -> List[Dict[str, Any]]:
    """Recursively find all skills in ~/.hermes/skills/.

    Args:
        skip_disabled: If True, return ALL skills regardless of disabled
            state (used by ``hermes skills`` config UI). Default False
            filters out disabled skills.

    Returns:
        List of skill metadata dicts (name, description, category).
    """
    catalog = _get_skill_catalog()

    disabled = set() if skip_disabled else _get_disabled_skill_names()
    return [
        {
            "name": entry.name,
            "description": entry.description,
            "category": entry.category,
            "scope": entry.scope,
        }
        for entry in catalog.visible_entries
        if entry.name not in disabled
    ]


def _load_category_description(category_dir: Path) -> Optional[str]:
    """
    Load category description from DESCRIPTION.md if it exists.

    Args:
        category_dir: Path to the category directory

    Returns:
        Description string or None if not found
    """
    desc_file = category_dir / "DESCRIPTION.md"
    if not desc_file.exists():
        return None

    try:
        content = desc_file.read_text(encoding="utf-8")
        # Parse frontmatter if present
        frontmatter, body = _parse_frontmatter(content)

        # Prefer frontmatter description, fall back to first non-header line
        description = frontmatter.get("description", "")
        if not description:
            for line in body.strip().split("\n"):
                line = line.strip()
                if line and not line.startswith("#"):
                    description = line
                    break

        # Truncate to reasonable length
        if len(description) > MAX_DESCRIPTION_LENGTH:
            description = description[: MAX_DESCRIPTION_LENGTH - 3] + "..."

        return description if description else None
    except (UnicodeDecodeError, PermissionError) as e:
        logger.debug("Failed to read category description %s: %s", desc_file, e)
        return None
    except Exception as e:
        logger.warning(
            "Error parsing category description %s: %s", desc_file, e, exc_info=True
        )
        return None


def skills_categories(verbose: bool = False, task_id: str = None) -> str:
    """
    List available skill categories with descriptions (progressive disclosure tier 0).

    Returns category names and descriptions for efficient discovery before drilling down.
    Categories can have a DESCRIPTION.md file with a description frontmatter field
    or first paragraph to explain what skills are in that category.

    Args:
        verbose: If True, include skill counts per category (default: False, but currently always included)
        task_id: Optional task identifier used to probe the active backend

    Returns:
        JSON string with list of categories and their descriptions
    """
    try:
        catalog = _get_skill_catalog()
        if not any(root.path.exists() for root in catalog.roots):
            return json.dumps(
                {
                    "success": True,
                    "categories": [],
                    "message": "No skills directory found.",
                },
                ensure_ascii=False,
            )

        category_dirs = {}
        category_counts: Dict[str, int] = {}
        for entry in catalog.visible_entries:
            if entry.category:
                category_counts[entry.category] = category_counts.get(entry.category, 0) + 1
                category_dirs.setdefault(
                    entry.category,
                    catalog.category_dirs.get(entry.category, entry.root_path / entry.category),
                )

        categories = []
        for name in sorted(category_dirs.keys()):
            category_dir = category_dirs[name]
            description = _load_category_description(category_dir)

            cat_entry = {"name": name, "skill_count": category_counts[name]}
            if description:
                cat_entry["description"] = description
            categories.append(cat_entry)

        return json.dumps(
            {
                "success": True,
                "categories": categories,
                "hint": "If a category is relevant to your task, use skills_list with that category to see available skills",
            },
            ensure_ascii=False,
        )

    except Exception as e:
        return json.dumps({"success": False, "error": str(e)}, ensure_ascii=False)


def skills_list(category: str = None, task_id: str = None) -> str:
    """
    List all available skills (progressive disclosure tier 1 - minimal metadata).

    Returns only name + description to minimize token usage. Use skill_view() to
    load full content, tags, related files, etc.

    Args:
        category: Optional category filter (e.g., "mlops")
        task_id: Optional task identifier used to probe the active backend

    Returns:
        JSON string with minimal skill info: name, description, category
    """
    try:
        catalog = _get_skill_catalog()
        if not any(root.path.exists() for root in catalog.roots):
            _current_skills_dir().mkdir(parents=True, exist_ok=True)
            return json.dumps(
                {
                    "success": True,
                    "skills": [],
                    "categories": [],
                    "message": "No skills found. Skills directory created at ~/.hermes/skills/",
                },
                ensure_ascii=False,
            )

        # Find all skills
        all_skills = _find_all_skills()

        if not all_skills:
            return json.dumps(
                {
                    "success": True,
                    "skills": [],
                    "categories": [],
                    "message": "No skills found in skills/ directory.",
                },
                ensure_ascii=False,
            )

        # Filter by category if specified
        if category:
            all_skills = [s for s in all_skills if s.get("category") == category]

        # Sort by category then name
        all_skills.sort(key=lambda s: (s.get("category") or "", s["name"]))

        # Extract unique categories
        categories = sorted(
            set(s.get("category") for s in all_skills if s.get("category"))
        )

        return json.dumps(
            {
                "success": True,
                "skills": all_skills,
                "categories": categories,
                "count": len(all_skills),
                "hint": "Use skill_view(name) to see full content, tags, and linked files",
            },
            ensure_ascii=False,
        )

    except Exception as e:
        return json.dumps({"success": False, "error": str(e)}, ensure_ascii=False)


def skill_view(name: str, file_path: str = None, task_id: str = None) -> str:
    """
    View the content of a skill or a specific file within a skill directory.

    Args:
        name: Name or path of the skill (e.g., "axolotl" or "03-fine-tuning/axolotl")
        file_path: Optional path to a specific file within the skill (e.g., "references/api.md")
        task_id: Optional task identifier used to probe the active backend

    Returns:
        JSON string with skill content or error message
    """
    try:
        catalog = _get_skill_catalog()
        allowed_roots = list(catalog.roots)
        _, project_roots = _discover_project_local_roots()
        all_roots = [*project_roots, SkillSearchRoot(path=_current_skills_dir(), scope="user")]

        if not any(root.path.exists() for root in all_roots):
            return json.dumps(
                {
                    "success": False,
                    "error": "Skills directory does not exist yet. It will be created on first install.",
                },
                ensure_ascii=False,
            )

        skill_dir = None
        skill_md = None
        catalog_entry: SkillCatalogEntry | None = None

        identifier_path = Path(name).expanduser()
        search_by_name = not identifier_path.is_absolute()
        direct_candidates: List[Path] = []
        if identifier_path.is_absolute():
            direct_candidates.append(identifier_path)
        else:
            direct_candidates.extend(root.path / name for root in allowed_roots)

        for direct_path in direct_candidates:
            if direct_path.is_dir() and (direct_path / "SKILL.md").exists():
                skill_dir = direct_path
                skill_md = direct_path / "SKILL.md"
                break
            if direct_path.with_suffix(".md").exists():
                skill_md = direct_path.with_suffix(".md")
                break

        if skill_md and project_roots and _path_within_any_root(skill_md, project_roots):
            if not _path_within_any_root(skill_md, allowed_roots):
                return json.dumps(
                    {
                        "success": False,
                        "error": (
                            "Project-local skills are disabled. "
                            "Set skills.project_local: true in config.yaml to allow them."
                        ),
                    },
                    ensure_ascii=False,
                )

        if skill_md:
            catalog_entry = catalog.entries_by_skill_md.get(
                _normalize_project_key(skill_md)
            )
            if catalog_entry and catalog_entry.skill_dir is not None:
                skill_dir = catalog_entry.skill_dir

        # Search by directory name within approved roots only
        if not skill_md and search_by_name:
            catalog_entry = next(
                (
                    entry
                    for entry in catalog.visible_entries
                    if entry.skill_dir is not None and entry.skill_dir.name == name
                ),
                None,
            )
            if catalog_entry is not None:
                skill_dir = catalog_entry.skill_dir
                skill_md = catalog_entry.skill_md

        # Legacy: flat .md files within approved roots only
        if not skill_md and search_by_name:
            for root in allowed_roots:
                if not root.path.exists():
                    continue
                for found_md in root.path.rglob(f"{name}.md"):
                    if found_md.name != "SKILL.md":
                        skill_md = found_md
                        break
                if skill_md:
                    break

        if not skill_md or not skill_md.exists():
            available = [s["name"] for s in _find_all_skills()[:20]]
            return json.dumps(
                {
                    "success": False,
                    "error": f"Skill '{name}' not found.",
                    "available_skills": available,
                    "hint": "Use skills_list to see all available skills",
                },
                ensure_ascii=False,
            )

        # Read the file once — reused for platform check and main content below
        try:
            content = skill_md.read_text(encoding="utf-8")
        except Exception as e:
            return json.dumps(
                {
                    "success": False,
                    "error": f"Failed to read skill '{name}': {e}",
                },
                ensure_ascii=False,
            )

        # Security: warn if skill is loaded from outside the trusted skills directory
        _outside_skills_dir = not _path_within_any_root(skill_md, allowed_roots)

        # Security: detect common prompt injection patterns
        _INJECTION_PATTERNS = [
            "ignore previous instructions",
            "ignore all previous",
            "you are now",
            "disregard your",
            "forget your instructions",
            "new instructions:",
            "system prompt:",
            "<system>",
            "]]>",
        ]
        _content_lower = content.lower()
        _injection_detected = any(p in _content_lower for p in _INJECTION_PATTERNS)

        if _outside_skills_dir or _injection_detected:
            _warnings = []
            if _outside_skills_dir:
                _warnings.append(
                    "skill file is outside the approved skill roots "
                    f"(~/.hermes/skills or approved project-local roots): {skill_md}"
                )
            if _injection_detected:
                _warnings.append("skill content contains patterns that may indicate prompt injection")
            import logging as _logging
            _logging.getLogger(__name__).warning("Skill security warning for '%s': %s", name, "; ".join(_warnings))

        parsed_frontmatter: Dict[str, Any] = {}
        try:
            parsed_frontmatter, _ = _parse_frontmatter(content)
        except Exception:
            parsed_frontmatter = {}

        if not skill_matches_platform(parsed_frontmatter):
            return json.dumps(
                {
                    "success": False,
                    "error": f"Skill '{name}' is not supported on this platform.",
                    "readiness_status": SkillReadinessStatus.UNSUPPORTED.value,
                },
                ensure_ascii=False,
            )

        # Check if the skill is disabled by the user
        resolved_name = parsed_frontmatter.get("name", skill_md.parent.name)
        if _is_skill_disabled(resolved_name):
            return json.dumps(
                {
                    "success": False,
                    "error": (
                        f"Skill '{resolved_name}' is disabled. "
                        "Enable it with `hermes skills` or inspect the files directly on disk."
                    ),
                },
                ensure_ascii=False,
            )

        # If a specific file path is requested, read that instead
        if file_path and skill_dir:
            # Security: Prevent path traversal attacks
            normalized_path = Path(file_path)
            if ".." in normalized_path.parts:
                return json.dumps(
                    {
                        "success": False,
                        "error": "Path traversal ('..') is not allowed.",
                        "hint": "Use a relative path within the skill directory",
                    },
                    ensure_ascii=False,
                )

            target_file = skill_dir / file_path

            # Security: Verify resolved path is still within skill directory
            try:
                resolved = target_file.resolve()
                skill_dir_resolved = skill_dir.resolve()
                if not resolved.is_relative_to(skill_dir_resolved):
                    return json.dumps(
                        {
                            "success": False,
                            "error": "Path escapes skill directory boundary.",
                            "hint": "Use a relative path within the skill directory",
                        },
                        ensure_ascii=False,
                    )
            except (OSError, ValueError):
                return json.dumps(
                    {
                        "success": False,
                        "error": f"Invalid file path: '{file_path}'",
                        "hint": "Use a valid relative path within the skill directory",
                    },
                    ensure_ascii=False,
                )
            if not target_file.exists():
                # List available files in the skill directory, organized by type
                available_files = {
                    "references": [],
                    "templates": [],
                    "assets": [],
                    "scripts": [],
                    "other": [],
                }

                # Scan for all readable files
                for f in skill_dir.rglob("*"):
                    if f.is_file() and f.name != "SKILL.md":
                        rel = str(f.relative_to(skill_dir))
                        if rel.startswith("references/"):
                            available_files["references"].append(rel)
                        elif rel.startswith("templates/"):
                            available_files["templates"].append(rel)
                        elif rel.startswith("assets/"):
                            available_files["assets"].append(rel)
                        elif rel.startswith("scripts/"):
                            available_files["scripts"].append(rel)
                        elif f.suffix in [
                            ".md",
                            ".py",
                            ".yaml",
                            ".yml",
                            ".json",
                            ".tex",
                            ".sh",
                        ]:
                            available_files["other"].append(rel)

                # Remove empty categories
                available_files = {k: v for k, v in available_files.items() if v}

                return json.dumps(
                    {
                        "success": False,
                        "error": f"File '{file_path}' not found in skill '{name}'.",
                        "available_files": available_files,
                        "hint": "Use one of the available file paths listed above",
                    },
                    ensure_ascii=False,
                )

            # Read the file content
            try:
                content = target_file.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                # Binary file - return info about it instead
                return json.dumps(
                    {
                        "success": True,
                        "name": name,
                        "file": file_path,
                        "content": f"[Binary file: {target_file.name}, size: {target_file.stat().st_size} bytes]",
                        "is_binary": True,
                    },
                    ensure_ascii=False,
                )

            return json.dumps(
                {
                    "success": True,
                    "name": name,
                    "file": file_path,
                    "content": content,
                    "file_type": target_file.suffix,
                },
                ensure_ascii=False,
            )

        # Reuse the parse from the platform check above
        frontmatter = parsed_frontmatter

        # Get reference, template, asset, and script files if this is a directory-based skill
        reference_files = []
        template_files = []
        asset_files = []
        script_files = []

        if skill_dir:
            references_dir = skill_dir / "references"
            if references_dir.exists():
                reference_files = [
                    str(f.relative_to(skill_dir)) for f in references_dir.glob("*.md")
                ]

            templates_dir = skill_dir / "templates"
            if templates_dir.exists():
                for ext in [
                    "*.md",
                    "*.py",
                    "*.yaml",
                    "*.yml",
                    "*.json",
                    "*.tex",
                    "*.sh",
                ]:
                    template_files.extend(
                        [
                            str(f.relative_to(skill_dir))
                            for f in templates_dir.rglob(ext)
                        ]
                    )

            # assets/ — agentskills.io standard directory for supplementary files
            assets_dir = skill_dir / "assets"
            if assets_dir.exists():
                for f in assets_dir.rglob("*"):
                    if f.is_file():
                        asset_files.append(str(f.relative_to(skill_dir)))

            scripts_dir = skill_dir / "scripts"
            if scripts_dir.exists():
                for ext in ["*.py", "*.sh", "*.bash", "*.js", "*.ts", "*.rb"]:
                    script_files.extend(
                        [str(f.relative_to(skill_dir)) for f in scripts_dir.glob(ext)]
                    )

        # Read tags/related_skills with backward compat:
        # Check metadata.hermes.* first (agentskills.io convention), fall back to top-level
        hermes_meta = {}
        metadata = frontmatter.get("metadata")
        if isinstance(metadata, dict):
            hermes_meta = metadata.get("hermes", {}) or {}

        tags = _parse_tags(hermes_meta.get("tags") or frontmatter.get("tags", ""))
        related_skills = _parse_tags(
            hermes_meta.get("related_skills") or frontmatter.get("related_skills", "")
        )

        # Build linked files structure for clear discovery
        linked_files = {}
        if reference_files:
            linked_files["references"] = reference_files
        if template_files:
            linked_files["templates"] = template_files
        if asset_files:
            linked_files["assets"] = asset_files
        if script_files:
            linked_files["scripts"] = script_files

        root_for_skill = None
        if catalog_entry is not None:
            root_for_skill = SkillSearchRoot(
                path=catalog_entry.root_path,
                scope=catalog_entry.scope,
            )
        if root_for_skill is None:
            root_for_skill = next(
                (root for root in allowed_roots if _path_within_root(skill_md, root.path)),
                None,
            )
        try:
            rel_path = (
                str(skill_md.relative_to(root_for_skill.path))
                if root_for_skill is not None
                else str(skill_md)
            )
        except Exception:
            rel_path = str(skill_md)
        skill_name = frontmatter.get(
            "name", skill_md.stem if not skill_dir else skill_dir.name
        )
        legacy_env_vars, _ = _collect_prerequisite_values(frontmatter)
        required_env_vars = _get_required_environment_variables(
            frontmatter, legacy_env_vars
        )
        backend = _get_terminal_backend_name()
        env_snapshot = load_env()
        missing_required_env_vars = [
            e
            for e in required_env_vars
            if backend in _REMOTE_ENV_BACKENDS
            or not _is_env_var_persisted(e["name"], env_snapshot)
        ]
        capture_result = _capture_required_environment_variables(
            skill_name,
            missing_required_env_vars,
        )
        if missing_required_env_vars:
            env_snapshot = load_env()
        remaining_missing_required_envs = _remaining_required_environment_names(
            required_env_vars,
            capture_result,
            env_snapshot=env_snapshot,
            backend=backend,
        )
        setup_needed = bool(remaining_missing_required_envs)

        result = {
            "success": True,
            "name": skill_name,
            "description": frontmatter.get("description", ""),
            "tags": tags,
            "related_skills": related_skills,
            "content": content,
            "path": rel_path,
            "skill_dir": str(skill_dir) if skill_dir else None,
            "source_scope": root_for_skill.scope if root_for_skill else "external",
            "linked_files": linked_files if linked_files else None,
            "usage_hint": "To view linked files, call skill_view(name, file_path) where file_path is e.g. 'references/api.md' or 'assets/config.yaml'"
            if linked_files
            else None,
            "required_environment_variables": required_env_vars,
            "required_commands": [],
            "missing_required_environment_variables": remaining_missing_required_envs,
            "missing_required_commands": [],
            "setup_needed": setup_needed,
            "setup_skipped": capture_result["setup_skipped"],
            "readiness_status": SkillReadinessStatus.SETUP_NEEDED.value
            if setup_needed
            else SkillReadinessStatus.AVAILABLE.value,
        }

        setup_help = next((e["help"] for e in required_env_vars if e.get("help")), None)
        if setup_help:
            result["setup_help"] = setup_help

        if capture_result["gateway_setup_hint"]:
            result["gateway_setup_hint"] = capture_result["gateway_setup_hint"]

        if setup_needed:
            missing_items = [
                f"env ${env_name}" for env_name in remaining_missing_required_envs
            ]
            setup_note = _build_setup_note(
                SkillReadinessStatus.SETUP_NEEDED,
                missing_items,
                setup_help,
            )
            if backend in _REMOTE_ENV_BACKENDS and setup_note:
                setup_note = f"{setup_note} {backend.upper()}-backed skills need these requirements available inside the remote environment as well."
            if setup_note:
                result["setup_note"] = setup_note

        # Surface agentskills.io optional fields when present
        if frontmatter.get("compatibility"):
            result["compatibility"] = frontmatter["compatibility"]
        if isinstance(metadata, dict):
            result["metadata"] = metadata

        return json.dumps(result, ensure_ascii=False)

    except Exception as e:
        return json.dumps({"success": False, "error": str(e)}, ensure_ascii=False)


# Tool description for model_tools.py
SKILLS_TOOL_DESCRIPTION = """Access skill documents providing specialized instructions, guidelines, and executable knowledge.

Progressive disclosure workflow:
1. skills_list() - Returns metadata (name, description, tags, linked_file_count) for all skills
2. skill_view(name) - Loads full SKILL.md content + shows available linked_files
3. skill_view(name, file_path) - Loads specific linked file (e.g., 'references/api.md', 'scripts/train.py')

Skills may include:
- references/: Additional documentation, API specs, examples
- templates/: Output formats, config files, boilerplate code
- assets/: Supplementary files (agentskills.io standard)
- scripts/: Executable helpers (Python, shell scripts)"""


if __name__ == "__main__":
    """Test the skills tool"""
    print("🎯 Skills Tool Test")
    print("=" * 60)

    # Test listing skills
    print("\n📋 Listing all skills:")
    result = json.loads(skills_list())
    if result["success"]:
        print(
            f"Found {result['count']} skills in {len(result.get('categories', []))} categories"
        )
        print(f"Categories: {result.get('categories', [])}")
        print("\nFirst 10 skills:")
        for skill in result["skills"][:10]:
            cat = f"[{skill['category']}] " if skill.get("category") else ""
            print(f"  • {cat}{skill['name']}: {skill['description'][:60]}...")
    else:
        print(f"Error: {result['error']}")

    # Test viewing a skill
    print("\n📖 Viewing skill 'axolotl':")
    result = json.loads(skill_view("axolotl"))
    if result["success"]:
        print(f"Name: {result['name']}")
        print(f"Description: {result.get('description', 'N/A')[:100]}...")
        print(f"Content length: {len(result['content'])} chars")
        if result.get("linked_files"):
            print(f"Linked files: {result['linked_files']}")
    else:
        print(f"Error: {result['error']}")

    # Test viewing a reference file
    print("\n📄 Viewing reference file 'axolotl/references/dataset-formats.md':")
    result = json.loads(skill_view("axolotl", "references/dataset-formats.md"))
    if result["success"]:
        print(f"File: {result['file']}")
        print(f"Content length: {len(result['content'])} chars")
        print(f"Preview: {result['content'][:150]}...")
    else:
        print(f"Error: {result['error']}")


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

SKILLS_LIST_SCHEMA = {
    "name": "skills_list",
    "description": "List available skills (name + description). Use skill_view(name) to load full content.",
    "parameters": {
        "type": "object",
        "properties": {
            "category": {
                "type": "string",
                "description": "Optional category filter to narrow results",
            }
        },
        "required": [],
    },
}

SKILL_VIEW_SCHEMA = {
    "name": "skill_view",
    "description": "Skills allow for loading information about specific tasks and workflows, as well as scripts and templates. Load a skill's full content or access its linked files (references, templates, scripts). First call returns SKILL.md content plus a 'linked_files' dict showing available references/templates/scripts. To access those, call again with file_path parameter.",
    "parameters": {
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "description": "The skill name (use skills_list to see available skills)",
            },
            "file_path": {
                "type": "string",
                "description": "OPTIONAL: Path to a linked file within the skill (e.g., 'references/api.md', 'templates/config.yaml', 'scripts/validate.py'). Omit to get the main SKILL.md content.",
            },
        },
        "required": ["name"],
    },
}

registry.register(
    name="skills_list",
    toolset="skills",
    schema=SKILLS_LIST_SCHEMA,
    handler=lambda args, **kw: skills_list(
        category=args.get("category"), task_id=kw.get("task_id")
    ),
    check_fn=check_skills_requirements,
    emoji="📚",
)
registry.register(
    name="skill_view",
    toolset="skills",
    schema=SKILL_VIEW_SCHEMA,
    handler=lambda args, **kw: skill_view(
        args.get("name", ""), file_path=args.get("file_path"), task_id=kw.get("task_id")
    ),
    check_fn=check_skills_requirements,
    emoji="📚",
)
