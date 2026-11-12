#!/usr/bin/env python3
"""anamnesis_dict.py — dictionary loader and shared path/config helpers.

The dictionary (`data/dictionary.json`) is the single git-tracked source for:
project names and aliases, store-type ranking weights, the ontology vocabulary
(node kinds, predicates, polysemy blocklist), canonical-path settings and the
registry sources the ontology compiler reads. See docs/dictionary.md for the
schema. The SQLite database never holds authoritative definitions; it is a
rebuildable projection of this file plus the other git-tracked registries.

Environment variables (all optional):
  ANAMNESIS_HOME        state directory (default ~/.anamnesis): database, dream/decay
                        output, validity shadow file and live flag
  ANAMNESIS_DATA_DIR    git-tracked memory directory (default <repo>/data)
  ANAMNESIS_DB          database file (default $ANAMNESIS_HOME/memory.db)
  ANAMNESIS_DICT_PATH   explicit dictionary.json path (overrides discovery)
  ANAMNESIS_REPO_ROOT   root that relative registry paths resolve against
                        (default: the repository containing this file)
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

_REPO = Path(__file__).resolve().parent.parent

# Used when no dictionary file exists, so every consumer keeps working with
# sensible vocabulary. Project and store-type lists are intentionally empty.
DEFAULT_NODE_KINDS = [
    "Project", "Service", "Machine", "Worker", "Capability",
    "Decision", "Policy", "Person", "Incident", "Topic",
]
DEFAULT_PREDICATES = [
    "RUNS_ON", "HOSTS_SERVICE", "DEPENDS_ON", "USES_CAPABILITY", "PROVIDES_CAPABILITY",
    "FALLBACK_TO", "REPLICATES_TO", "GOVERNS_SCOPE", "DECIDED_FOR", "SUPERSEDES",
    "ENFORCED_BY", "OBSERVED_IN", "CONTACT_FOR_PROJECT",
]
DEFAULT_DICTIONARY: Dict[str, Any] = {
    "version": "0",
    "projects": [],
    "store_types": {},
    "ontology": {
        "node_kinds": DEFAULT_NODE_KINDS,
        "predicates": DEFAULT_PREDICATES,
        "polysemy_blocklist": [],
    },
    "dev_root_candidates": ["src", "code", "projects"],
}

_CACHE: Dict[str, Tuple[int, Dict[str, Any]]] = {}


# ---------------------------------------------------------------------------
# Locations
# ---------------------------------------------------------------------------


def anamnesis_home() -> Path:
    """State directory (`ANAMNESIS_HOME`, default `~/.anamnesis`)."""
    return Path(os.environ.get("ANAMNESIS_HOME") or (Path.home() / ".anamnesis")).expanduser()


def data_dir() -> Path:
    """Git-tracked memory directory (`ANAMNESIS_DATA_DIR`, default `<repo>/data`)."""
    env = os.environ.get("ANAMNESIS_DATA_DIR")
    return Path(env).expanduser() if env else _REPO / "data"


def db_path() -> Path:
    """Database path (`ANAMNESIS_DB`, default `$ANAMNESIS_HOME/memory.db`)."""
    env = os.environ.get("ANAMNESIS_DB")
    return Path(env).expanduser() if env else anamnesis_home() / "memory.db"


def repo_root() -> Path:
    """Root that relative registry paths resolve against."""
    env = os.environ.get("ANAMNESIS_REPO_ROOT")
    return Path(env).expanduser() if env else _REPO


def _resolve_dict_path() -> Path:
    """Locate dictionary.json: explicit override, data dir, repo data/, state dir."""
    override = os.environ.get("ANAMNESIS_DICT_PATH")
    if override:
        return Path(override).expanduser()
    here = Path(__file__).resolve().parent
    candidates = (
        data_dir() / "dictionary.json",
        _REPO / "data" / "dictionary.json",
        anamnesis_home() / "dictionary.json",
        here / "data" / "dictionary.json",
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_dictionary(strict: bool = False) -> Dict[str, Any]:
    """Return the parsed dictionary.

    A missing file yields DEFAULT_DICTIONARY unless `strict` is set. Parsed
    content is cached per (path, mtime) so hot paths can call this freely.
    """
    path = _resolve_dict_path()
    if not path.exists():
        if strict:
            raise FileNotFoundError(f"Anamnesis dictionary not found at {path}")
        return DEFAULT_DICTIONARY
    mtime = path.stat().st_mtime_ns
    cached = _CACHE.get(str(path))
    if cached and cached[0] == mtime:
        return cached[1]
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{path}: dictionary must be a JSON object")
    _CACHE[str(path)] = (mtime, data)
    return data


def dictionary_path() -> Path:
    """Path the loader reads (or would read) the dictionary from."""
    return _resolve_dict_path()


def reset_cache() -> None:
    _CACHE.clear()


def get_project_entries(enabled_only: bool = True) -> List[Dict[str, Any]]:
    projects = load_dictionary().get("projects", [])
    if enabled_only:
        return [p for p in projects if p.get("enabled", True)]
    return list(projects)


def get_all_projects(enabled_only: bool = True) -> List[str]:
    return [p["name"] for p in get_project_entries(enabled_only=enabled_only)]


def get_store_weights() -> Dict[str, float]:
    store_types = load_dictionary().get("store_types", {})
    return {k: float(v.get("weight", 1.0)) for k, v in store_types.items()}


def get_store_types_with(flag: str, default: Tuple[str, ...] = ()) -> Tuple[str, ...]:
    """Store types whose dictionary entry sets `flag` truthy; `default` if none do.

    Used for `resolve_target` (stores link repair may bind to) and
    `evidence_source` (stores scanned for ontology evidence).
    """
    store_types = load_dictionary().get("store_types", {})
    names = tuple(k for k, v in store_types.items() if isinstance(v, dict) and v.get(flag))
    return names or default


def get_ontology_node_kinds() -> Set[str]:
    return set(load_dictionary().get("ontology", {}).get("node_kinds", []))


def get_ontology_predicates() -> Set[str]:
    return set(load_dictionary().get("ontology", {}).get("predicates", []))


def get_polysemy_blocklist() -> Set[str]:
    return {w.lower() for w in load_dictionary().get("ontology", {}).get("polysemy_blocklist", [])}


def get_ontology_setting(key: str, default: Any = None) -> Any:
    """Read an optional `ontology.<key>` setting (e.g. spof_kinds, global_scopes)."""
    return load_dictionary().get("ontology", {}).get(key, default)


def get_registries() -> Dict[str, Any]:
    """The `registries` section (sources for the ontology compiler); {} if absent."""
    reg = load_dictionary().get("registries", {})
    return reg if isinstance(reg, dict) else {}


def get_policies() -> List[Dict[str, Any]]:
    pols = load_dictionary().get("policies", [])
    return [p for p in pols if isinstance(p, dict) and p.get("id")]


def get_path_roots() -> List[str]:
    """Home-relative directory names stripped when canonicalizing source paths.

    Union of `dev_root_candidates` and `path_roots` from the dictionary plus the
    os.pathsep-separated `ANAMNESIS_PATH_ROOTS` env var.
    """
    data = load_dictionary()
    roots: List[str] = []
    for key in ("dev_root_candidates", "path_roots"):
        roots.extend(str(r).strip("/") for r in data.get(key, []) if str(r).strip("/"))
    env = os.environ.get("ANAMNESIS_PATH_ROOTS", "")
    roots.extend(r.strip("/") for r in env.split(os.pathsep) if r.strip("/"))
    seen: List[str] = []
    for r in roots:
        if r not in seen:
            seen.append(r)
    return seen


def resolve_project_path(project_name: str, base_home: Optional[Path] = None) -> Optional[Path]:
    """Find a project's checkout under `<home>/<dev_root_candidate>/<name>`."""
    home = base_home or Path.home()
    for cand in get_path_roots():
        target = home / cand / project_name
        if target.exists() and target.is_dir():
            return target
    return None


if __name__ == "__main__":
    data = load_dictionary()
    print(f"Dictionary v{data.get('version')} loaded from {_resolve_dict_path()}.")
    print(f"Projects ({len(get_all_projects())}): {', '.join(get_all_projects())}")
    print(f"Store types ({len(get_store_weights())}): {', '.join(get_store_weights().keys())}")
    print(f"Ontology node kinds ({len(get_ontology_node_kinds())}): {', '.join(sorted(get_ontology_node_kinds()))}")
