# Dictionary and registries

The dictionary (`data/dictionary.json`) is the git-tracked source of truth for names,
ranking weights, the ontology vocabulary and the registry sources the ontology compiler
reads. The SQLite database is a rebuildable projection of these files; never edit it by hand.

## Locating files (environment variables, all optional)

| Variable | Default | Purpose |
|---|---|---|
| `ANAMNESIS_HOME` | `~/.anamnesis` | State directory: database, dream/decay output, validity shadow file and live flag |
| `ANAMNESIS_DATA_DIR` | `<repo>/data` | Git-tracked memory directory |
| `ANAMNESIS_DB` | `$ANAMNESIS_HOME/memory.db` | Database file |
| `ANAMNESIS_DICT_PATH` | discovery | Explicit dictionary path (otherwise: data dir, repo `data/`, state dir) |
| `ANAMNESIS_REPO_ROOT` | repository root | Root that relative registry paths resolve against |
| `ANAMNESIS_PATH_ROOTS` | none | Extra home-relative checkout dirs (`os.pathsep` separated) stripped when canonicalizing source paths |
| `ANAMNESIS_LOCK_TIMEOUT_S` | 120 | Writer-lock wait for batch jobs |
| `ANAMNESIS_MEMORY_DIR` | `<data>/memory` | Memory note directory scanned by dream |
| `ANAMNESIS_MEMORY_MD_MAX_LINES` / `_MAX_BYTES` | 200 / 25000 | Budget for the memory index file |
| `ANAMNESIS_ALERT_WEBHOOK` | unset | Optional http(s) URL; dream skip-streak alerts are sent as a JSON POST `{"title","text","priority"}` |
| `ANAMNESIS_LLM_PROVIDER` | unset | Required for `--judge`: `gemini` or `command`. Unset means `--judge` refuses and nothing leaves the machine |
| `ANAMNESIS_LLM_MODEL` | `gemini-3.8-flash` | Model id for the `gemini` provider (key from `GEMINI_API_KEY` only) |
| `ANAMNESIS_LLM_COMMAND` | unset | For provider `command`: argv-style command (no shell) reading the prompt on stdin. Trusted-operator control; the child runs with a scrubbed environment (`*_API_KEY`, `*_TOKEN`, `*_SECRET` removed) |
| `ANAMNESIS_LLM_TIMEOUT_S` | 900 | Positive number of seconds; an invalid value is reported as "LLM unavailable" |
| `ANAMNESIS_VALIDITY_SHADOW`, `_LIVE_FILE`, `_LIVE`, `_GATE_DAYS`, `_MIN_COVERAGE` | see `anamnesis_validity.py` | Link-validity shadow file, live flag, forced-live switch, shadow window (30 days) and minimum key coverage (0.5) |
| `ANAMNESIS_DEFAULT_PROJECT` | `default` | Project label for new memories |

Without a dictionary file the loader falls back to a built-in vocabulary (empty projects and
store types). `load_dictionary(strict=True)` raises instead.

## Top-level schema

```json
{
  "version": "1.0.0",
  "projects": [{"name": "example-app", "aliases": ["example"], "description": "...", "enabled": true}],
  "store_types": {
    "decision": {"weight": 1.5, "description": "...", "resolve_target": true, "evidence_source": true}
  },
  "ontology": {
    "node_kinds": ["Project", "Service", "..."],
    "predicates": ["RUNS_ON", "DEPENDS_ON", "..."],
    "polysemy_blocklist": ["run", "make", "work"],
    "spof_kinds": ["Service", "Worker"],
    "global_scopes": [],
    "exclude_path_markers": ["/mirrors/"]
  },
  "dev_root_candidates": ["src", "code", "projects"],
  "path_roots": [],
  "canonical_default_project": "",
  "registries": {},
  "policies": []
}
```

- `store_types.<name>.weight`: ranking multiplier. `resolve_target`: link repair may bind
  prose references to chunks of this type. `evidence_source`: scanned for ontology evidence.
- `ontology.polysemy_blocklist`: common words that never become evidence links on their own.
- `ontology.exclude_path_markers`: chunk paths containing a marker (for example `"/mirrors/"` for a directory of disposable branch copies) are mirrors, never link targets. Empty by default.
- `dev_root_candidates` / `path_roots` / `ANAMNESIS_PATH_ROOTS`: home-relative directories
  stripped so the same file on two machines gets the same `chunk_key`.
- `canonical_default_project`: project that root-relative `docs/`, `scripts/` paths belong to.
- `policies`: policies declared directly (see below). They win over annotations with the same id.

## Registries

`registries` maps each registry type to a list of paths (relative to `ANAMNESIS_REPO_ROOT`,
or absolute / `~/`). Every key is optional; defaults are shown.

```json
{
  "rules_files": ["CLAUDE.md", "AGENTS.md"],
  "decision_logs": ["data/decisions/log.md"],
  "capabilities": ["data/capabilities.json"],
  "capability_edges": ["data/capability-edges.json"],
  "contacts": ["data/contacts.json"],
  "entities": ["data/entities.json"],
  "enforcement": {
    "dirs": ["tests", "scripts", "hooks", ".github/workflows"],
    "extensions": [".py", ".sh", ".yml", ".yaml"],
    "telemetry": ["data/rule-firings.jsonl"]
  }
}
```

Missing files are skipped. Malformed entries are skipped with a warning
(`compile_ontology()` returns `warnings`).

### Policies

In any rules file, annotate a section with an HTML comment (shell-style quoting):

```markdown
## Review

<!-- policy: id=review-required scope=project/example-app enforced_by=required_reviews,CODEOWNERS name="Reviewed before merge" -->
Every change lands through a reviewed pull request.
```

`id` is required; `scope` defaults to `global`; `name` defaults to the nearest heading.
`enforced_by` is a comma-separated list of substrings searched in the enforcement
directories. A policy is `enforced` when a witness file matches, `observed` when telemetry
shows it fired, otherwise `unverified`. The same fields can be given as objects in the
dictionary `policies` array (`id`, `name`, `scope`, `source_path`, `evidence_span`,
`description`, `enforcement_patterns`, `origin`, `valid_from`).

Telemetry JSONL, one object per line: `{"rule": "<policy id>", "ts": "<iso time>"}`.

### Decision log

```markdown
### 2026-01-01 — memory is a git-native build artifact
**Context:** ...
**Decision:** ...
**Supersedes:** 2025-06-01, other-decision-slug
**Origin:** human
**Topic:** storage
**Scope:** project/example-app
```

Headings are `#`-`####` with `YYYY-MM-DD`, a dash or colon, and a title. Bold `**Field:**`
lines carry fields. `Supersedes` takes comma/semicolon separated dates or title slugs and must
match exactly one entry; the older decision becomes `superseded`. `Origin: agent_inferred`
marks the claim as extracted prose rather than authoritative.

### Entities and relations (`entities.json`)

```json
{"entities": [{"id": "service/api", "kind": "Service", "name": "Example API",
               "aliases": ["api"], "attributes": {"port": 8080}}],
 "relations": [{"subject": "service/api", "predicate": "RUNS_ON", "object": "machine/build-host",
                "valid_from": "2026-01-01", "origin": "human"}]}
```

`kind` must be in `ontology.node_kinds` and `predicate` in `ontology.predicates`.

### Capabilities, edges, contacts

```json
{"projects": {"example-app": {"status": "active", "capabilities": [
   {"capability_id": "export-csv", "description": "...", "maturity": "production"}]}}}
```

```json
{"edges": [{"consumer": "docs-site", "provider": "example-app", "capability_id": "export-csv",
            "status": "active", "integration_pattern": "http api"}]}
```

A consumer of `all` or `*` expands to every active project. Contacts:
`{"contacts": [{"name": "...", "role": "...", "organization": "...", "projects": ["example-app"]}]}`.

## Terms

A *nugget* is a short extracted note linked to a source chunk. Links between chunks live in
the `nugget_links` table (the table name is kept for compatibility with the server).

## Lifecycle tools

- `anamnesis_validity.py`: `shadow`, `status`, `gate`, `apply` (bi-temporal link supersession; shadow
  first, gate, then apply; live flag at `$ANAMNESIS_HOME/.validity-live`).
- `dream.py`: dry run by default. `--propose` queues prune proposals under the review dir,
  `--approve ID` / `--reject ID`, `--apply-approved` (snapshot plus undo journal), `--undo`.
  `--judge` is the only mode that calls an LLM. `--apply` is disabled.
- `decay.py`: propose-only value scoring; never retires anything itself.
- `wiki_link_resolver.py`: deterministic `[[slug]]` links as `references` edges.
