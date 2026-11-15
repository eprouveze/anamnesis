"""Secret redaction shared by the chat importer and the conversation clipper.

`redact_text` replaces well-known credential shapes with `[REDACTED:<LABEL>]` BEFORE text is
printed, chunked, embedded or stored. It is a safety net, not a guarantee: patterns cover
common vendor formats plus generic `key = value` and `Bearer ...` shapes, and cannot catch
every secret. Do not feed it material you would not be willing to have partially exposed.
"""
from __future__ import annotations

import re

_KV_NAMES = r"(?:api[_-]?key|secret[_-]?key|client[_-]?secret|access[_-]?token|auth[_-]?token|password|passwd)"

# Order matters: specific shapes first, generic fallbacks last.
REDACTION_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----", re.DOTALL), "PRIVATE_KEY"),
    (re.compile(r"eyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}"), "JWT"),
    (re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b"), "AWS_ACCESS_KEY"),
    (re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}"), "GITHUB_TOKEN"),
    (re.compile(r"\bgithub_pat_[A-Za-z0-9_]{20,}"), "GITHUB_TOKEN"),
    (re.compile(r"\b[sr]k_(?:live|test)_[A-Za-z0-9]+"), "PAYMENT_KEY"),
    (re.compile(r"\bxox[abposr]-[A-Za-z0-9-]{10,}"), "SLACK_TOKEN"),
    (re.compile(r"\bAIza[0-9A-Za-z_\-]{35}"), "API_KEY"),
    (re.compile(r"\bhf_[A-Za-z0-9]{30,}"), "API_KEY"),
    (re.compile(r"\bgsk_[A-Za-z0-9]{20,}"), "API_KEY"),
    (re.compile(r"\banm_[A-Za-z0-9_-]{20,}"), "ANAMNESIS_TOKEN"),
    (re.compile(r"\bsk-[A-Za-z0-9_-]{20,}"), "API_KEY"),  # sk-, sk-proj-, sk-ant-, sk-or-v1-...
    (re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+/=-]{16,}"), "BEARER_TOKEN"),
    (re.compile(rf"(?i)\b{_KV_NAMES}\b\s*[:=]\s*[\"']?[^\s\"',;]{{8,}}"), "API_KEY"),
]


def redact_text(content: str) -> tuple[str, int]:
    """Return (redacted_text, number_of_redactions)."""
    count = 0
    for pattern, label in REDACTION_PATTERNS:
        content, n = pattern.subn(f"[REDACTED:{label}]", content)
        count += n
    return content, count
