#!/usr/bin/env python3
"""anamnesis_llm.py — one opt-in text-completion call for dream.py and decay.py.

This is the only code in the consolidation tools that talks to a model, and it is
only reached when the operator asks for it (`dream.py --judge`, `decay.py --judge`).
Everything it sends is memory content, so it leaves your machine for the provider
you configure: choose a provider whose data handling you accept, or use the
`command` provider with a local model.

Environment variables:
  ANAMNESIS_LLM_PROVIDER   REQUIRED to use --judge: "gemini" or "command". There is no
                           default, so memory content never leaves the machine by accident.
  ANAMNESIS_LLM_MODEL      model id for the gemini provider (default gemini-3.8-flash)
  ANAMNESIS_LLM_COMMAND    for provider "command": an argv-style command line (no shell)
                           that reads the prompt on stdin and prints the answer on stdout,
                           e.g. a local-model CLI wrapper. It is a trusted-operator control:
                           it runs with your privileges, so only set it to a program you trust.
                           The child gets a scrubbed environment (names containing KEY,
                           TOKEN, SECRET, PASSWORD, CREDENTIAL or AUTH are removed unless
                           listed in ANAMNESIS_LLM_COMMAND_ENV_PASS).
  ANAMNESIS_LLM_TIMEOUT_S  per-call timeout in seconds, a positive number (default 900)
  GEMINI_API_KEY           API key for the gemini provider (GOOGLE_API_KEY also read);
                           taken only from the environment
"""
from __future__ import annotations

import os
import shlex
import subprocess

DEFAULT_MODEL = "gemini-3.8-flash"
DEFAULT_TIMEOUT_S = 900.0


class LLMUnavailable(RuntimeError):
    """The configured provider cannot answer (missing key, SDK, command or timeout)."""


def provider() -> str:
    """Configured provider name; empty when unset (calls then refuse)."""
    return (os.environ.get("ANAMNESIS_LLM_PROVIDER") or "").strip().lower()


def default_model() -> str:
    return os.environ.get("ANAMNESIS_LLM_MODEL") or DEFAULT_MODEL


def _timeout(timeout: float | None) -> float:
    raw = timeout if timeout is not None else os.environ.get("ANAMNESIS_LLM_TIMEOUT_S") or DEFAULT_TIMEOUT_S
    try:
        value = float(raw)
    except (TypeError, ValueError):
        raise LLMUnavailable(f"ANAMNESIS_LLM_TIMEOUT_S must be a number, got {raw!r}") from None
    if not value > 0 or value != value or value == float("inf"):
        raise LLMUnavailable(f"ANAMNESIS_LLM_TIMEOUT_S must be a positive finite number, got {raw!r}")
    return value


_SECRET_MARKERS = ("KEY", "TOKEN", "SECRET", "PASSWORD", "PASSWD", "CREDENTIAL", "AUTH")


def _scrubbed_env() -> dict:
    """Environment for the command child: parent env minus credential-looking variables.

    Any variable whose name contains KEY, TOKEN, SECRET, PASSWORD, CREDENTIAL or AUTH is
    dropped, except names listed in ANAMNESIS_LLM_COMMAND_ENV_PASS (comma-separated), so the
    operator decides exactly which credential the command may see.
    """
    passthrough = {n.strip().upper() for n in os.environ.get("ANAMNESIS_LLM_COMMAND_ENV_PASS", "").split(",") if n.strip()}
    return {k: v for k, v in os.environ.items()
            if k.upper() in passthrough or not any(mk in k.upper() for mk in _SECRET_MARKERS)}


def _complete_command(prompt: str, timeout: float) -> str:
    raw = os.environ.get("ANAMNESIS_LLM_COMMAND", "").strip()
    if not raw:
        raise LLMUnavailable("ANAMNESIS_LLM_PROVIDER=command needs ANAMNESIS_LLM_COMMAND")
    try:
        r = subprocess.run(shlex.split(raw), input=prompt, capture_output=True, text=True,
                           timeout=timeout, env=_scrubbed_env())
    except (OSError, ValueError, subprocess.TimeoutExpired) as exc:  # ValueError: bad quoting / undecodable output
        raise LLMUnavailable(f"llm command failed: {exc.__class__.__name__}") from exc
    if r.returncode != 0 or not r.stdout.strip():
        raise LLMUnavailable(f"llm command exited {r.returncode} with {len(r.stdout)} bytes of output")
    return r.stdout.strip()


def _complete_gemini(prompt: str, model: str, timeout: float) -> str:
    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key:
        raise LLMUnavailable("set GEMINI_API_KEY (or choose ANAMNESIS_LLM_PROVIDER=command)")
    try:
        from google import genai
        from google.genai import types
    except ImportError as exc:
        raise LLMUnavailable("google-genai is not installed (pip install google-genai)") from exc
    try:
        client = genai.Client(api_key=key, http_options=types.HttpOptions(timeout=int(timeout * 1000)))
        text = client.models.generate_content(model=model, contents=prompt).text
    except Exception as exc:  # SDK raises many types; surface one without leaking the key
        raise LLMUnavailable(f"gemini call failed: {exc.__class__.__name__}") from exc
    if not text or not text.strip():
        raise LLMUnavailable("gemini returned an empty answer")
    return text.strip()


def complete(prompt: str, model: str | None = None, timeout: float | None = None) -> str:
    """Return the model's text answer, or raise LLMUnavailable."""
    prov = provider()
    if not prov:
        raise LLMUnavailable(
            "no LLM provider configured: set ANAMNESIS_LLM_PROVIDER=command (with "
            "ANAMNESIS_LLM_COMMAND, a local model) or ANAMNESIS_LLM_PROVIDER=gemini "
            "(with GEMINI_API_KEY; memory content is sent to that provider)")
    t = _timeout(timeout)
    if prov == "command":
        return _complete_command(prompt, t)
    if prov == "gemini":
        return _complete_gemini(prompt, model or default_model(), t)
    raise LLMUnavailable(f"unknown ANAMNESIS_LLM_PROVIDER {prov!r} (use gemini or command)")
