#!/usr/bin/env python3
# Shared writer-lock helper for Anamnesis DB writers (indexers, importers).
# One busy writer holding a long transaction starves every other writer at the
# sqlite3 default 5s busy timeout; this helper waits longer and warns loudly.
from __future__ import annotations

import os
import sqlite3
import sys
import time
from pathlib import Path

DEFAULT_TIMEOUT = 120.0
WARN_AFTER = 30.0


def connect_with_lock_warning(
    db_path: "str | Path",
    timeout: "float | None" = None,
    label: str = "",
) -> sqlite3.Connection:
    """Open db_path, retrying past transient 'database is locked' errors.

    Uses a short per-attempt timeout so we can poll and log a WARN once the
    total wait exceeds WARN_AFTER seconds, instead of blocking silently for
    the full `timeout`. With no explicit timeout, ANAMNESIS_LOCK_TIMEOUT_S
    (set by long batch jobs such as the nightly rebuild) overrides the default.
    """
    if timeout is None:
        timeout = float(os.environ.get("ANAMNESIS_LOCK_TIMEOUT_S") or DEFAULT_TIMEOUT)
    start = time.time()
    warned = False
    while True:
        conn = sqlite3.connect(str(db_path), timeout=1.0)
        try:
            conn.execute("BEGIN IMMEDIATE")
            conn.execute("ROLLBACK")
            return conn
        except sqlite3.OperationalError as e:
            conn.close()
            if "locked" not in str(e):
                raise
            elapsed = time.time() - start
            if elapsed > timeout:
                raise
            if elapsed > WARN_AFTER and not warned:
                print(
                    f"WARN: {label or 'writer'} waiting on Anamnesis DB lock "
                    f"({elapsed:.0f}s elapsed, timeout {timeout:.0f}s)",
                    file=sys.stderr,
                )
                warned = True
            time.sleep(1)
