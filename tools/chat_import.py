"""Parse chat-assistant exports (ChatGPT, Claude, Gemini) into index chunks.

Only the user's own messages are kept. Supported layouts, auto-detected under the path
given to `index.py import-chats`:

  chatgpt/*.json              one conversation per file ({"title", "timestamp", "messages": [...]})
  claude/conversations.json   the Claude data-export list of conversations
  gemini*/ or Takeout/        Gemini JSON files or the Takeout MyActivity.json

No network and no dependencies; the output feeds `index.import_chat_exports`.
"""
import json
import hashlib
from pathlib import Path

try:
    from anamnesis_redact import redact_text
except ImportError:  # imported as a package module
    from .anamnesis_redact import redact_text
from datetime import datetime, timezone


def parse_chatgpt_file(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    title = data.get("title", Path(filepath).stem)
    timestamp = data.get("timestamp", "")
    messages = data.get("messages", [])

    user_messages = []
    for msg in messages:
        if not isinstance(msg, dict) or msg.get("role") != "user":
            continue
        content = msg.get("content", "")
        if not isinstance(content, str):
            continue
        content = content.strip()
        if not content:
            continue
        user_messages.append(dict(
            text=content,
            timestamp=msg.get("timestamp", timestamp),
        ))

    if not user_messages:
        return None

    return dict(
        platform="chatgpt",
        title=title,
        date=timestamp,
        user_messages=user_messages,
        source_file=str(filepath),
    )


def parse_chatgpt_dir(dirpath):
    conversations = []
    dirpath = Path(dirpath)
    json_files = sorted(dirpath.glob("*.json"))
    for f in json_files:
        try:
            conv = parse_chatgpt_file(f)
            if conv:
                conversations.append(conv)
        except (json.JSONDecodeError, KeyError, TypeError, AttributeError) as e:
            print("  Warning: skipping %s: %s" % (f.name, e))
    return conversations


def parse_claude_file(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        print("  Warning: Claude export is not a list, got %s" % type(data).__name__)
        return []

    conversations = []
    for conv in data:
        if not isinstance(conv, dict):
            continue
        title = conv.get("name", "Untitled")
        created_at = conv.get("created_at", "")
        chat_messages = conv.get("chat_messages", [])

        user_messages = []
        for msg in chat_messages:
            if not isinstance(msg, dict) or msg.get("sender") != "human":
                continue
            text = msg.get("text", "")
            if not text or not text.strip():
                continue
            user_messages.append(dict(
                text=text.strip(),
                timestamp=msg.get("created_at", created_at),
            ))

        if not user_messages:
            continue

        conversations.append(dict(
            platform="claude",
            title=title,
            date=created_at,
            user_messages=user_messages,
            source_file=str(filepath),
        ))

    return conversations


def parse_gemini_dir(dirpath):
    dirpath = Path(dirpath)
    conversations = []

    json_files = list(dirpath.rglob("*.json"))

    if not json_files:
        print("  Warning: No JSON files found in Gemini export at %s" % dirpath)
        print("  Gemini Takeout may only contain HTML metadata")
        return []

    for f in json_files:
        try:
            with open(f, "r", encoding="utf-8") as fh:
                data = json.load(fh)

            # Google Takeout MyActivity.json: list of entries with title/time/safeHtmlItem
            if isinstance(data, list) and data and "safeHtmlItem" in data[0]:
                conversations.extend(_parse_gemini_takeout_activity(data, str(f)))
                continue

            messages = []
            title = f.stem

            if isinstance(data, dict):
                messages = data.get("messages", data.get("content", []))
                title = data.get("title", data.get("name", f.stem))
            elif isinstance(data, list):
                messages = data

            user_messages = []
            for msg in messages:
                if not isinstance(msg, dict):
                    continue
                role = msg.get("role", "")
                if role not in ("user", "USER", "0"):
                    continue
                parts = msg.get("parts", [])
                for part in parts:
                    if isinstance(part, dict):
                        text = part.get("text", "")
                    elif isinstance(part, str):
                        text = part
                    else:
                        continue
                    text = text.strip()
                    if text:
                        user_messages.append(dict(
                            text=text,
                            timestamp=msg.get("create_time", ""),
                        ))

            if user_messages:
                conversations.append(dict(
                    platform="gemini",
                    title=title,
                    date="",
                    user_messages=user_messages,
                    source_file=str(f),
                ))
        except (json.JSONDecodeError, KeyError, TypeError, AttributeError) as e:
            print("  Warning: skipping %s: %s" % (f.name, e))

    return conversations


def _parse_gemini_takeout_activity(entries, source_file):
    """Parse Google Takeout MyActivity.json format.

    Each entry has: title (user prompt prefixed with 'Prompted '),
    time (ISO timestamp), safeHtmlItem (AI response as HTML).
    """
    import re

    conversations = []
    for entry in entries:
        title = entry.get("title", "")
        timestamp = entry.get("time", "")

        # Extract user prompt from title
        prompt = title
        if prompt.startswith("Prompted "):
            prompt = prompt[9:]
        prompt = prompt.strip()
        if not prompt:
            continue

        # Extract response text from safeHtmlItem HTML
        response_text = ""
        safe_html = entry.get("safeHtmlItem", [])
        if safe_html and isinstance(safe_html, list):
            html = safe_html[0].get("html", "")
            # Strip HTML tags for plain text
            response_text = re.sub(r"<[^>]+>", " ", html)
            response_text = re.sub(r"\s+", " ", response_text).strip()

        conversations.append(dict(
            platform="gemini",
            title=prompt[:120],
            date=timestamp,
            user_messages=[dict(text=prompt, timestamp=timestamp)],
            source_file=source_file,
        ))

    return conversations


def auto_detect_and_parse(path):
    path = Path(path)

    if not path.exists():
        print("Error: path does not exist: %s" % path)
        return []

    all_conversations = []

    if path.is_file():
        try:
            if path.name == "conversations.json":
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, list) and data and isinstance(data[0], dict) and "chat_messages" in data[0]:
                    print("Detected: Claude conversations.json")
                    return parse_claude_file(path)
                print("Warning: unrecognized conversations.json format")
                return []
            if path.suffix == ".json":
                conv = parse_chatgpt_file(path)
                return [conv] if conv else []
        except (OSError, json.JSONDecodeError, AttributeError, TypeError) as e:
            print("  Warning: could not parse %s: %s" % (path.name, e))
            return []

    if path.is_dir():
        subdirs = {}
        for d in path.iterdir():
            if d.is_dir():
                subdirs[d.name] = d

        if "chatgpt" in subdirs:
            print("Detected: ChatGPT export directory")
            all_conversations.extend(parse_chatgpt_dir(subdirs["chatgpt"]))

        if "claude" in subdirs:
            claude_convs = subdirs["claude"] / "conversations.json"
            if claude_convs.exists():
                print("Detected: Claude conversations.json")
                all_conversations.extend(parse_claude_file(claude_convs))

        gemini_dirs = [d for name, d in subdirs.items() if name.startswith("gemini")]
        for gdir in sorted(gemini_dirs):
            print("Detected: Gemini export directory (%s)" % gdir.name)
            all_conversations.extend(parse_gemini_dir(gdir))

        if all_conversations or any(k in subdirs for k in ("chatgpt", "claude")) or gemini_dirs:
            return all_conversations

        chatgpt_files = list(path.glob("ChatGPT-*.json"))
        if chatgpt_files:
            print("Detected: ChatGPT export directory")
            return parse_chatgpt_dir(path)

        if (path / "Takeout").exists():
            print("Detected: Gemini Takeout directory")
            return parse_gemini_dir(path)

        convs_file = path / "conversations.json"
        if convs_file.exists():
            print("Detected: Claude export directory")
            return parse_claude_file(convs_file)

    print("Warning: could not detect platform for %s" % path)
    return all_conversations


CHUNK_MAX_CHARS = 3500


def _split_messages(texts, max_chars=CHUNK_MAX_CHARS):
    """Group message texts into parts of at most max_chars (an oversized single message is
    split on its own). Nothing is dropped."""
    parts, cur, size = [], [], 0
    for t in texts:
        pieces = [t[i:i + max_chars] for i in range(0, len(t), max_chars)] or [t]
        for piece in pieces:
            if cur and size + len(piece) + 2 > max_chars:
                parts.append("\n\n".join(cur))
                cur, size = [], 0
            cur.append(piece)
            size += len(piece) + 2
    if cur:
        parts.append("\n\n".join(cur))
    return parts


def conversations_to_chunks(conversations, project="chat"):
    """Turn parsed conversations into index chunks (store_type chat_export).

    Only the user's own messages are indexed, with secrets redacted first (anamnesis_redact). A long conversation becomes several
    chunks ("part i/n") instead of being truncated. chunk_id is deterministic from
    platform, title, date and source file, so re-importing the same export is idempotent.
    """
    chunks = []
    for conv in conversations:
        texts = [redact_text(m["text"])[0] for m in conv["user_messages"]]  # redact BEFORE chunking/embedding
        title_clean = redact_text(conv["title"])[0]
        parts = _split_messages(texts)
        platform = conv["platform"]
        title = title_clean
        key = "\x1f".join([platform, title, str(conv.get("date", "")), str(conv.get("source_file", ""))])
        digest = hashlib.sha256(key.encode()).hexdigest()[:16]

        date_str = conv.get("date", "")
        modified_at = None
        if isinstance(date_str, str) and date_str:
            try:
                if date_str.endswith("Z"):
                    date_str = date_str[:-1] + "+00:00"
                modified_at = datetime.fromisoformat(date_str).isoformat()
            except (ValueError, TypeError):
                modified_at = None
        if modified_at is None:
            modified_at = datetime.now(timezone.utc).isoformat()

        for i, content in enumerate(parts):
            suffix = " (part %d/%d)" % (i + 1, len(parts)) if len(parts) > 1 else ""
            chunks.append(dict(
                chunk_id="chat_%s_%s_%d" % (platform, digest, i),
                store_type="chat_export",
                source_path=conv["source_file"],
                title="[" + platform.upper() + "] " + title + suffix,
                content=content,
                modified_at=modified_at,
                project=project,
                topic_ids="chat," + platform,
                weight=1.1,
            ))

    return chunks


def import_summary(conversations):
    by_platform = {}
    for conv in conversations:
        p = conv["platform"]
        if p not in by_platform:
            by_platform[p] = dict(count=0, messages=0)
        by_platform[p]["count"] += 1
        by_platform[p]["messages"] += len(conv["user_messages"])

    lines = ["Chat Import Summary:"]
    total_convs = 0
    total_msgs = 0
    for platform, stats in sorted(by_platform.items()):
        count = stats["count"]
        msgs = stats["messages"]
        lines.append("  %s: %d conversations, %d user messages" % (platform, count, msgs))
        total_convs += count
        total_msgs += msgs
    lines.append("  Total: %d conversations, %d user messages" % (total_convs, total_msgs))
    return "\n".join(lines)
