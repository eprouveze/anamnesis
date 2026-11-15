#!/usr/bin/env python3
"""Assess recall quality against a golden question set, using an LLM (or heuristic) judge.

    python tools/eval_recall.py                 # run the full assessment
    python tools/eval_recall.py --resume        # continue from the last checkpoint
    python tools/eval_recall.py --report        # print the report from the last run

Golden set (JSON): {"questions": [{"id", "question", "ground_truth", "category", "difficulty"}]}
Each question is searched with `index.search`; the judge sees the question, the ground truth
and the top passages and returns {recall, accuracy 0-3, hallucination, reasoning}.

Configuration (environment):
    ANAMNESIS_GOLDEN_SET        golden set file          $ANAMNESIS_DATA_DIR/golden-set.json
    ANAMNESIS_EVAL_DIR          results + history        $ANAMNESIS_HOME/eval
    ANAMNESIS_JUDGE_PROVIDER    gemini | anthropic | heuristic      gemini
    ANAMNESIS_JUDGE_MODEL       judge model id           gemini: ANAMNESIS_GENERATION_MODEL; anthropic: required
    GEMINI_API_KEY / ANTHROPIC_API_KEY   key for the chosen provider (environment only)

`heuristic` needs no key or network: recall is "yes" when most distinctive ground-truth terms
appear in the passages. It is a smoke check, not a quality measure.
"""
from __future__ import annotations

import io
import json
import os
import random
import re
import sys
import time
from collections import defaultdict
from contextlib import redirect_stdout
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

HOME = Path.home()
ANAMNESIS_HOME = Path(os.environ.get("ANAMNESIS_HOME") or HOME / ".anamnesis").expanduser()
DATA_DIR = Path(os.environ.get("ANAMNESIS_DATA_DIR") or Path(__file__).resolve().parent.parent / "data").expanduser()
GOLDEN_SET = Path(os.environ.get("ANAMNESIS_GOLDEN_SET") or DATA_DIR / "golden-set.json")
EVAL_DIR = Path(os.environ.get("ANAMNESIS_EVAL_DIR") or ANAMNESIS_HOME / "eval")
NEW_RESULTS = EVAL_DIR / "results.json"
PREVIOUS_RESULTS = EVAL_DIR / "results-previous.json"
HISTORY_FILE = EVAL_DIR / "recall-history.csv"
JUDGE_PROVIDER = os.environ.get("ANAMNESIS_JUDGE_PROVIDER", "gemini").strip().lower()
JUDGE_MODEL = os.environ.get("ANAMNESIS_JUDGE_MODEL") or (
    os.environ.get("ANAMNESIS_GENERATION_MODEL", "gemini-3.8-flash") if JUDGE_PROVIDER == "gemini" else None)

PROMPT = """You are assessing a memory retrieval system. Given a question, ground truth answer, and retrieved passages, judge:

1. **recall**: Did any passage contain the key information needed to answer? "yes" or "no"
2. **accuracy**: How complete is the answer from the passages? 0=wrong, 1=partial, 2=mostly correct, 3=fully correct
3. **hallucination**: Do the passages contain false claims that could mislead? "yes" or "no"

Question: {question}

Ground Truth: {ground_truth}

Retrieved Passages:{passages}

Respond in JSON only:
{{"recall": "yes"|"no", "accuracy": 0-3, "hallucination": "yes"|"no", "reasoning": "brief explanation"}}"""


def bootstrap_ci(values: list[float], confidence: float = 0.95, n_resamples: int = 2000, seed: int = 0):
    """Percentile bootstrap CI of the mean. Returns (mean, lower, upper)."""
    if not values:
        return 0.0, 0.0, 0.0
    rng = random.Random(seed)
    n = len(values)
    means = sorted(sum(values[rng.randrange(n)] for _ in range(n)) / n for _ in range(n_resamples))
    lo = means[int(((1 - confidence) / 2) * (n_resamples - 1))]
    hi = means[int((1 - (1 - confidence) / 2) * (n_resamples - 1))]
    return sum(values) / n, lo, hi


def _format_passages(passages: list[dict]) -> str:
    text = ""
    for i, p in enumerate(passages, 1):
        text += (f"\n[{i}] (type: {p['store_type']}, project: {p['project']}, "
                 f"similarity: {p['similarity']:.2f})\n{p['content'][:400]}\n")
    return text


def _parse_judgment(text: str) -> dict:
    text = (text or "").strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    return json.loads(text)


def _heuristic_judgment(ground_truth: str, passages: list[dict]) -> dict:
    terms = {w for w in re.findall(r"\w+", ground_truth.lower()) if len(w) >= 5}
    if not terms:
        return {"recall": "unknown", "accuracy": 0, "hallucination": "no", "reasoning": "no distinctive terms"}
    blob = " ".join(p["content"].lower() for p in passages)
    frac = sum(1 for t in terms if t in blob) / len(terms)
    return {"recall": "yes" if frac >= 0.5 else "no",
            "accuracy": 3 if frac >= 0.9 else 2 if frac >= 0.5 else 1 if frac else 0,
            "hallucination": "no", "reasoning": f"heuristic: {frac:.0%} of ground-truth terms present"}


def make_judge():
    """Return judge(question, ground_truth, passages) -> dict for the configured provider."""
    if JUDGE_PROVIDER == "heuristic":
        return lambda q, gt, passages: _heuristic_judgment(gt, passages)
    if JUDGE_PROVIDER == "gemini":
        key = os.environ.get("GEMINI_API_KEY")
        if not key:
            sys.exit("ERROR: GEMINI_API_KEY is not set (or use ANAMNESIS_JUDGE_PROVIDER=heuristic)")
        from google import genai
        client = genai.Client(api_key=key)

        def judge(q, gt, passages):
            prompt = PROMPT.format(question=q, ground_truth=gt, passages=_format_passages(passages))
            return _parse_judgment(client.models.generate_content(model=JUDGE_MODEL, contents=prompt).text)
        return judge
    if JUDGE_PROVIDER == "anthropic":
        key = os.environ.get("ANTHROPIC_API_KEY")
        if not key or not JUDGE_MODEL:
            sys.exit("ERROR: provider anthropic needs ANTHROPIC_API_KEY and ANAMNESIS_JUDGE_MODEL")
        import anthropic
        client = anthropic.Anthropic(api_key=key)

        def judge(q, gt, passages):
            prompt = PROMPT.format(question=q, ground_truth=gt, passages=_format_passages(passages))
            msg = client.messages.create(model=JUDGE_MODEL, max_tokens=512,
                                         messages=[{"role": "user", "content": prompt}])
            return _parse_judgment("".join(b.text for b in msg.content if getattr(b, "type", "") == "text"))
        return judge
    sys.exit(f"ERROR: unknown ANAMNESIS_JUDGE_PROVIDER {JUDGE_PROVIDER!r} (gemini | anthropic | heuristic)")


def run_assessment(resume: bool = False, search_fn=None, judge=None, sleep_s: float = 0.3):
    """Run the full assessment, saving after every question. search_fn / judge are injectable."""
    if not GOLDEN_SET.exists():
        sys.exit(f"ERROR: golden set not found at {GOLDEN_SET} (set ANAMNESIS_GOLDEN_SET)")
    questions = json.loads(GOLDEN_SET.read_text())["questions"]
    EVAL_DIR.mkdir(parents=True, exist_ok=True)

    progress: dict = {}
    if resume and NEW_RESULTS.exists():
        progress = json.loads(NEW_RESULTS.read_text())
        print(f"Resuming from {len(progress)} completed questions")
    elif NEW_RESULTS.exists():
        NEW_RESULTS.replace(PREVIOUS_RESULTS)  # keep the last run as the comparison baseline

    judge = judge or make_judge()
    if search_fn is None:
        import index
        search_fn = index.search

    total = len(questions)
    print(f"Assessing {total} questions...\n")
    for q in questions:
        qid = q["id"]
        if qid in progress:
            continue
        print(f"[{len(progress) + 1}/{total}] {qid}: {q['question'][:60]}...", end=" ", flush=True)

        with redirect_stdout(io.StringIO()):  # search prints its own report
            results = search_fn(q["question"], limit=5)

        if not results:
            judgment = {"recall": "no", "accuracy": 0, "hallucination": "no", "reasoning": "No results returned"}
        else:
            try:
                judgment = judge(q["question"], q["ground_truth"], results)
            except Exception as e:
                print(f"JUDGE ERROR: {e}")
                judgment = {"recall": "unknown", "accuracy": 0, "hallucination": "unknown",
                            "reasoning": f"Error: {e}"}

        progress[qid] = {"question": q["question"], "category": q.get("category", "unknown"),
                         "difficulty": q.get("difficulty", "unknown"),
                         "num_results": len(results) if results else 0, "judgment": judgment}
        NEW_RESULTS.write_text(json.dumps(progress, indent=2))  # incremental save
        print(f"recall={judgment.get('recall', '?')} acc={judgment.get('accuracy', '?')} "
              f"hall={judgment.get('hallucination', '?')}")
        if sleep_s:
            time.sleep(sleep_s)

    print("\nAssessment complete!")
    print_report(progress)
    append_to_history(progress)


def append_to_history(progress: dict):
    """Append a row to recall-history.csv for tracking over time."""
    total = len(progress)
    if not total:
        return
    recalls = sum(1 for v in progress.values() if v.get("judgment", {}).get("recall") == "yes")
    halls = sum(1 for v in progress.values() if v.get("judgment", {}).get("hallucination") == "yes")
    accs = [v["judgment"]["accuracy"] for v in progress.values()
            if v.get("judgment", {}).get("accuracy") is not None and v["judgment"]["accuracy"] > 0]
    avg_acc = sum(accs) / len(accs) if accs else 0
    new_file = not HISTORY_FILE.exists()
    with open(HISTORY_FILE, "a") as f:
        if new_file:
            f.write("timestamp,total,recalls,recall_pct,hallucinations,hallucination_pct,avg_accuracy,judge\n")
        f.write(f"{datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')},{total},{recalls},"
                f"{recalls / total * 100:.1f},{halls},{halls / total * 100:.1f},{avg_acc:.2f},"
                f"{JUDGE_PROVIDER}:{JUDGE_MODEL or '-'}\n")
    print(f"\nResults appended to {HISTORY_FILE}")


def print_report(progress: dict | None = None):
    """Print the assessment report (from `progress`, else the last saved run)."""
    if progress is None:
        if NEW_RESULTS.exists():
            progress = json.loads(NEW_RESULTS.read_text())
        elif PREVIOUS_RESULTS.exists():
            progress = json.loads(PREVIOUS_RESULTS.read_text())
        else:
            print("No assessment data found.")
            return

    total = len(progress)
    if not total:
        print("No assessment data found.")
        return
    recalls = sum(1 for v in progress.values() if v.get("judgment", {}).get("recall") == "yes")
    halls = sum(1 for v in progress.values() if v.get("judgment", {}).get("hallucination") == "yes")
    accs = [v["judgment"]["accuracy"] for v in progress.values()
            if v.get("judgment", {}).get("accuracy") is not None and v["judgment"]["accuracy"] > 0]
    flags = [1.0 if v.get("judgment", {}).get("recall") == "yes" else 0.0 for v in progress.values()]
    _, lo, hi = bootstrap_ci(flags)

    print(f"\n{'=' * 50}\nRECALL ASSESSMENT\n{'=' * 50}")
    print(f"Total questions: {total}")
    print(f"Recall:          {recalls}/{total} ({recalls / total * 100:.1f}%) [95% CI: {lo * 100:.1f}% - {hi * 100:.1f}%]")
    print(f"Hallucinations:  {halls}/{total} ({halls / total * 100:.1f}%)")
    if accs:
        print(f"Avg accuracy:    {sum(accs) / len(accs):.2f}/3")

    cats = defaultdict(lambda: {"total": 0, "recalled": 0, "hallucinated": 0})
    for v in progress.values():
        c = cats[v.get("category", "unknown")]
        c["total"] += 1
        c["recalled"] += v.get("judgment", {}).get("recall") == "yes"
        c["hallucinated"] += v.get("judgment", {}).get("hallucination") == "yes"
    print("\nBy category:")
    for cat, d in sorted(cats.items(), key=lambda x: x[1]["recalled"] / max(x[1]["total"], 1)):
        mark = f" [!]{d['hallucinated']}H" if d["hallucinated"] else ""
        print(f"  {cat}: {d['recalled']}/{d['total']} ({d['recalled'] / d['total'] * 100:.0f}%){mark}")

    if PREVIOUS_RESULTS.exists() and NEW_RESULTS.exists():
        old = json.loads(PREVIOUS_RESULTS.read_text())
        if old and len(old) == total:
            old_recalls = sum(1 for v in old.values() if v.get("judgment", {}).get("recall") == "yes")
            delta = recalls - old_recalls
            print(f"\n{'=' * 50}\nvs PREVIOUS: {old_recalls}/{len(old)} ({old_recalls / len(old) * 100:.1f}%) -> "
                  f"{recalls}/{total} ({recalls / total * 100:.1f}%)")
            print(f"Delta: {'+' if delta >= 0 else ''}{delta} questions ({'+' if delta >= 0 else ''}{delta / total * 100:.1f}%)")


if __name__ == "__main__":
    if "--report" in sys.argv:
        print_report()
    else:
        run_assessment(resume="--resume" in sys.argv)
