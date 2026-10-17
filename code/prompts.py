"""Prompt construction and answer parsing for the paper-01 LLM conditions.

The class list and the team label block are derived from the TRAIN split exactly the way
baselines.py derives them, so the LLM is offered the same label space as the classical
baselines. The label block is the dominant input cost (8092 tokens for team top-20; see
results/cost_team_top20.json) and is placed first, which also makes it a stable cache prefix
for DeepSeek's automatic context caching.
"""
import json
import os
import re
from collections import Counter

ROOT = os.path.join(os.path.dirname(__file__), "..")
LABEL_KEYS = {"fine": "label_fine", "team": "label_team", "coarse": "label_coarse"}
QUERY_CHARS = 4000
EX_CHARS = 1000

ANSWER_RULES = (
    "Answer with a single JSON object and nothing else, with exactly these keys:\n"
    '  "top3": list of up to 3 names, most likely first, copied verbatim from the NAMES line at the '
    "top of the TEAMS block. Never answer with a component name such as `Firefox::Address Bar`: "
    "components are context, only names from the NAMES line are answers;\n"
    '  "abstain": true if the report is too ambiguous to route confidently, otherwise false;\n'
    '  "confidence": number in [0,1] for the first name;\n'
    '  "evidence": one short sentence naming the clue you used.'
)

INSTRUCTION = (
    "You route Mozilla bug reports to the engineering team that owns the fix. "
    "You get a list of teams with their components, then one bug report. "
    "Pick the team most likely to own the fix.\n" + ANSWER_RULES
)

AGENT_INSTRUCTION = (
    "You route Mozilla bug reports to the engineering team that owns the fix. "
    "You get a list of teams with their components, then one bug report. "
    "Before answering you may call the provided tools to gather evidence; call them as often "
    "as you need, then stop calling tools and give your answer. "
    "Pick the team most likely to own the fix.\n" + ANSWER_RULES
)


def load_rows(dataset):
    with open(dataset, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def label_space(rows, label="team", topn=20):
    """The top-`topn` classes by TRAIN frequency, in descending order (same rule as baselines.py)."""
    key = LABEL_KEYS[label]
    counts = Counter(r[key] for r in rows
                     if r["split"] == "train" and r[key] not in ("?", "Other"))
    return [c for c, _ in counts.most_common(topn)]


def label_block(components_path, classes, label="team", desc_chars=200):
    """The label block: the answerable names as an explicit closed list, then the component mapping.

    Order matters, and the first version got it wrong. It opened straight into the component mapping,
    so the class names existed only as bracketed headings among hundreds of `Component: description`
    lines. Every zero-shot answer then came back as a component name instead of a team name: in
    results/llm_zeroshot_deepseek-flash_test2026_disabled_n100_seed0.jsonl, top3 was empty in 99 of
    99 records and the rejected values were all of the form `Product::Component`. The names are now
    listed first, as a closed set, and the instruction points at that line. The names stay first so
    the block remains a stable prompt-cache prefix.
    """
    with open(components_path, encoding="utf-8") as fh:
        comps = json.load(fh)
    lines = {}
    for product in comps["products"]:
        for comp in product["components"]:
            fine = f"{product['name']}::{comp['name']}"
            lab = {"fine": fine, "team": comp.get("team_name") or "?",
                   "coarse": product["name"]}[label]
            if lab in classes:
                desc = (comp.get("description") or "")[:desc_chars]
                lines.setdefault(lab, []).append(f"{fine}: {desc}")
    out = [f"NAMES ({len(classes)} allowed answers, use them verbatim):",
           ", ".join(classes),
           "",
           "COMPONENTS, GROUPED UNDER THE NAME THAT OWNS THEM:"]
    for lab in classes:                      # keep the class order, so the prefix is stable
        if lab in lines:
            out.append(f"[{lab}]\n" + "\n".join(lines[lab]))
    return "\n".join(out)


def bug_text(row, chars=QUERY_CHARS):
    text = row["summary"]
    if row.get("description"):
        text += "\n\n" + row["description"][:chars]
    return text


def example_text(row, label_key, chars=EX_CHARS):
    return f"BUG REPORT\n{bug_text(row, chars)}\nCORRECT TEAM: {row[label_key]}"


def messages_zeroshot(block, bug):
    return [{"role": "system", "content": INSTRUCTION},
            {"role": "user", "content": f"TEAMS\n{block}\n\nBUG REPORT\n{bug}"}]


def messages_rag(block, bug, examples, label_key="label_team"):
    joined = "\n\n".join(example_text(r, label_key) for r in examples)
    return [{"role": "system", "content": INSTRUCTION},
            {"role": "user",
             "content": (f"TEAMS\n{block}\n\nALREADY ROUTED BUGS\n{joined}\n\nBUG REPORT\n{bug}")}]


def messages_agent(block, bug):
    return [{"role": "system", "content": AGENT_INSTRUCTION},
            {"role": "user", "content": f"TEAMS\n{block}\n\nBUG REPORT\n{bug}"}]


def parse_answer(text, classes):
    """Tolerant parse of the strict-JSON answer into a normalised record."""
    obj = None
    if text:
        try:
            obj = json.loads(text)
        except Exception:
            match = re.search(r"\{.*\}", text, re.S)
            if match:
                try:
                    obj = json.loads(match.group(0))
                except Exception:
                    obj = None
    if not isinstance(obj, dict):
        return {"parse_ok": False, "top3": [], "abstain": None, "confidence": None,
                "evidence": "", "unknown_labels": []}
    valid = set(classes)
    raw = obj.get("top3") or []
    if isinstance(raw, str):
        raw = [raw]
    top3, seen, unknown = [], set(), []
    for item in raw:
        if not isinstance(item, str):
            continue
        item = item.strip()
        if item in valid:
            if item not in seen:
                seen.add(item)
                top3.append(item)
        elif item:
            unknown.append(item)
    try:
        confidence = float(obj.get("confidence"))
    except (TypeError, ValueError):
        confidence = None
    abstain = obj.get("abstain")
    return {"parse_ok": True, "top3": top3,
            "abstain": None if abstain is None else bool(abstain),
            "confidence": confidence, "evidence": str(obj.get("evidence") or "")[:300],
            "unknown_labels": unknown}
