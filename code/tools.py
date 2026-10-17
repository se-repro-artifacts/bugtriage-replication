"""Tools exposed to the routing agent (paper01_continue.md step 2 item 4).

Three tools: search similar historical bugs, look up component/team ownership, and list a
team's recent bugs. Every tool reads the TRAIN split only, so no test information can leak into
the answer. Each tool returns a compact plain-text block, which keeps tool-call output small.
"""
import json

SEARCH = {
    "type": "function",
    "function": {
        "name": "search_similar_bugs",
        "description": ("Find already-routed historical bug reports similar to the current one. "
                        "Returns each bug's id, the team that fixed it, and its summary."),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string",
                          "description": ("Extra keywords to focus the search (component, feature or "
                                          "error text). Pass an empty string to search using the bug "
                                          "text alone.")},
                "k": {"type": "integer", "description": "How many bugs to return, 1-20."},
            },
            "required": [],
        },
    },
}

COMPONENT_INFO = {
    "type": "function",
    "function": {
        "name": "get_component_info",
        "description": ("Look up which team owns components whose name contains the given text, "
                        "e.g. 'Layout', 'WebRTC', 'DevTools'."),
        "parameters": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Component or team name fragment."},
            },
            "required": ["name"],
        },
    },
}

TEAM_RECENT = {
    "type": "function",
    "function": {
        "name": "recent_bugs_for_team",
        "description": "List the most recently created training bug reports owned by one team.",
        "parameters": {
            "type": "object",
            "properties": {
                "team": {"type": "string", "description": "Exact team name from the team list."},
                "n": {"type": "integer", "description": "How many bugs to return, 1-10."},
            },
            "required": ["team"],
        },
    },
}

TOOLS = [SEARCH, COMPONENT_INFO, TEAM_RECENT]
TOOL_NAMES = [t["function"]["name"] for t in TOOLS]


def _clamp(value, low, high, default):
    try:
        return max(low, min(high, int(value)))
    except (TypeError, ValueError):
        return default


def dispatch(name, args, retriever, bug_text_value, classes):
    """Execute one tool call and return its textual result. Never raises."""
    if not isinstance(args, dict):
        args = {}
    try:
        if name == "search_similar_bugs":
            extra = str(args.get("query") or "").strip()
            k = _clamp(args.get("k"), 1, 20, 5)
            query = (bug_text_value + " " + extra).strip()
            hits = retriever.search(query, k)
            if not hits:
                return "no similar bugs found"
            return "\n".join(
                f"bug {row['id']} | team {row[retriever.label_key]} | {row['summary'][:160]}"
                for row, _ in hits)

        if name == "get_component_info":
            needle = str(args.get("name") or "").strip()
            if not needle:
                return "error: 'name' is required"
            lines = retriever.component_lines(needle)
            if not lines:
                return f"no component matches {needle!r}"
            return "\n".join(lines)

        if name == "recent_bugs_for_team":
            team = str(args.get("team") or "").strip()
            if team not in classes:
                return (f"{team!r} is not one of the teams. Valid teams: "
                        + ", ".join(sorted(classes)))
            n = _clamp(args.get("n"), 1, 10, 5)
            rows = retriever.recent_for_team(team, n)
            if not rows:
                return f"no training bugs recorded for {team}"
            return "\n".join(f"bug {r['id']} | {r['summary'][:160]}" for r in rows)

        return f"error: unknown tool {name!r}"
    except Exception as exc:  # a tool failure must not abort the run
        return f"error while running {name}: {exc!r}"


def as_json(tool_calls):
    return [{"name": tc.get("function", {}).get("name"),
             "arguments": tc.get("function", {}).get("arguments")} for tc in tool_calls]
