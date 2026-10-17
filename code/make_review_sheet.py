"""Build the review sheet for the manual ownership-label check.

The review has to answer one question per bug: which of the four ownership categories does this bug
really belong to? Answering it from the CSV alone means opening Bugzilla for every row, because the
CSV shows only the initial and final component. The full sequence of moves was already fetched into
data/history.jsonl and data/2026/history.jsonl, so it is rendered here instead: the reviewer sees
every component and product change, with its timestamp and whether a bot made it, on one page.

The cards are rendered here, in Python, and not by JavaScript. An earlier version built them in the
browser, which meant a viewer that does not execute scripts showed an empty page --- and a review
sheet that is blank until a script runs is the wrong trade for something a person has to work
through. JavaScript is now only an enhancement (keeping answers in the browser and exporting them);
everything needed to *read* and to *decide* is in the HTML.

Outputs
  results/review_sheet.html          self-contained, readable without scripts
  results/manual_review_sample.csv   refreshed with a `moves` column for spreadsheet users

Usage: python make_review_sheet.py
"""
import csv
import html
import json
import os

ROOT = os.path.join(os.path.dirname(__file__), "..")
SAMPLE = os.path.join(ROOT, "results", "manual_review_sample.csv")
SHEET = os.path.join(ROOT, "results", "review_sheet.html")
CATEGORIES = ("never_changed", "inbox_triage", "specific_reassign", "bot_only")
CATEGORY_ZH = {
    "never_changed": "从未改派（组件名一直没变）",
    "inbox_triage": "从收件箱分流（General / Untriaged）",
    "specific_reassign": "从具体组件被改派走（本文的正类）",
    "bot_only": "只有机器人改过（排除）",
}


def load_moves():
    """bug id -> list of moves, from both period files."""
    moves = {}
    for name in ("history.jsonl", os.path.join("2026", "history.jsonl")):
        path = os.path.join(ROOT, "data", name)
        if not os.path.exists(path):
            continue
        for line in open(path, encoding="utf-8"):
            if line.strip():
                row = json.loads(line)
                moves[row["id"]] = row.get("moves") or []
    return moves


def chain_text(moves):
    """One line per move, in the order they happened, for the CSV column."""
    if not moves:
        return ""
    parts = []
    for m in moves:
        who = "bot" if m.get("by_bot") else "human"
        parts.append(f"{m.get('when', '')[:16]} {m.get('field')}: "
                     f"{m.get('removed')} -> {m.get('added')} [{who}]")
    return " ; ".join(parts)


def moves_table(bug):
    if not bug["moves"]:
        return ('<div class="missing">数据集中没有变更记录：初始与最终组件名相同，'
                '核对上面两列是否一致即可。</div>')
    rows = []
    for m in bug["moves"]:
        who = "机器人" if m.get("by_bot") else "人"
        cls = ' class="bot"' if m.get("by_bot") else ""
        rows.append(
            f"<tr><td>{html.escape((m.get('when') or '').replace('T', ' ')[:16])}</td>"
            f"<td>{html.escape(str(m.get('field')))}</td>"
            f"<td>{html.escape(str(m.get('removed')))}</td>"
            f"<td>{html.escape(str(m.get('added')))}</td>"
            f"<td{cls}>{who}</td></tr>")
    return ('<table class="mv"><tr><th>时间</th><th>字段</th><th>改前</th><th>改后</th>'
            '<th>谁改的</th></tr>' + "".join(rows) + '</table>')


def card_html(bug, index):
    options = "".join(
        f'<label class="opt"><input type="radio" name="c{bug["id"]}" value="{c}" '
        f'data-id="{bug["id"]}"> <span>{html.escape(CATEGORY_ZH[c])}</span> '
        f'<code>{c}</code></label>'
        for c in CATEGORIES)
    return f"""<div class="card todo" id="card{bug['id']}">
  <div class="head">
    <span class="id">#{index} · bug {bug['id']}</span>
    <span class="meta">{html.escape(bug['split'])} · 创建于 {html.escape(bug['created'])}</span>
    <a href="{html.escape(bug['url'])}" target="_blank" rel="noopener">在 Bugzilla 打开 ↗</a>
  </div>
  <div class="summary">{html.escape(bug['summary'])}</div>
  <table class="mv">
    <tr><th>自动判断</th><th>初始组件</th><th>最终组件</th><th>n_moves</th>
        <th>有人工改动</th><th>有机器人改动</th></tr>
    <tr><td class="auto">{html.escape(bug['auto'])}</td>
        <td>{html.escape(bug['initial'])}</td><td>{html.escape(bug['final'])}</td>
        <td>{html.escape(bug['n_moves'])}</td><td>{html.escape(bug['human'])}</td>
        <td>{html.escape(bug['bot'])}</td></tr>
  </table>
  <div class="hint">完整变更历史：</div>
  {moves_table(bug)}
  <div class="opts">{options}</div>
  <input type="text" placeholder="备注（只在认为自动判断错了、或拿不准时填）"
         value="" data-note="{bug['id']}">
</div>"""


def sheet_markdown(bugs):
    """The same content as markdown.

    Rendered alongside the HTML because it costs nothing and it removes a dependency on the viewer:
    a markdown preview, a text editor and a phone all show it, where an embedded script might be
    stripped. Answers go in the CSV either way.
    """
    lines = [
        f"# 人工抽查：归属歧义标签（{len(bugs)} 条）",
        "",
        "填写位置：`results/manual_review_sample.csv` 的 `reviewer_category` 与 `reviewer_notes` 两列。",
        "交互式版本（可一键作答并导出）：`results/review_sheet.html`，用**浏览器**打开。",
        "",
        "**判断规则（按顺序）**",
        "",
    ]
    for i, c in enumerate(CATEGORIES, 1):
        lines.append(f"{i}. {CATEGORY_ZH[c]} — `{c}`")
    lines += ["", "---", ""]
    for i, b in enumerate(bugs, 1):
        lines += [
            f"## #{i} · bug {b['id']}",
            "",
            f"- **自动判断**：`{b['auto']}`",
            f"- **初始组件**：`{b['initial']}`",
            f"- **最终组件**：`{b['final']}`",
            f"- n_moves={b['n_moves']} · 有人工改动={b['human']} · 有机器人改动={b['bot']}",
            f"- 测试期：{b['split']} · 创建于 {b['created']}",
            f"- [在 Bugzilla 打开]({b['url']})",
            "",
            f"**摘要**：{b['summary']}",
            "",
        ]
        if b["moves"]:
            lines += ["**完整变更历史**", "",
                      "| 时间 | 字段 | 改前 | 改后 | 谁改的 |",
                      "|---|---|---|---|---|"]
            for m in b["moves"]:
                lines.append(
                    f"| {(m.get('when') or '').replace('T', ' ')[:16]} | {m.get('field')} | "
                    f"{m.get('removed')} | {m.get('added')} | "
                    f"{'机器人' if m.get('by_bot') else '人'} |")
            lines.append("")
        else:
            lines += ["_数据集中没有变更记录：初始与最终组件名相同。_", ""]
        lines += [f"**你的判断**：`____________`　**备注**：", "", "---", ""]
    return "\n".join(lines)


def main():
    with open(SAMPLE, encoding="utf-8-sig") as fh:
        rows = list(csv.DictReader(fh))
    moves_by_id = load_moves()

    bugs = []
    for row in rows:
        bug_id = int(row["bug_id"])
        moves = moves_by_id.get(bug_id, [])
        bugs.append({
            "id": bug_id, "url": row["url"], "split": row["split"],
            "created": (row.get("created") or "")[:10], "auto": row["category"],
            "summary": row["summary"], "initial": row["initial_component"],
            "final": row["final_component"], "n_moves": row.get("n_moves", ""),
            "human": row.get("any_human_move", ""), "bot": row.get("any_bot_move", ""),
            "moves": moves, "chain": chain_text(moves),
        })

    # refresh the CSV with the chain, so a spreadsheet reviewer is not left guessing either
    fieldnames = list(rows[0].keys())
    if "moves" not in fieldnames:
        fieldnames.insert(fieldnames.index("summary"), "moves")
    with open(SAMPLE, "w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row, bug in zip(rows, bugs):
            row["moves"] = bug["chain"]
            writer.writerow(row)
    print(f"refreshed {os.path.relpath(SAMPLE, ROOT)} with a moves column")

    cards = "\n".join(card_html(b, i + 1) for i, b in enumerate(bugs))
    rules = "\n".join(
        f"<li><b>{i + 1}.</b> {html.escape(CATEGORY_ZH[c])} <code>{c}</code></li>"
        for i, c in enumerate(CATEGORIES))
    payload = json.dumps([{k: b[k] for k in ("id", "auto")} for b in bugs], ensure_ascii=False)

    page = f"""<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8">
<title>人工抽查：归属歧义标签（{len(bugs)} 条）</title>
<style>
  :root {{ --line:#d5d5d5; --muted:#666; --accent:#1a5fb4; --ok:#2a7d4f; --warn:#b45309; }}
  * {{ box-sizing: border-box; }}
  body {{ font-family:"Microsoft YaHei","Segoe UI",sans-serif; margin:0; background:#f6f7f9;
          color:#111; }}
  header {{ position:sticky; top:0; background:#fff; border-bottom:1px solid var(--line);
            padding:10px 16px; z-index:10; }}
  h1 {{ font-size:17px; margin:0 0 6px; }}
  .bar {{ display:flex; align-items:center; gap:14px; flex-wrap:wrap; font-size:13px; }}
  .bar b {{ color:var(--accent); }}
  .rules {{ font-size:12px; color:#333; background:#f0f4ff; border:1px solid #ccd8f5;
            border-radius:6px; padding:8px 10px; margin:8px 0; }}
  .rules ol {{ margin:4px 0 0 18px; padding:0; }}
  button {{ font:inherit; padding:5px 12px; border:1px solid var(--line); border-radius:5px;
            background:#fff; cursor:pointer; }}
  button:hover {{ border-color:var(--accent); color:var(--accent); }}
  main {{ padding:14px 16px 60px; max-width:1000px; margin:0 auto; }}
  .card {{ background:#fff; border:1px solid var(--line); border-radius:8px; padding:12px 14px;
           margin-bottom:12px; }}
  .card.done {{ border-left:5px solid var(--ok); }}
  .card.todo {{ border-left:5px solid var(--warn); }}
  .head {{ display:flex; justify-content:space-between; gap:10px; align-items:baseline;
           flex-wrap:wrap; }}
  .id {{ font-weight:700; font-size:15px; }}
  .meta {{ font-size:12px; color:var(--muted); }}
  .summary {{ font-size:13px; margin:6px 0 8px; }}
  table.mv {{ border-collapse:collapse; width:100%; font-size:12px; margin:4px 0 8px; }}
  table.mv th, table.mv td {{ border:1px solid var(--line); padding:3px 6px; text-align:left; }}
  table.mv th {{ background:#f2f2f2; font-weight:600; }}
  .bot {{ color:#b45309; }}
  .auto {{ font-weight:700; }}
  .opts {{ display:flex; gap:14px; flex-wrap:wrap; margin:8px 0 6px; font-size:13px; }}
  .opt {{ display:flex; align-items:center; gap:4px; cursor:pointer; }}
  .opt code {{ font-size:11px; color:var(--muted); }}
  input[type=text] {{ font:inherit; font-size:12px; width:100%; padding:5px 7px;
                      border:1px solid var(--line); border-radius:5px; }}
  .hint {{ font-size:11px; color:var(--muted); margin-top:4px; }}
  textarea#out {{ width:100%; height:110px; font-family:Consolas,monospace; font-size:11px;
                  margin-top:8px; }}
  .missing {{ font-size:12px; color:var(--muted); font-style:italic; }}
  noscript div {{ background:#fff8e6; border:1px solid #f0d9a0; padding:8px 10px;
                  border-radius:6px; font-size:12px; margin:8px 0; }}
</style>
</head>
<body>
<header>
  <h1>人工抽查：归属歧义标签</h1>
  <div class="bar">
    <span>共 <b>{len(bugs)}</b> 条，已审 <b id="done">0</b> 条</span>
    <button id="export">导出结果（复制 CSV）</button>
    <button id="clear">清空我的作答</button>
    <span class="meta">答案自动存在浏览器里，关掉页面不会丢</span>
  </div>
  <div class="rules">
    <b>判断规则（按顺序）</b>
    <ol>{rules}</ol>
    <div style="margin-top:4px">下面是每一条的<b>完整变更历史</b>，不需要再打开 Bugzilla。</div>
  </div>
  <textarea id="out" placeholder="点「导出结果」后，把这里的内容发给我（或粘回 CSV 的两列）"></textarea>
  <noscript><div><b>你的查看器没有运行 JavaScript。</b>全部 {len(bugs)} 条记录仍然可以正常阅读和判断，
    只是答案不会被保存、也无法一键导出。可以在
    <code>results/manual_review_sample.csv</code>（Excel 打开）里直接填
    <code>reviewer_category</code> 和 <code>reviewer_notes</code> 两列。</div></noscript>
</header>
<main id="cards">
{cards}
</main>
<script>
const IDS = {payload};
const KEY = "bugtriage_review_v1";
let state = {{}};
try {{ state = JSON.parse(localStorage.getItem(KEY) || "{{}}"); }} catch (e) {{ state = {{}}; }}

function save() {{ try {{ localStorage.setItem(KEY, JSON.stringify(state)); }} catch (e) {{}} }}
function updateCount() {{
  const n = IDS.filter(b => (state[b.id] || {{}}).category).length;
  document.getElementById("done").textContent = n;
}}
function restore() {{
  for (const b of IDS) {{
    const s = state[b.id];
    if (!s) continue;
    const card = document.getElementById("card" + b.id);
    if (s.category) {{
      const el = card.querySelector('input[value="' + s.category + '"]');
      if (el) el.checked = true;
      card.className = "card done";
    }}
    const note = card.querySelector('input[data-note]');
    if (note && s.note) note.value = s.note;
  }}
  updateCount();
}}
document.querySelectorAll('input[type=radio]').forEach(el => {{
  el.addEventListener("change", () => {{
    const id = el.dataset.id;
    state[id] = state[id] || {{}};
    state[id].category = el.value;
    save();
    document.getElementById("card" + id).className = "card done";
    updateCount();
  }});
}});
document.querySelectorAll('input[data-note]').forEach(el => {{
  el.addEventListener("change", () => {{
    const id = el.dataset.note;
    state[id] = state[id] || {{}};
    state[id].note = el.value;
    save();
  }});
}});
document.getElementById("export").addEventListener("click", () => {{
  const lines = ["bug_id,category,reviewer_category,reviewer_notes"];
  for (const b of IDS) {{
    const card = document.getElementById("card" + b.id);
    const picked = card.querySelector('input[type=radio]:checked');
    const note = (card.querySelector('input[data-note]').value || "").replace(/"/g, '""');
    lines.push(b.id + "," + b.auto + "," + (picked ? picked.value : "") + ',"' + note + '"');
  }}
  const box = document.getElementById("out");
  box.value = lines.join("\\n");
  box.select();
  try {{ document.execCommand("copy"); }} catch (e) {{}}
}});
document.getElementById("clear").addEventListener("click", () => {{
  if (confirm("清空所有作答？")) {{ state = {{}}; save(); location.reload(); }}
}});
document.addEventListener("keydown", e => {{
  if (e.target.tagName === "INPUT" || e.target.tagName === "TEXTAREA") return;
  const n = parseInt(e.key, 10);
  if (!(n >= 1 && n <= 4)) return;
  const next = IDS.find(b => !(state[b.id] || {{}}).category);
  if (!next) return;
  state[next.id] = state[next.id] || {{}};
  state[next.id].category = {json.dumps(list(CATEGORIES))}[n - 1];
  save();
  const card = document.getElementById("card" + next.id);
  card.className = "card done";
  card.querySelector('input[type=radio]:checked') ||
    (card.querySelectorAll('input[type=radio]')[n - 1].checked = true);
  updateCount();
  const after = IDS.find(b => !(state[b.id] || {{}}).category);
  if (after) document.getElementById("card" + after.id)
    .scrollIntoView({{behavior: "smooth", block: "start"}});
}});
restore();
</script>
</body>
</html>
"""
    with open(SHEET, "w", encoding="utf-8") as fh:
        fh.write(page)
    print(f"wrote {os.path.relpath(SHEET, ROOT)} "
          f"({len(bugs)} bugs, {sum(1 for b in bugs if b['moves'])} with a full move history)")
    md_path = os.path.join(ROOT, "results", "review_sheet.md")
    with open(md_path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(sheet_markdown(bugs))
    print(f"wrote {os.path.relpath(md_path, ROOT)}")


if __name__ == "__main__":
    main()
