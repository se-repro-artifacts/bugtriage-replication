"""Mechanical consistency checks on the paper, before a human reads it.

What a script can settle: that every reference resolves, that every citation key exists in the
shared bibliography, that every number macro is defined, that no draft marker or placeholder
survived, that the structural sections are all present, and that the abstract is short enough for
arXiv. What it cannot settle is whether the argument holds, which is why its output says what it did
*not* check.

Usage: python check_paper.py
"""
import json
import os
import re
import sys

ROOT = os.path.join(os.path.dirname(__file__), "..")
TEX = os.path.join(ROOT, "paper", "main.tex")
# The shared bibliography lives above the project during development, but the replication package
# ships a staged copy at paper/references.bib and has no shared directory above it.
BIB_CANDIDATES = [
    os.path.join(ROOT, "..", "..", "shared", "references.bib"),
    os.path.join(ROOT, "paper", "references.bib"),
]
NUMBERS = os.path.join(ROOT, "paper", "tables", "numbers.json")


def main():
    text = open(TEX, encoding="utf-8").read()
    # strip comments so commented-out draft text does not trip the checks
    body = re.sub(r"(?m)(?<!\\)%.*$", "", text)

    # Labels and floats live in the table files that main.tex \inputs, so those have to be scanned
    # too. Checking only main.tex reports every table reference as broken, which is a false alarm
    # that trains the reader to ignore the checker.
    tables_dir = os.path.join(ROOT, "paper", "tables")
    included = ""
    if os.path.isdir(tables_dir):
        for name in sorted(os.listdir(tables_dir)):
            if name.endswith(".tex"):
                included += "\n" + re.sub(r"(?m)(?<!\\)%.*$", "",
                                          open(os.path.join(tables_dir, name), encoding="utf-8").read())
    everything = body + included
    tables_used = len(re.findall(r"\\input\{tables/", body))

    failures, notes = [], []

    # 1. every \ref / \label pair
    labels = set(re.findall(r"\\label\{([^}]+)\}", everything))
    refs = set(re.findall(r"\\ref\{([^}]+)\}", body))
    missing = sorted(refs - labels)
    if missing:
        failures.append(f"\\ref targets with no \\label: {missing}")
    unused = sorted(labels - refs)
    if unused:
        notes.append(f"labels never referenced: {unused}")
    notes.append(f"labels: {len(labels)}, distinct refs: {len(refs)}")

    # 2. every \cite key is in the bibliography
    cites = set()
    for group in re.findall(r"\\cite\{([^}]+)\}", body):
        cites.update(k.strip() for k in group.split(","))
    bib_path = next((p for p in BIB_CANDIDATES if os.path.exists(p)), None)
    bib = open(bib_path, encoding="utf-8").read() if bib_path else ""
    keys = set(re.findall(r"(?m)^@\w+\{([^,]+),", bib))
    unknown = sorted(cites - keys)
    if unknown:
        failures.append(f"\\cite keys missing from references.bib: {unknown}")
    uncited = sorted(keys - cites)
    if uncited:
        notes.append(f"bibliography entries never cited: {uncited}")
    notes.append(f"citations: {len(cites)} distinct, bibliography entries: {len(keys)}")

    # 3. every number macro is defined
    defined = set(json.load(open(NUMBERS, encoding="utf-8")).keys()) if os.path.exists(NUMBERS) else set()
    used = set(re.findall(r"\\n([A-Z][A-Za-z]+)\b", body))
    undefined = sorted(u for u in used if u not in defined)
    if undefined:
        failures.append(f"number macros used but not generated: {undefined}")
    notes.append(f"number macros used: {len(used)} of {len(defined)} generated")

    # 4. no draft markers or placeholders left
    for pattern, label in ((r"TODO", "TODO"), (r"FIXME", "FIXME"), (r"\?\?", "unresolved ??"),
                           (r"\{\{", "placeholder marker"), (r"\bTBD\b", "TBD")):
        hits = len(re.findall(pattern, body))
        if hits:
            failures.append(f"{label} appears {hits} time(s) in the body")
    # commented-out drafts are allowed but are worth knowing about
    commented = len(re.findall(r"(?m)^\s*%.*(TODO|FIXME|DRAFT)", text))
    if commented:
        notes.append(f"{commented} commented-out draft marker(s), not visible in the PDF")

    # 5. structure. The abstract is an environment, not a \section, so it is checked separately.
    wanted = ["Introduction", "Related Work", "Data", "Methods", "Experimental setup", "Results",
              "Error analysis", "Limitations", "Reproducibility", "Conclusion"]
    have = re.findall(r"\\section\{([^}]+)\}", body)
    for section in wanted:
        if not any(section.lower() in h.lower() for h in have):
            failures.append(f"expected a section like '{section}'; found {have}")
    notes.append(f"sections: {have}")

    # 6. abstract length, which arXiv caps
    abstract = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", body, re.S)
    if abstract:
        plain = re.sub(r"\\[a-zA-Z]+\{?|\}", "", abstract.group(1))
        plain = re.sub(r"\s+", " ", plain).strip()
        notes.append(f"abstract: {len(plain)} characters / {len(plain.split())} words "
                     f"(arXiv guidance is under 1920 characters)")
        if len(plain) > 1920:
            failures.append("abstract is longer than arXiv's 1920-character guidance")
    else:
        failures.append("no abstract found")

    # 7. floats, for the record. Counted across the included table files as well.
    notes.append(f"figures: {len(re.findall(r'begin.figure', body))}, "
                 f"table files \\input by main.tex: {tables_used}, "
                 f"table floats found: {len(re.findall(r'begin.table', everything))}, "
                 f"of which full width: {len(re.findall(r'begin.table[*]', everything))}")

    # 8. a control word swallows the space after it, so "\nAgentAurc against" prints as
    # "0.077against". This is invisible in the source and obvious in the PDF, which is exactly the
    # sort of thing a machine should be checking.
    swallowed = [(i, m.group(0)) for i, line in enumerate(body.splitlines(), 1)
                 if not line.lstrip().startswith("%")
                 for m in [re.search(r"\\n[A-Z][A-Za-z]+(\s+)[A-Za-z]", line)] if m]
    if swallowed:
        failures.append(f"number macro followed by a space, which TeX swallows "
                        f"({len(swallowed)} place(s)): add {{}} after the macro")
        for line_no, frag in swallowed[:6]:
            notes.append(f"  line {line_no}: {frag[:40]}")

    print("FAILURES")
    print("  none" if not failures else "\n".join("  - " + f for f in failures))
    print("\nNOTES")
    for note in notes:
        print("  - " + note)
    print("\nNOT CHECKED BY THIS SCRIPT: whether each claim is supported by the table it cites, "
          "whether the prose and the tables agree in substance, and whether the argument is "
          "convincing. Those need a reader.")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
