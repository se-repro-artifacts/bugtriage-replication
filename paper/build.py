"""Build the paper: stage the shared bibliography, then compile with whatever engine is present.

Why a staging step: Tectonic refuses to read files above the main file's directory, so
\\bibliography{../../shared/references} silently produces an empty bibliography. This script
copies ../../shared/references.bib to paper/references.bib first, which is why main.tex says
\\bibliography{references}. The same staging works before a pdflatex build.

Usage:
  python build.py                     # tectonic (default)
  python build.py --engine pdflatex   # needs a full TeX distribution on PATH
"""
import argparse
import glob
import json
import os
import re
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SHARED = os.path.abspath(os.path.join(HERE, "..", "..", "..", "shared", "references.bib"))
STAGED = os.path.join(HERE, "references.bib")
DEFAULT_TECTONIC = r"tectonic"


def find(name):
    return shutil.which(name)


ZH_MD = os.path.abspath(os.path.join(HERE, "..", "论文中文版.md"))
ZH_CSS = os.path.join(HERE, "zh_print.css")
ZH_PANDOC = find("pandoc") or r"~\AppData\Local\Pandoc\pandoc.exe"
ZH_BROWSERS = [r"C:\Program Files\Google\Chrome\Application\chrome.exe",
               r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"]


def build_chinese(out_pdf):
    """Build the Chinese PDF as markdown -> HTML -> browser print-to-pdf.

    Not LaTeX, and the reason is worth recording because it is not obvious. Tectonic confines file
    access to the directory of the main file, so XeTeX cannot open the Windows CJK fonts at all --
    by family name, by Path+Extension, or by full path. With the Fandol faces that ship inside the
    TeX distribution it does compile, but the resulting PDF has a wrong ToUnicode map: the Chinese
    renders and cannot be copied, searched or read by a screen reader, which makes it a defective
    document rather than a working one. A browser resolves system fonts by name and writes correct
    text, so this route is the only one available here that produces an accurate PDF.
    """
    for label, path in (("Chinese source", ZH_MD), ("print stylesheet", ZH_CSS)):
        if not os.path.exists(path):
            print(f"missing {label}: {path}")
            return 1
    if not os.path.exists(ZH_PANDOC):
        print("pandoc not found; it is needed to turn the markdown into HTML")
        return 1
    browser = next((b for b in ZH_BROWSERS if os.path.exists(b)), None)
    if not browser:
        print("no Chromium-based browser found for --print-to-pdf")
        return 1

    # abspath matters twice here. Without it the path keeps a ".." component, and a file:// URL
    # containing ".." is refused by the browser, which then exits non-zero with no message at all.
    out_pdf = os.path.abspath(out_pdf)
    # Intermediates go to the workspace-wide .cache directory. Writing them next to the PDF is what
    # made the browser step fail here while the identical command succeeded by hand, so they are
    # kept out of the document tree.
    cache = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".cache"))
    os.makedirs(cache, exist_ok=True)
    html = os.path.join(cache, "zh_build.html")

    print("$", ZH_PANDOC, os.path.basename(ZH_MD), flush=True)
    # The Chinese source carries {{Placeholder}} markers instead of typed figures, mirroring the
    # LaTeX macros: code/make_tables.py writes the same numbers to numbers.json, so both language
    # versions track the files in results/ as the sweep adds bugs.
    md_in = ZH_MD
    numbers_path = os.path.join(HERE, "tables", "numbers.json")
    if os.path.exists(numbers_path):
        with open(numbers_path, encoding="utf-8") as fh:
            values = json.load(fh)
        tables = {}
        tables_path = os.path.join(HERE, "tables", "zh_tables.json")
        if os.path.exists(tables_path):
            with open(tables_path, encoding="utf-8") as fh:
                tables = json.load(fh)
        text = open(ZH_MD, encoding="utf-8").read()
        missing = set()

        def table_repl(match):
            # whole tables are substituted too, not only single figures: the Chinese paper's tables
            # were hand-copied and silently fell a week behind the results
            key = match.group(1)
            if key not in tables:
                missing.add("TABLE:" + key)
                return match.group(0)
            return tables[key]

        def repl(match):
            key = match.group(1)
            if key not in values:
                missing.add(key)
                return match.group(0)
            # the values carry LaTeX's {,} thousands separator, which markdown should not show
            return str(values[key]).replace("{,}", ",")

        text = re.sub(r"\{\{TABLE:(\w+)\}\}", table_repl, text)
        text = re.sub(r"\{\{(\w+)\}\}", repl, text)
        md_in = os.path.join(cache, "zh_build.md")
        with open(md_in, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(text)
        print(f"  filled {len(values)} figures and {len(tables)} tables into the Chinese source")
        if missing:
            print("  WARNING: placeholders with no value left as-is:", sorted(missing))

    result = subprocess.run([ZH_PANDOC, md_in, "-s", "-o", html, "--toc", "--toc-depth=2",
                             "-c", ZH_CSS, "--embed-resources"],
                            capture_output=True, text=True, errors="replace")
    if result.returncode != 0:
        print(result.stderr[-900:])
        print("pandoc failed")
        return result.returncode

    url = "file:///" + html.replace("\\", "/")
    # A unique profile directory per run: a shared one is exactly what makes a second browser
    # instance wait forever on a lock the first one holds.
    profile = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".cache",
                                           f"chrome-profile-{os.getpid()}"))
    os.makedirs(profile, exist_ok=True)
    base_args = ["--headless=new", "--disable-gpu", "--no-pdf-header-footer"]
    # The order matters. Letting the browser choose its own profile is tried first because it is
    # what works here; forcing a profile is the fallback for when its default (%LOCALAPPDATA%,
    # which this sandbox denies) is unusable and it reports only "Missing headless user data
    # directory" before a bare non-zero exit.
    attempts = []
    for exe in [browser] + [b for b in ZH_BROWSERS if b != browser]:
        if not os.path.exists(exe):
            continue
        attempts.append([exe] + base_args + [f"--print-to-pdf={out_pdf}", url])
        attempts.append([exe] + base_args + [f"--user-data-dir={profile}",
                                             f"--print-to-pdf={out_pdf}", url])
    ps = shutil.which("pwsh") or shutil.which("powershell")
    for cmd in attempts:
        print("$", cmd[0], " ".join(cmd[1:3]), flush=True)
        # The browser step is delegated to PowerShell on purpose. Spawned directly from this process
        # Chrome returns non-zero with "Missing headless user data directory" or hangs, while the
        # byte-for-byte identical command run by PowerShell writes the PDF. That was established by
        # testing both repeatedly; this is ugly, and it is the only form that works here.
        line = "& " + " ".join("'" + str(part) + "'" for part in cmd)
        try:
            result = subprocess.run([ps, "-NoProfile", "-Command", line],
                                    capture_output=True, text=True, errors="replace",
                                    timeout=300, cwd=os.path.dirname(out_pdf))
        except subprocess.TimeoutExpired:
            print("   timed out after 300 s; trying the next option")
            continue
        if os.path.exists(out_pdf):
            print(f"OK  {out_pdf}  ({os.path.getsize(out_pdf)} bytes)")
            return 0
        last = ((result.stderr or "").strip().splitlines() or [""])[-1]
        print("   no PDF:", last[:150])

    print("every browser attempt failed")
    return 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", default="tectonic", choices=["tectonic", "pdflatex"])
    ap.add_argument("--tex", default="main.tex")
    ap.add_argument("--zh", action="store_true",
                    help="build the Chinese PDF from ../论文中文版.md instead")
    a = ap.parse_args()
    if a.zh:
        return build_chinese(os.path.join(HERE, "..", "论文中文版.pdf"))
    tex = os.path.join(HERE, a.tex)
    stem = a.tex[:-4] if a.tex.endswith(".tex") else a.tex

    if os.path.exists(SHARED):
        shutil.copyfile(SHARED, STAGED)
        n_entries = sum(1 for line in open(SHARED, encoding="utf-8") if line.startswith("@"))
        print(f"staged {SHARED} -> {STAGED} ({n_entries} bib entries)")
    elif os.path.exists(STAGED):
        # The replication package ships the staged copy and has no shared directory above it, so a
        # missing SHARED is normal there rather than an error.
        print(f"no shared bibliography at {SHARED}; using the local {os.path.basename(STAGED)}")
    else:
        print("WARNING: no bibliography at", SHARED, "and none staged at", STAGED)
        return 1

    if a.engine == "tectonic":
        exe = DEFAULT_TECTONIC if os.path.exists(DEFAULT_TECTONIC) else find("tectonic")
        if not exe:
            print("tectonic not found; set it up or pass --engine pdflatex")
            return 1
        steps = [[exe, "-X", "compile", a.tex, "--keep-intermediates"]]
    else:
        exe = find("pdflatex")
        bib = find("bibtex")
        if not exe or not bib:
            print("pdflatex/bibtex not found on PATH")
            return 1
        steps = [[exe, "-interaction=nonstopmode", a.tex],
                 [bib, stem],
                 [exe, "-interaction=nonstopmode", a.tex],
                 [exe, "-interaction=nonstopmode", a.tex]]

    output = []
    for step in steps:
        print("$", " ".join(step))
        # Capture rather than inherit: Tectonic writes LaTeX warnings to stderr and keeps no .log
        # file even with --keep-logs, so this text is the only place an undefined reference or a
        # duplicate label would show up.
        result = subprocess.run(step, cwd=HERE, capture_output=True, text=True, errors="replace")
        sys.stdout.write(result.stdout)
        sys.stderr.write(result.stderr)
        output.append(result.stdout + result.stderr)
        if result.returncode != 0:
            print("build FAILED at:", step[0])
            return result.returncode

    pdf = os.path.join(HERE, stem + ".pdf")
    bbl = os.path.join(HERE, stem + ".bbl")
    if not os.path.exists(pdf):
        print("no PDF produced")
        return 1
    print(f"OK  {pdf}  ({os.path.getsize(pdf)} bytes)")
    if os.path.exists(bbl):
        items = sum(1 for line in open(bbl, encoding="utf-8", errors="replace")
                    if line.startswith("\\bibitem"))
        print(f"bibliography entries typeset: {items}")

    text = "\n".join(output)
    bad = [l.strip() for l in text.splitlines()
           if "undefined" in l.lower() or "multiply" in l.lower()]
    print(f"undefined-reference / duplicate-label warnings: {len(bad)}")
    for line in bad[:8]:
        print("  ", line)
    over = sum(1 for l in text.splitlines() if "Overfull" in l)
    under = sum(1 for l in text.splitlines() if "Underfull" in l)
    print(f"cosmetic spacing warnings: {over} overfull, {under} underfull")
    leftovers = [os.path.basename(p) for p in glob.glob(os.path.join(HERE, "bibcheck*"))]
    if leftovers:
        print("note: temporary check files still present:", leftovers)
    return 0


if __name__ == "__main__":
    sys.exit(main())
