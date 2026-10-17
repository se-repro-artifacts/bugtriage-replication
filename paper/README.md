# paper/ — LaTeX source for paper 01

## Policy for this directory

`main.tex` is a **skeleton**: section headings and short comments only. Two workspace rules
explain why:

- **Rule 1** — no number may appear that did not come from code that actually ran. Every table is
  `\input` from `tables/`, which `../code/make_tables.py` regenerates from `../results/`. Never
  type a number into a `tabular` by hand; regenerate the table instead.
- **Rule 7** — the author reads and approves every section. Prose is written one section at a
  time and stopped for review, so nothing is drafted ahead of the evidence.

## Regenerate every table

```bash
cd papers/01-bugtriage
.venv/Scripts/python code/baselines_ext.py --dataset data/dataset_all.jsonl --test-split test2026
.venv/Scripts/python code/make_tables.py --split test2026
```

`make_tables.py` refuses to include any `metrics_*mock*.json` file: mock runs only test the
pipeline and must never reach a paper table.

## Compile

The machine had no TeX distribution. **Tectonic** (a single self-contained binary, no
administrator rights needed) was installed to `~\tools\tectonic\`:

```powershell
tectonic -X compile main.tex
```

Tectonic downloads the packages it needs (including `acmart`) on first use and caches them.
Equivalently, any full TeX distribution with `pdflatex` + `bibtex` works:

```bash
pdflatex main && bibtex main && pdflatex main && pdflatex main
```

The submission checklist in `../../shared/arxiv_checklist.md` asks for a local `pdflatex`
compile with no errors and no `??` references; check that with whichever engine is used.

## Build the Chinese version

The Chinese paper is `../论文中文版.md`. It carries `{{Placeholder}}` markers instead of typed
numbers, so it cannot drift away from the tables. `code/make_tables.py` writes both
`tables/numbers.tex` (LaTeX macros used by `main.tex`) and `tables/numbers.json` (the same figures
for the markdown). Rebuild it with:

```powershell
cd papers/01-bugtriage
$md = Get-Content '论文中文版.md' -Raw -Encoding utf8
$j  = Get-Content 'paper\tables\numbers.json' -Raw -Encoding utf8 | ConvertFrom-Json
foreach ($p in $j.PSObject.Properties) {
  $md = $md.Replace('{{' + $p.Name + '}}', ($p.Value -replace '\{,\}', ',')) }
Set-Content '.cache\zh_build.md' -Value $md -Encoding utf8 -NoNewline
pandoc '.cache\zh_build.md' -s -o '.cache\zh_build.html' --toc --toc-depth=2 -c paper/zh_print.css --embed-resources
& 'C:\Program Files\Google\Chrome\Application\chrome.exe' --headless=new --disable-gpu --no-pdf-header-footer --print-to-pdf='论文中文版.pdf' 'file:///E:/Research/.cache/zh_build.html'
```

Three dead ends, recorded so nobody repeats them:

- **LaTeX is not viable for Chinese on this machine.** Tectonic confines file access to the
  directory of the main file, so XeTeX cannot open `C:/Windows/Fonts` at all — by family name, by
  `Path`+`Extension`, or by full path. Compiling with the Fandol faces that ship inside the TeX
  distribution does work, but the PDF's ToUnicode map is wrong: the Chinese renders and cannot be
  copied, searched or read aloud. A browser resolves system fonts by name and writes correct text.
- **`paper/build.py --zh` does not work here.** Chrome refuses to print when spawned from Python in
  this environment — "Missing headless user data directory", or a bare non-zero exit, or a hang —
  while the byte-for-byte identical command run from the shell writes the PDF. The flag is kept for
  a normal machine, not for this one.
- **The `powershell` reachable on `PATH` is a restricted shim** that rejects `Get-Content -Raw` and
  even `-Encoding`. Run the sequence above from the shell the harness provides.

## Target venue

**ISSTA 2027 Research Papers** — Tue 7 to Fri 10 September 2027, Singapore. Re-checked 2026-10-15 at
<https://conf.researchr.org/track/issta-2027/issta-2027-research-papers>:

| Milestone | Date (AoE, UTC-12h) |
|---|---|
| Mandatory abstract | Fri 2027-01-08 |
| Full paper | Mon 2027-01-11 |
| Author response | Mon 2027-03-22 to Thu 2027-03-25 |
| Initial notification (incl. early reject) | Tue 2027-04-20 |
| Major revision submission | Thu 2027-05-20 |
| Final notification | Thu 2027-06-17 |
| Camera ready | Thu 2027-07-22 |

Program co-chairs: Dan Hao (Peking University) and Jacques Klein (University of Luxembourg). The
review model includes a **major revision** round, so a borderline paper can be revised rather than
rejected outright.

On both checks (2026-10-12 and 2026-10-15) the call-for-papers body was still "TBD", so **the page
limit, template and anonymity rules remain unconfirmed**. `main.tex` therefore uses the ACM
`sigconf` style that SIGSOFT venues use. Re-check the CFP and switch to `IEEEtran` if the final CFP
says so. Fallback venue: FSE 2027 workshops (suggested paper deadline 2027-02-11).

The submission is 7 pages in the current two-column `sigconf` layout, which leaves room to expand
inside any plausible limit.
