"""Rasterise a PDF so the layout can actually be looked at instead of inferred from warnings.

`Overfull \\hbox` warnings say a box is wider than its column, but not whether that pushes a table
into the neighbouring column, over the page edge, or just into the margin. Rendering the page is the
only way to tell, and it is what "the tables overlap the text" requires.

Usage: python render_pdf.py <pdf> <outdir> [dpi] [--pages 1,2,3]
"""
import os
import sys

import pypdfium2 as pdfium


def main():
    if len(sys.argv) < 3:
        raise SystemExit(__doc__)
    pdf_path, outdir = sys.argv[1], sys.argv[2]
    dpi = 110
    pages = None
    for arg in sys.argv[3:]:
        if arg.startswith("--pages"):
            pages = [int(p) for p in arg.split("=", 1)[1].split(",")]
        else:
            dpi = int(arg)
    os.makedirs(outdir, exist_ok=True)
    doc = pdfium.PdfDocument(pdf_path)
    wanted = pages or range(1, len(doc) + 1)
    scale = dpi / 72.0
    for number in wanted:
        page = doc[number - 1]
        image = page.render(scale=scale).to_pil()
        out = os.path.join(outdir, f"page{number:02d}.png")
        image.save(out)
        print(f"page {number}: {image.size[0]}x{image.size[1]} -> {out}")
    print(f"{len(doc)} pages total")


if __name__ == "__main__":
    main()
