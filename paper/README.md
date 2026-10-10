# Paper

`main.tex` is the complete paper in one file. It compiles with pdflatex (twice) on Overleaf or locally, with `IEEEtran.cls` and the figure PDFs in this folder.

The sections are kept in `parts/`. Tables are not typed by hand: `python gen/fill.py` writes `build/part_*.tex` with every table filled from `../results/summary/`. `main.tex` was made by inlining those files. `gen/make_architecture.py` and `gen/make_overview.py` draw Fig. 1 and Fig. 2, and `../figures/make_v3_figures.py` draws the result figures.

`response/response.tex` lists the reviewer objections of the internal review and how the revision answers each.
