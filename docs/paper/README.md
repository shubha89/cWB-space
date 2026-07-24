# cWB-LISA pipeline paper draft

Files:

- `cwb_lisa_pipeline.tex`: REVTeX/APS manuscript draft based on the supplied `apssamp.tex` template.
- `cwb_lisa_pipeline.bib`: bibliography for the draft.

Build from this directory with the installed Tectonic toolchain:

```bash
tectonic cwb_lisa_pipeline.tex
```

Alternatively, with a traditional LaTeX installation:

```bash
pdflatex cwb_lisa_pipeline
bibtex cwb_lisa_pipeline
pdflatex cwb_lisa_pipeline
pdflatex cwb_lisa_pipeline
```

Technical implementation details are intentionally kept in the appendices so the main paper stays focused on the pipeline design, TDI/coherence logic, and development roadmap.
