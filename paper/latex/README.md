# ASA Paper (LaTeX)

## Build (local)

```bash
cd paper/latex
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

Overleaf: upload the `paper/latex/` folder and set `main.tex` as the root file.
