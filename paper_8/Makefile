LATEX = pdflatex -interaction=nonstopmode

.PHONY: pdf clean

pdf:
	$(LATEX) main.tex
	bibtex main
	$(LATEX) main.tex
	$(LATEX) main.tex

clean:
	rm -f main.aux main.bbl main.blg main.out main.log main.pdf main.synctex.gz
