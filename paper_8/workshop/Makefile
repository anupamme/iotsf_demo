TEX = pdflatex
BIB = bibtex
MAIN = main

all: $(MAIN).pdf

$(MAIN).pdf: $(MAIN).tex workshop_refs.bib
	$(TEX) $(MAIN)
	$(BIB) $(MAIN)
	$(TEX) $(MAIN)
	$(TEX) $(MAIN)

clean:
	rm -f $(MAIN).aux $(MAIN).bbl $(MAIN).blg $(MAIN).log $(MAIN).out $(MAIN).pdf

.PHONY: all clean
