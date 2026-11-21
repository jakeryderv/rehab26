# Build the reports. Each top-level .tex file is one report; its PDF lands in dist/.
#   make          build every report
#   make part1    build one
REPORTS := $(basename $(wildcard *.tex))

.PHONY: all clean $(REPORTS)
all: $(REPORTS)

$(REPORTS):
	latexmk -xelatex -interaction=nonstopmode -halt-on-error -outdir=build/$@ $@.tex
	mkdir -p dist && cp build/$@/$@.pdf dist/$@.pdf

clean:
	rm -rf build
