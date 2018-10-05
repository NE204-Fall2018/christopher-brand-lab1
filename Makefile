manuscript = report
latexopt = -file-line-error -halt-on-error

# Build the PDF of the lab report from the source files
$(manuscript).pdf: $(manuscript).tex text/*.tex references.bib images/*.png
	pdflatex $(latexopt) $(manuscript).tex
	bibtex $(manuscript).aux
	bibtex $(manuscript).aux
	pdflatex $(latexopt) $(manuscript).tex
	pdflatex $(latexopt) $(manuscript).tex

# Get/download necessary data
data :
	# Uncomment below to download WARNING: data set is large
	#curl -L -o data/test_input.h5 https://berkeley.box.com/s/z4q1us6e6ao4179dmmbnrc3hasj0se29/test_input.h5 
	#curl -L -o data/cs137_co60.h5 https://berkeley.box.com/s/z4q1us6e6ao4179dmmbnrc3hasj0se29/cs137_co60.h5 
	#curl -L -o data/co57.h5 https://berkeley.box.com/s/z4q1us6e6ao4179dmmbnrc3hasj0se29/co57.h5 
	#curl -L -o data/ba133.h5 https://berkeley.box.com/s/z4q1us6e6ao4179dmmbnrc3hasj0se29/ba133.h5 
	#curl -L -o data/am241.h5 https://berkeley.box.com/s/z4q1us6e6ao4179dmmbnrc3hasj0se29/am241.h5 

# Validate that downloaded data is not corrupted
validate :

# Run tests on analysis code
test :

# Automate running the analysis code
analysis :
	cd code/ && python ProduceGraphs.py
parameters :
	cd code/ && python find_k.py/ && find_m.py/ && find_Tau.py

clean :
	rm -f *.aux *.log *.bbl *.lof *.lot *.blg *.out *.toc *.run.xml *.bcf
	rm -f text/*.aux
	rm $(manuscript).pdf
	rm code/*.pyc

# Make keyword for commands that don't have dependencies
.PHONY : test data validate analysis clean
