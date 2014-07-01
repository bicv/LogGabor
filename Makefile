default: test_

# DECORRELATION figures
White_pdf = figures/whitening.pdf figures/whitening_corr.pdf figures/whitening_atick.pdf
White_src = $(White_pdf:.pdf=.svg)

Test_pdf =  test_Image.pdf test_LogGabor.pdf
Test_src = $(Test_pdf:.pdf=.ipynb)
Test_html = $(Test_pdf:.html=.ipynb)
test_: $(Test_html) LogGabor.py

#exp_src = experiment_whitening.py experiment_edges.py experiment_animals.py
#experiment_: $(exp_src:.py=)

$(White_src): experiment_whitening.py
	python experiment_whitening.py

white: $(White_pdf)

linux_edit:
	texmaker LogGabor.py &
	gedit Makefile

web: experiment_whitening.py experiment_edges.py LogGabor.py
	zip web.zip LogGabor.py $(Test_html)

# https://docs.python.org/2/distutils/packageindex.html
pypi_tags:
	git tag 0.1.3 -m "Adds a tag so that we can put this on PyPI."
	git push --tags origin master

pypi_push:
	python setup.py register

pypi_upload:
	python setup.py sdist bdist_wininst upload

pypi_docs: index.html
	zip web.zip index.html
	open http://pypi.python.org/pypi?action=pkg_edit&name=LogGabor

todo:
	grep -R * (^|#)[ ]*(TODO|FIXME|XXX|HINT|TIP)( |:)([^#]*)
# macros for tests
%.html: %.ipynb
	runipy $< --html $@

test_%.pdf: test_%.ipynb
	ipython nbconvert --SphinxTransformer.author='Laurent Perrinet (INT, UMR7289)' --to latex --post PDF $<

experiment_%: experiment_%.py LogGabor.py
	python  $<

linux_view:
	evince $(Test_pdf) &

mac_view:
	open $(Test_pdf) &

# cleaning macros
clean_tmp:
	#find . -name .AppleDouble -type d -exec rm -fr {} \;
	find .  -name *lock* -exec rm -fr {} \;
	rm frioul.*
	rm log-edge-debug.log

clean_SVM:
	rm frioul.* figures/*png figures/*SVM*txt mat/*hist* mat/*SVM* mat/*lock figures/*lock
clean:
	rm -f figures/* white*.mat $(latexfile).pdf *.pyc *.py~ *.npy

.PHONY: clean
