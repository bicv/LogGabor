default: test_

# DECORRELATION figures
White_pdf = figures/whitening.pdf figures/whitening_corr.pdf figures/whitening_atick.pdf
White_src = $(White_pdf:.pdf=.svg)

Test_pdf =  test_Image.pdf test_LogGabor.pdf
Test_src = $(Test_pdf:.pdf=.ipynb)
Test_html = $(Test_pdf:.html=.ipynb)
test_: $(Test_pdf) LogGabor.py

#exp_src = experiment_whitening.py experiment_edges.py experiment_animals.py
#experiment_: $(exp_src:.py=)

$(White_src): experiment_whitening.py
	python experiment_whitening.py

white: $(White_pdf)

linux_edit:
	texmaker LogGabor.py &
	gedit Makefile

web: experiment_whitening.py experiment_edges.py LogGabor.py
	ipython nbconvert --SphinxTransformer.author='Laurent Perrinet (INT, UMR7289)' --to HTML $<
	zip web.zip LogGabor.py $(Test_html)

todo:
	grep -R * (^|#)[ ]*(TODO|FIXME|XXX|HINT|TIP)( |:)([^#]*)
# macros for tests
test_%.pdf: test_%.ipynb LogGabor.py
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
