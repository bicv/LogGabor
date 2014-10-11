default: pypi_docs
NAME = LogGabor

edit:
	mvim -p setup.py __init__.py $(NAME).py README.md Makefile requirements.txt

pypi_all: pypi_tags pypi_push pypi_upload pypi_docs
# https://docs.python.org/2/distutils/packageindex.html
pypi_tags:
	git commit -am' tagging for PyPI '
	# in case you wish to delete tags, visit http://wptheming.com/2011/04/add-remove-github-tags/
	git tag 0.1.1 -m "Adds a tag so that we can put this on PyPI."
	git push --tags origin master

pypi_push:
	python setup.py register

pypi_upload:
	python setup.py sdist upload

pypi_docs:
	runipy $(NAME).ipynb  --html  index.html
	zip web.zip index.html
	open http://pypi.python.org/pypi?action=pkg_edit&name=$(NAME)

todo:
	grep -R * (^|#)[ ]*(TODO|FIXME|XXX|HINT|TIP)( |:)([^#]*)

# macros for tests
%.html: %.ipynb
	runipy $< --html $@

%.pdf: %.ipynb
	ipython nbconvert --SphinxTransformer.author='Laurent Perrinet (INT, UMR7289)' --to latex --post PDF $<

# cleaning macros
clean_tmp:
	#find . -name .AppleDouble -type d -exec rm -fr {} \;
	find .  -name *lock* -exec rm -fr {} \;
	rm frioul.*
	rm log-edge-debug.log

clean:
	rm -fr figures/* *.pyc *.py~ build dist

.PHONY: clean
