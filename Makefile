default: pypi_docs
NAME = LogGabor
version = 0.3 # << to change in setup.py
 
edit:
	mvim -p setup.py src/__init__.py src/$(NAME).py README.md Makefile requirements.txt

pypi_all: pypi_tags pypi_push pypi_upload
# https://docs.python.org/2/distutils/packageindex.html
pypi_tags:
	git commit -am' tagging for PyPI '
	# in case you wish to delete tags, visit http://wptheming.com/2011/04/add-remove-github-tags/
	git tag $(version) -m "Adds a tag so that we can put this on PyPI."
	git push --tags origin master

pypi_push:
	python setup.py register

pypi_upload:
	python setup.py sdist upload

pypi_docs:
	#rm web.zip
	#ipython nbconvert --to html $(NAME).ipynb
	#mv $(NAME).html index.html
	#runipy $(NAME).ipynb  --html  index.html
	zip web.zip index.html
	open https://pypi.python.org/pypi?action=pkg_edit&name=$NAME

install_dev:
	pip3 uninstall -y $(NAME) ; pip3 install -e .
todo:
	grep -R * (^|#)[ ]*(TODO|FIXME|XXX|HINT|TIP)( |:)([^#]*)


console:
	open -a /Applications/Utilities/Console.app/ log-sparseedges-debug.log

# macros for tests
index.html: $(NAME).ipynb
	runipy $(NAME).ipynb -o
	ipython nbconvert --SphinxTransformer.author='Laurent Perrinet (INT, UMR7289)' --to html index.html $(NAME).ipynb

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
