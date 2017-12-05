[![Binder](https://mybinder.org/badge.svg)](https://mybinder.org/v2/gh/bicv/LogGabor/master)
[![PyPI version](https://badge.fury.io/py/LogGabor.svg)](https://badge.fury.io/py/LogGabor)
[![Research software impact](http://depsy.org/api/package/pypi/LogGabor/badge.svg)](http://depsy.org/package/python/LogGabor)

LogGabor
========

A log-Gabor pyramid is an oriented multiresolution scheme for images inspired by biology.

Examples and documentation is available @ https://pythonhosted.org/LogGabor/

  ![ScreenShot of the implementation provided in http://invibe.net/LaurentPerrinet/Publications/Fischer07cv](http://invibe.net/cgi-bin/index.cgi/Figures/Fischer07cv/FigureUn?action=AttachFile&do=get&target=loggabor.png)

  Log-Gabor filters. To represent the edges of the image at different levels and orientations, we use a multi-scale approach constructing a set of filters of different scales and according to oriented log-Gabor filters. This is represented here by stacking images on a Golden Rectangle Perrinet (2008), that is where the aspect ratio is the golden section ϕ=1+5√2. The level represents coefficients' amplitude, hue corresponds to orientation. We present here the base image on the left and the successive levels of the pyramid in a clockwise fashion (for clarity, we stopped at level 8). Note that here we also use ϕ^2 (that is ϕ+1) as the down-scaling factor so that the pixelwise resolution of the pyramid images correspond across scales.

This package provides with a python implementation.

More information is available @ http://nbviewer.ipython.org/github/bicv/LogGabor/blob/master/LogGabor.ipynb

  ![ScreenShot ](http://invibe.net/cgi-bin/index.cgi/Figures/Perrinet08spie/FigureDeux?action=AttachFile&do=get&target=GoldenPyramid.png)

  The Golden Laplacian Pyramid. To represent the edges of the image at different levels, we may use a simple recursive approach constructing progressively a set of images of decreasing sizes, from a base to the summit of a pyramid (see http://invibe.net/LaurentPerrinet/Publications/Perrinet15bicv for more details).
