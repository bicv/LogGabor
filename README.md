[![Binder](https://mybinder.org/badge.svg)](https://mybinder.org/v2/gh/bicv/LogGabor/master)
[![PyPI version](https://badge.fury.io/py/LogGabor.svg)](https://badge.fury.io/py/LogGabor)
[![Research software impact](http://depsy.org/api/package/pypi/LogGabor/badge.svg)](http://depsy.org/package/python/LogGabor)

LogGabor
========

The Log-Gabor function proposed by Field [1987] is an alternative to the Gabor function to efficiently represent edges in natural images. Log-Gabor filters can be constructed with arbitrary bandwidth and the bandwidth can be optimised to produce a filter with minimal spatial extent. We develop here a log-Gabor representation, which is well suited to represent a wide range of natural images.

  ![Comparison of edge function as presented in https://laurentperrinet.github.io/publication/fischer-07-cv](https://laurentperrinet.github.io/publication/fischer-07-cv/figure1.png)

This framework was presented in the following paper by [Sylvain Fischer, Filip Šroubek, Laurent U Perrinet, Rafael Redondo and Gabriel Cristóbal (2007)](https://laurentperrinet.github.io/publication/fischer-07-cv). Examples and documentation is available @ https://pythonhosted.org/LogGabor/ and this package provides with a python implementation.

  ![ScreenShot of the implementation provided in https://laurentperrinet.github.io/publication/fischer-07-cv](https://laurentperrinet.github.io/publication/fischer-07-cv/featured.png)

 Log-Gabor pyramid
 -----------------
 
A log-Gabor pyramid is an oriented multiresolution scheme for images inspired by biology.

To represent the edges of the image at different levels and orientations, we use a multi-scale approach constructing a set of filters of different scales and according to oriented log-Gabor filters. This is represented here by stacking images on a Golden Rectangle Perrinet (2008), that is where the aspect ratio is the golden section ϕ=1+5√2. The level represents coefficients' amplitude, hue corresponds to orientation. We present here the base image on the left and the successive levels of the pyramid in a clockwise fashion (for clarity, we stopped at level 8). Note that here we also use ϕ^2 (that is ϕ+1) as the down-scaling factor so that the pixelwise resolution of the pyramid images correspond across scales.

  ![ScreenShot ](https://laurentperrinet.github.io/publication/perrinet-08-spie/featured.png)

  The Golden Laplacian Pyramid. To represent the edges of the image at different levels, we may use a simple recursive approach constructing progressively a set of images of decreasing sizes, from a base to the summit of a pyramid (see https://laurentperrinet.github.io/publication/perrinet-15-bicv for more details).
