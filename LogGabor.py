# -*- coding: utf8 -*-
"""
LogGabor

See http://pythonhosted.org/LogGabor

"""
__author__ = "(c) Laurent Perrinet INT - CNRS"
import numpy as np
import scipy.ndimage as nd
from scipy.fftpack import fft2, fftshift, ifft2, ifftshift

class LogGabor:
    """
    defines a LogGabor transform.

    Its envelope is equivalent to a log-normal probability distribution on the frequency axis, and von-mises on the radial axis.


    """
    def __init__(self, im):
        """
        initializes the LogGabor structure

        """
        self.pe = im.pe
        self.im = im
        self.n_x = im.n_x
        self.n_y = im.n_y

        self.f_x, self.f_y = self.im.f_x, self.im.f_y
        self.f = self.im.f

        self.color = self.enveloppe_color(alpha=self.pe.alpha)

    ## LOW LEVEL OPERATIONS

    def enveloppe_color(self, alpha):
        # 0.0, 1.0, 2.0 are resp. white, pink, red/brownian envelope
        # (see http://en.wikipedia.org/wiki/1/f_noise )
        if alpha == 0:
            return 1.
        else:
            f_radius = np.zeros(self.f.shape)
            f_radius = self.f**alpha
            f_radius[(self.n_x-1)//2 + 1 , (self.n_y-1)//2 + 1 ] = np.inf
            return 1. / f_radius

    def band(self, sf_0, B_sf, correct=False):
        # selecting a donut (the ring around a prefered frequency)
        #
        if sf_0 == 0.: return 1.
        # see http://en.wikipedia.org/wiki/Log-normal_distribution
        env = 1./self.f*np.exp(-.5*(np.log(self.f/sf_0)**2)/B_sf**2)
        return env

    def orientation(self, theta, B_theta):
        # selecting one direction,  theta is the mean direction, B_theta the spread
        # we use a von-mises distribution on the orientation
        # see http://en.wikipedia.org/wiki/Von_Mises_distribution
        angle = np.arctan2(self.f_y, self.f_x)
        cos_angle = np.cos(angle-theta)
        enveloppe_orientation = np.exp(cos_angle/B_theta**2)
#        As shown in:
#        http://www.csse.uwa.edu.au/~pk/research/matlabfns/PhaseCongruency/Docs/convexpl.html
#        this simple bump allows (without the symmetric) to code both symmetric and anti-symmetric parts
        return enveloppe_orientation

    def loggabor(self, u, v, sf_0, B_sf, theta, B_theta):
        env = self.band(sf_0, B_sf) * \
                self.orientation(theta, B_theta) * \
                self.im.trans(u*1., v*1.) * self.color
        # normalizing energy:
        env /= np.sqrt((np.abs(env)**2).mean())
        # in the case a a single bump (see radius()), we should compensate the fact that the distribution gets complex:
        env *= np.sqrt(2.)
        return env

    def show_loggabor(self, u, v, sf_0, B_sf, theta, B_theta, phase=0.):
        FT_lg = self.loggabor(u, v, sf_0, B_sf, theta, B_theta)
        fig, a1, a2 = self.im.show_FT(FT_lg * np.exp(-1j*phase))
        return fig, a1, a2

def _test():
    import doctest
    doctest.testmod()
#####################################
#
if __name__ == '__main__':
    _test()

    #### Main
    """
    Some examples of use for the class

    """
    from pylab import imread
    # whitening
    image = imread('database/gris512.png')[:,:,0]
    lg = LogGabor(image.shape)


