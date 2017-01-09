# -*- coding: utf8 -*-
from __future__ import division
"""
LogGabor

See http://pythonhosted.org/LogGabor

"""
__author__ = "(c) Laurent Perrinet INT - CNRS"
import numpy as np

from SLIP import Image

class LogGabor(Image):
    """
    Defines a LogGabor framework by defining a ``loggabor`` function which return the envelope of a log-Gabor filter.

    Its envelope is equivalent to a log-normal probability distribution on the frequency axis, and von-mises on the radial axis.


    """
    def __init__(self, pe):
        Image.__init__(self, pe)
        self.init_logging(name='LogGabor')
        self.init()

    ## PYRAMID
    def init(self):
        Image.init(self)

        self.n_levels = int(np.log(np.max((self.pe.N_X, self.pe.N_Y)))/np.log(self.pe.base_levels))
        self.sf_0 = 1. / np.logspace(1, self.n_levels, self.n_levels, base=self.pe.base_levels)
        self.theta = np.linspace(-np.pi/2, np.pi/2, self.pe.n_theta+1)[1:]
        self.oc = (self.pe.N_X * self.pe.N_Y * self.pe.n_theta * self.n_levels) #(1 - self.pe.base_levels**-2)**-1)

    def linear_pyramid(self, image):
        # theta_bin = (self.theta + np.hstack((self.theta[-1]-np.pi, self.theta[:-1]))) /2
        #self.theta[:-1] + self.theta[1:]) / 2. # middles
        # binedges_theta = np.hstack((theta_bin, theta_bin[0]+np.pi))
        # width = binedges_theta[1:] - binedges_theta[:-1]
        # B_thetas = self.pe.B_theta * width / (np.pi/self.pe.n_theta)
        # print(B_thetas)

        C = np.empty((self.pe.N_X, self.pe.N_Y, self.pe.n_theta, self.n_levels), dtype=np.complex)
        for i_sf_0, sf_0 in enumerate(self.sf_0):
            for i_theta, theta in enumerate(self.theta):
                FT_lg = self.loggabor(0, 0, sf_0=sf_0, B_sf=self.pe.B_sf,
                                    theta=theta, B_theta=self.pe.B_theta)
                C[:, :, i_theta, i_sf_0] = self.FTfilter(image, FT_lg, full=True)
        return C

    def golden_pyramid(self, z):
        """
        The Golden Laplacian Pyramid.
        To represent the edges of the image at different levels, we may use a simple recursive approach constructing progressively a set of images of decreasing sizes, from a base to the summit of a pyramid. Using simple down-scaling and up-scaling operators we may approximate well a Laplacian operator. This is represented here by stacking images on a Golden Rectangle, that is where the aspect ratio is the golden section $\phi \eqdef \frac{1+\sqrt{5}}{2}$. We present here the base image on the left and the successive levels of the pyramid in a clockwise fashion (for clarity, we stopped at level $8$). Note that here we also use $\phi^2$ (that is $\phi+1$) as the down-scaling factor so that the resolution of the pyramid images correspond across scales. Note at last that coefficient are very kurtotic: most are near zero, the distribution of coefficients has long tails.

        """
        import matplotlib.pyplot as plt

        phi = (np.sqrt(5) +1.)/2. # golden ratio
        opts= {'vmin':0., 'vmax':1., 'interpolation':'nearest', 'origin':'upper'}
        fig_width = 13
        fig = plt.figure(figsize=(fig_width, fig_width/phi))
        xmin, ymin, size = 0, 0, 1.
        axs = []
        for i_sf_0, sf_0_ in enumerate(self.sf_0):
            ax = fig.add_axes((xmin/phi, ymin, size/phi, size), axisbg='w')
            ax.axis(c='b', lw=0)
            plt.setp(ax, xticks=[], yticks=[])
            im_RGB = np.zeros((self.pe.N_X, self.pe.N_Y, 3))
            for i_theta, theta_ in enumerate(self.theta):
                im_abs = np.absolute(z[:, :, i_theta, i_sf_0])
                RGB = np.array([.5*np.sin(2*theta_ + 2*i*np.pi/3)+.5 for i in range(3)])
                im_RGB += im_abs[:,:, np.newaxis] * RGB[np.newaxis, np.newaxis, :]

            im_RGB /= im_RGB.max()
            ax.imshow(im_RGB, **opts)
            #a.grid(False)
            ax.grid(b=False, which="both")
            i_orientation = np.mod(i_sf_0, 4)
            if i_orientation==0:
                xmin += size
                ymin += size/phi**2
            elif i_orientation==1:
                xmin += size/phi**2
                ymin += -size/phi
            elif i_orientation==2:
                xmin += -size/phi
            elif i_orientation==3:
                ymin += size
            axs.append(ax)
            size /= phi

        return fig, axs

    ## LOW LEVEL OPERATIONS
    def band(self, sf_0, B_sf):
        # selecting a donut (the ring around a prefered frequency)
        if sf_0 == 0.: return 1.
        # see http://en.wikipedia.org/wiki/Log-normal_distribution
        env = 1./self.f*np.exp(-.5*(np.log(self.f/sf_0)**2)/B_sf**2)
        return env

    def orientation(self, theta, B_theta):
        """
    Returns the orientation envelope:
    We use a von-Mises distribution on the orientation:
    - mean orientation is ``theta`` (in radians),
    - ``B_theta`` is the bandwidth (in radians). It is equal to the standard deviation of the Gaussian
    envelope which approximate the distribution for low bandwidths. The Half-Width at Half Height is
    given by approximately np.sqrt(2*B_theta_**2*np.log(2)).

        # selecting one direction,  theta is the mean direction, B_theta the spread
        # we use a von-mises distribution on the orientation
        # see http://en.wikipedia.org/wiki/Von_Mises_distribution
        """
        if B_theta is np.inf: # for large bandwidth, returns a strictly flat envelope
            enveloppe_orientation = 1.
        else: # non pathological case
            cos_angle = np.cos(self.f_theta-theta)
            enveloppe_orientation = np.exp(cos_angle/B_theta**2)
#        As shown in:
#        http://www.csse.uwa.edu.au/~pk/research/matlabfns/PhaseCongruency/Docs/convexpl.html
#        this single bump allows (without the symmetric) to code both symmetric and anti-symmetric parts
        return enveloppe_orientation

    ## MID LEVEL OPERATIONS
    def loggabor(self, u, v, sf_0, B_sf, theta, B_theta, preprocess=True):
        """

        Note that the convention for coordinates follows that of matrices: the origin is at the top left of the image, and coordinates are first the rows (vertical axis, going down) then the columns (horizontal axis, going right).

        """

        env = self.band(sf_0, B_sf) * \
              self.orientation(theta, B_theta) * \
              self.trans(u*1., v*1.)
        if preprocess : env *= self.f_mask
        # normalizing energy:
        env /= np.sqrt((np.abs(env)**2).mean())
        # in the case a a single bump (see ``orientation``), we should compensate the fact that the distribution gets complex:
        env *= np.sqrt(2.)
        return env

    def show_loggabor(self, u, v, sf_0, B_sf, theta, B_theta, title='', phase=0.):
        FT_lg = self.loggabor(u, v, sf_0, B_sf, theta, B_theta)
        fig, a1, a2 = self.show_FT(FT_lg * np.exp(-1j*phase))
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
    lg = LogGabor('default_param.py')
    from SLIP import imread
    image = imread('database/lena512.png')[:,:,0]
    lg.set_size(image)
