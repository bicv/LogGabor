# -*- coding: utf8 -*-
from __future__ import division
"""
LogGabor

See http://pythonhosted.org/LogGabor

"""
__author__ = "(c) Laurent Perrinet INT - CNRS"
import numpy as np
from SLIP import Image
from numpy.fft import fft2, fftshift, ifft2, ifftshift
import scipy.optimize as opt


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
        self.sf_0 = .5 * (1 - 1/self.n_levels) / np.logspace(0, self.n_levels-1, self.n_levels, base=self.pe.base_levels, endpoint=False)
        self.theta = np.linspace(-np.pi/2, np.pi/2, self.pe.n_theta+1)[1:]
        self.oc = (self.pe.N_X * self.pe.N_Y * self.pe.n_theta * self.n_levels) #(1 - self.pe.base_levels**-2)**-1)
        if self.pe.use_cache is True:
            self.cache = {'band':{}, 'orientation':{}}
        # self.envelope = np.zeros((self.pe.N_X, self.pe.N_Y))

    def linear_pyramid(self, image):

        C = np.empty((self.pe.N_X, self.pe.N_Y, self.pe.n_theta, self.n_levels), dtype=np.complex)
        for i_sf_0, sf_0 in enumerate(self.sf_0):
            for i_theta, theta in enumerate(self.theta):
                FT_lg = self.loggabor(0, 0, sf_0=sf_0, B_sf=self.pe.B_sf,
                                    theta=theta, B_theta=self.pe.B_theta)
                C[:, :, i_theta, i_sf_0] = self.FTfilter(image, FT_lg, full=True)
        return C

    def golden_pyramid(self, z, mask=False):
        """
        The Golden Laplacian Pyramid.
        To represent the edges of the image at different levels, we may use a simple recursive approach constructing progressively a set of images of decreasing sizes, from a base to the summit of a pyramid. Using simple down-scaling and up-scaling operators we may approximate well a Laplacian operator. This is represented here by stacking images on a Golden Rectangle, that is where the aspect ratio is the golden section $\phi \eqdef \frac{1+\sqrt{5}}{2}$. We present here the base image on the left and the successive levels of the pyramid in a clockwise fashion (for clarity, we stopped at level $8$). Note that here we also use $\phi^2$ (that is $\phi+1$) as the down-scaling factor so that the resolution of the pyramid images correspond across scales. Note at last that coefficient are very kurtotic: most are near zero, the distribution of coefficients has long tails.

        """
        import matplotlib.pyplot as plt

        phi = (np.sqrt(5)+1.)/2. # golden ratio
        opts= {'vmin':0., 'vmax':1., 'interpolation':'nearest', 'origin':'upper'}
        fig_width = 13
        fig = plt.figure(figsize=(fig_width, fig_width/phi), frameon=True)
        xmin, ymin, size = 0, 0, 1.
        axs = []
        for i_sf_0 in range(len(self.sf_0)):
            ax = fig.add_axes((xmin/phi, ymin, size/phi, size), axisbg='w')
            ax.axis(c='w', lw=1)
            plt.setp(ax, xticks=[], yticks=[])
            im_RGB = np.zeros((self.pe.N_X, self.pe.N_Y, 3))
            for i_theta, theta_ in enumerate(self.theta):
                im_abs = np.absolute(z[:, :, i_theta, i_sf_0])
                RGB = np.array([.5*np.sin(2*theta_ + 2*i*np.pi/3)+.5 for i in range(3)])
                im_RGB += im_abs[:,:, np.newaxis] * RGB[np.newaxis, np.newaxis, :]

            im_RGB /= im_RGB.max()
            ax.imshow(1-im_RGB, **opts)
            ax.grid(b=False, which="both")
            if mask:
                linewidth_mask = 1 #
                from matplotlib.patches import Ellipse
                circ = Ellipse((.5*self.pe.N_Y, .5*self.pe.N_X),
                                self.pe.N_Y-linewidth_mask, self.pe.N_X-linewidth_mask,
                                fill=False, facecolor='none', edgecolor = 'black', alpha = 0.5, ls='dashed', lw=linewidth_mask)
                ax.add_patch(circ)
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
    def band(self, sf_0, B_sf, force=False):
        """
        Returns the radial frequency envelope:

        Selects a preferred spatial frequency ``sf_0`` and a bandwidth ``B_sf``.

        """
        if sf_0 == 0.:
            return 1.
        elif self.pe.use_cache and not force:
            tag = str(sf_0) + '_' + str(B_sf)
            try:
                return self.cache['band'][tag]
            except:
                if self.pe.verbose>50: print('doing band cache for tag ', tag)
                self.cache['band'][tag] = self.band(sf_0, B_sf, force=True)
                return self.cache['band'][tag]
        else:
            # see http://en.wikipedia.org/wiki/Log-normal_distribution
            env = 1./self.f*np.exp(-.5*(np.log(self.f/sf_0)**2)/B_sf**2)
        return env

    def orientation(self, theta, B_theta, force=False):
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
        elif self.pe.use_cache and not force:
            tag = str(theta) + '_' + str(B_theta)
            try:
                return self.cache['orientation'][tag]
            except:
                if self.pe.verbose>50: print('doing orientation cache for tag ', tag)
                self.cache['orientation'][tag] = self.orientation(theta, B_theta, force=True)
                return self.cache['orientation'][tag]
        else: # non pathological case
            # As shown in:
            #  http://www.csse.uwa.edu.au/~pk/research/matlabfns/PhaseCongruency/Docs/convexpl.html
            # this single bump allows (without the symmetric) to code both symmetric
            # and anti-symmetric parts in one shot.
            cos_angle = np.cos(self.f_theta-theta)
            enveloppe_orientation = np.exp(cos_angle/B_theta**2)
        return enveloppe_orientation

    ## MID LEVEL OPERATIONS
    def loggabor(self, u, v, sf_0, B_sf, theta, B_theta, preprocess=True):
        """

        Note that the convention for coordinates follows that of matrices: the origin is at the top left of the image, and coordinates are first the rows (vertical axis, going down) then the columns (horizontal axis, going right).

        """

        env = np.multiply(self.band(sf_0, B_sf), self.orientation(theta, B_theta))
        if not(u==0.) and not(v==0.): # bypass translation whenever none is needed
              env = env.astype(np.complex128) * self.trans(u*1., v*1.)
        if preprocess : env *= self.f_mask # retina processing
        # normalizing energy:
        env /= np.sqrt((np.abs(env)**2).mean())
        # in the case a a single bump (see ``orientation``), we should compensate the fact that the distribution gets complex:
        env *= np.sqrt(2.)
        return env

    def show_loggabor(self, u, v, sf_0, B_sf, theta, B_theta, title='', phase=0.):
        FT_lg = self.loggabor(u, v, sf_0, B_sf, theta, B_theta)
        fig, a1, a2 = self.show_FT(FT_lg * np.exp(-1j*phase))
        return fig, a1, a2

    ##ADDED FUNTIONS

    def invert(self, FT_image, full=False):
        if full:
            return ifft2(ifftshift(FT_image))
        else:
            return ifft2(ifftshift(FT_image)).real

    def argmax(self, C):
        """
        Returns the ArgMax from C by returning the
        (x_pos, y_pos, theta, scale)  tuple

        """
        ind = np.absolute(C).argmax()
        return np.unravel_index(ind, C.shape)



    def twoD_Gaussian(self,xy ,x_pos, y_pos, theta, sf_0, phase, B_sf):

        FT_lg = self.loggabor(x_pos, y_pos, sf_0=np.absolute(sf_0), B_sf=B_sf, theta=theta, B_theta=self.pe.B_theta)
        FT_lg = FT_lg * np.exp(1j * phase)

        return self.invert(FT_lg).ravel()

    def twoD_Gaussian2(self, x_in, B_theta):

        FT_lg = self.loggabor(x_in[0], x_in[1], sf_0=x_in[3], B_sf=x_in[5], theta=x_in[2], B_theta=B_theta)
        FT_lg = FT_lg * np.exp(1j * x_in[4])

        return self.invert(FT_lg).ravel()

    def twoD_Gaussian3(self, x_pos, y_pos, theta, sf_0, phase, B_sf, B_theta):

        FT_lg = self.loggabor(x_pos, y_pos, sf_0=np.absolute(sf_0), B_sf=B_sf, theta=theta, B_theta=B_theta)
        FT_lg = FT_lg * np.exp(1j * phase)

        return self.invert(FT_lg).ravel()

    def LogGaborFit(self ,patch, N_X, N_Y):

        self.set_size((N_X, N_Y))
        self.pe.N_X = N_X
        self.pe.N_Y = N_Y

        x = np.arange(self.pe.N_X)
        y = np.arange(self.pe.N_Y)
        xy = np.meshgrid(x, y)


        C = self.linear_pyramid(np.reshape(patch, (N_X, N_Y)))
        idx = self.argmax(C)
        #initial guess from Matching Pursuit
        initial_guess = [idx[0], idx[1], self.theta[idx[2]], self.sf_0[idx[3]], 0, self.pe.B_sf]
        #Adjusting parameters
        popt, pcov = opt.curve_fit(self.twoD_Gaussian, xy, patch.ravel(), p0=initial_guess)
        popt2, pcov2 = opt.curve_fit(self.twoD_Gaussian2, popt, patch.ravel(), p0=(self.pe.B_theta))

        # define bigger patch to avoid artifacts

        self.pe.N_X = N_X + N_X // 2
        self.pe.N_Y = N_Y + N_Y // 2
        self.set_size((int(N_X + N_X / 2), int(N_Y + N_Y / 2)))
        x = np.arange(self.pe.N_X)
        y = np.arange(self.pe.N_Y)
        xy2 = np.meshgrid(x, y)
        popt[0] = popt[0] + N_X / 4
        popt[1] = popt[1] + N_Y / 4

        popt3 = np.concatenate((popt, popt2))

        patch_fit = np.reshape(self.twoD_Gaussian3(*popt3), (N_X + N_X // 2, N_Y + N_Y // 2))

        patch_fit = patch_fit[np.arange(N_X // 4, N_X + N_X // 4), :]

        patch_fit = patch_fit[:, np.arange(N_Y // 4, N_X + N_Y // 4)]

        popt3[0] = popt3[0] - N_X / 4
        popt3[1] = popt3[1] - N_Y / 4

        return patch_fit.ravel(), popt3

        #return np.zeros((1,int( N_X * N_Y))), np.zeros((1, 7))

    def LogGaborFit_dictionary(self, dictx, verbose = False, get_unfitted = False, whoswho = False):

        if whoswho:
            names=[]
            names.append('dictx_fit_param[:,0] = x0')
            names.append('dictx_fit_param[:,1] = y0')
            names.append('dictx_fit_param[:,2] = theta')
            names.append('dictx_fit_param[:,3] = sf_0')
            names.append('dictx_fit_param[:,4] = Phase')
            names.append('dictx_fit_param[:,5] = B_sf')
            names.append('dictx_fit_param[:,6] = B_theta')

        dictx_fit = np.zeros_like(dictx)
        dictx_fit_param = np.zeros((dictx_fit.shape[0], 7))
        idx_unfitted = []

        for i in range(dictx.shape[0]):

            if verbose:
                print("Fitting patch % 3i /  % 3i"
                    %(i + 1, dictx.shape[0]))

            try:
                dictx_fit[i, :], dictx_fit_param[i, :] = self.LogGaborFit(dictx[i, :], int(np.sqrt(dictx.shape[1])),
                                                                          int(np.sqrt(dictx.shape[1])))
            except:
                if verbose:
                    print("Couldn't fit patch number % 3i" %i)
                dictx_fit[i, :] = np.zeros((1, dictx.shape[1]))
                dictx_fit_param[i, :] = np.zeros((1, 7))
                idx_unfitted.append(i)

        if get_unfitted:
            if whoswho:
                return dictx_fit, dictx_fit_param, idx_unfitted, names
            else:
                return dictx_fit, dictx_fit_param, idx_unfitted
        else:
            if whoswho:
                return dictx_fit, dictx_fit_param, names
            else:
                return dictx_fit, dictx_fit_param


    ##END ADDED FUNTIONS

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
