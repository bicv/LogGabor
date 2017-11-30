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

    def argmax(self, C):
        """
        Returns the ArgMax from C by returning the
        (x_pos, y_pos, theta, scale)  tuple

        >>> C = np.random.randn(10, 10, 5, 4)
        >>> x_pos, y_pos, theta, scale = mp.argmax(C)
        >>> C[x_pos][y_pos][theta][scale] = C.max()

        """
        ind = np.absolute(C).argmax()
        return np.unravel_index(ind, C.shape)

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
    def loggabor(self, x_pos, y_pos, sf_0, B_sf, theta, B_theta, preprocess=True):
        """
        Returns the envelope of a LogGabor

        Note that the convention for coordinates follows that of matrices: the origin is at the top left of the image, and coordinates are first the rows (vertical axis, going down) then the columns (horizontal axis, going right).

        """

        env = np.multiply(self.band(sf_0, B_sf), self.orientation(theta, B_theta))
        if not(x_pos==0.) and not(y_pos==0.): # bypass translation whenever none is needed
              env = env.astype(np.complex128) * self.trans(x_pos*1., y_pos*1.)
        if preprocess : env *= self.f_mask # retina processing
        # normalizing energy:
        env /= np.sqrt((np.abs(env)**2).mean())
        # in the case a a single bump (see ``orientation``), we should compensate the fact that the distribution gets complex:
        env *= np.sqrt(2.)
        return env

    def loggabor_image(self, x_pos, y_pos, theta, sf_0, phase, B_sf, B_theta):
        FT_lg = self.loggabor(x_pos, y_pos, sf_0=sf_0, B_sf=B_sf, theta=theta, B_theta=B_theta)
        FT_lg = FT_lg * np.exp(1j * phase)
        return self.invert(FT_lg, full=False)

    def show_loggabor(self, u, v, sf_0, B_sf, theta, B_theta, title='', phase=0.):
        FT_lg = self.loggabor(u, v, sf_0, B_sf, theta, B_theta)
        fig, a1, a2 = self.show_FT(FT_lg * np.exp(-1j*phase))
        return fig, a1, a2


class LogGaborFit(LogGabor):
    """
    Defines a  framework to fit a LogGabor.

    """
    def __init__(self, pe):
        LogGabor.__init__(self, pe)
        self.init_logging(name='LogGaborFit')
        self.init()

    def LogGaborFit(self, patch):
        from lmfit import Parameters, minimize, fit_report
        N_X, N_Y = patch.shape

        #initial guess is the one corresponding to the Maximum Likelihood Estimate over the linear pyramid

        C = self.linear_pyramid(patch) # np.reshape(patch, (N_X, N_Y)))
        idx = self.argmax(C)
        fit_params = Parameters()
        fit_params.add('x_pos', value=idx[0], min=0, max=N_X)
        fit_params.add('y_pos', value=idx[1], min=0, max=N_Y)
        fit_params.add('theta', value=self.theta[idx[2]], min=-np.pi/2, max=np.pi/2)
        fit_params.add('sf_0', value=self.sf_0[idx[3]], min=0.001)
        fit_params.add('phase', value=np.angle(C[idx]))
        fit_params.add('B_theta', value=self.pe.B_sf, min=0.001)
        fit_params.add('B_sf', value=self.pe.B_theta, min=0.001)

        out = minimize(self.residual, fit_params, kws={'data':patch}, nan_policy='omit')

        self.set_size((N_X + N_X // 2, N_Y + N_Y // 2))

        patch_fit = self.loggabor_image(**out.params)
        patch_fit = patch_fit[:N_X, :N_Y]

        self.set_size((N_X, N_Y))

        return patch_fit.ravel(), out.params

    def residual(self, pars, data):  # =None, eps=None):
        # unpack parameters:
        #  extract .value attribute for each parameter
        parvals = pars.valuesdict()
        x_pos = parvals['x_pos']
        y_pos = parvals['y_pos']
        theta = parvals['theta']  # % np.pi
        B_theta = parvals['B_theta']
        sf_0 = np.abs(parvals['sf_0'])
        B_sf = parvals['B_sf']
        phase = parvals['phase']  # % (2*np.pi)

        model = self.loggabor_image(x_pos, y_pos, theta, sf_0, phase, B_sf, B_theta)

        #energy norm
        model /= np.sqrt(np.sum(model ** 2))
        data /= np.sqrt(np.sum(data ** 2))

        return (model - data).ravel()


    def LogGaborFit_dictionary(self, dictx, verbose=False, get_unfitted=False, whoswho=False):

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

        N_X = int(np.sqrt(dictx.shape[1]))
        N_Y = N_X

        self.set_size((N_X, N_Y))
        self.pe.N_X = N_X
        self.pe.N_Y = N_Y

        for i in range(dictx.shape[0]):

            if verbose:
                print("Fitting patch % 3i /  % 3i" %(i + 1, dictx.shape[0]))

            try:
                N_X, N_Y = int(np.sqrt(dictx.shape[1])), int(np.sqrt(dictx.shape[1]))
                dict_ = np.reshape(dictx[i, :], (N_X, N_Y))
                #print(dictx_fit[i, :].shape)#, dictx_fit_param[i, :].shape, self.LogGaborFit(dict_))
                patch_fit, out_params = self.LogGaborFit(dict_)
                dictx_fit[i, :] = patch_fit.ravel()
                dictx_fit_param[i, :] = np.array([out_params[key].value for key in list(out_params)])
            except ValueError:
                print(out_params)
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
