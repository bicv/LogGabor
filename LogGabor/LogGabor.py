# -*- coding: utf8 -*-
"""
LogGabor

See http://invibe.net/LaurentPerrinet/Publications/Perrinet11sfn

"""
__author__ = "(c) Laurent Perrinet INT - CNRS"
import numpy as np
import scipy.ndimage as nd
from scipy.fftpack import fft2, fftshift, ifft2, ifftshift

def init_pylab():
    ############################  FIGURES   ########################################
    from NeuroTools import check_dependency
    HAVE_MATPLOTLIB = check_dependency('matplotlib')
    if HAVE_MATPLOTLIB:
        import matplotlib
        matplotlib.use("Agg") # agg-backend, so we can create figures without x-server (no PDF, just PNG etc.)
    HAVE_PYLAB = check_dependency('pylab')
    if HAVE_PYLAB:
        import pylab
        # parameters for plots
        fig_width_pt = 500.  # Get this from LaTeX using \showthe\columnwidth
        inches_per_pt = 1.0/72.27               # Convert pt to inches
        fig_width = fig_width_pt*inches_per_pt  # width in inches
        fontsize = 8
        # pe.edge_scale_chevrons, line_width = 64., .75
        params = {'backend': 'Agg',
                 'origin': 'upper',
                  'font.family': 'serif',
                  'font.serif': 'Times',
                  'font.sans-serif': 'Arial',
                  'text.usetex': True,
        #          'mathtext.fontset': 'stix', #http://matplotlib.sourceforge.net/users/mathtext.html
                  'interpolation':'nearest',
                  'axes.labelsize': fontsize,
                  'text.fontsize': fontsize,
                  'legend.fontsize': fontsize,
                  'figure.subplot.bottom': 0.17,
                  'figure.subplot.left': 0.15,
                  'ytick.labelsize': fontsize,
                  'xtick.labelsize': fontsize,
                  'savefig.dpi': 100,
                }
        pylab.rcParams.update(params)

def adjust_spines(ax,spines):
    for loc, spine in ax.spines.iteritems():
        if loc in spines:
            spine.set_position(('outward',1)) # outward by 10 points
#            spine.set_smart_bounds(True)
        else:
            spine.set_color('none') # don't draw spine

    # turn off ticks where there is no spine
    if 'left' in spines:
        ax.yaxis.set_ticks_position('left')
    else:
        # no yaxis ticks
        ax.yaxis.set_ticks([])

    if 'bottom' in spines:
        ax.xaxis.set_ticks_position('bottom')
    else:
        # no xaxis ticks
        ax.xaxis.set_ticks([])
############################  FIGURES   ########################################
class Image:
    """
    Collects image processing routines for one given image size:
        - load_in_database : loads a random image in a folder and
        - patch : takes a random patch of the correct size
        - energy, normalize,
        - fourier_grid : defines a useful grid for generating filters in FFT
        - show_FT : displays the envelope and impulse response of a filter
        - convert / invert : go from one side to the other of the fourier transform
        - trans : translation filter in Fourier space

    """
    def __init__(self, pe):
        """
        initializes the Image class

        """
        self.pe = pe
        self.n_x = pe.N_X # n_x
        self.n_y = pe.N_X # n_y
#        self.url_database = url_database
        if self.n_x%2 or self.n_y%2: print('having images of uneven dimensions will fail')

        self.f_x, self.f_y = self.fourier_grid()
        self.f = np.sqrt(self.f_x**2 + self.f_y**2)
#        print ' hello ' , self.f[(self.n_x-1)//2 + 1, (self.n_y-1)//2 + 1], self.f[(self.n_x-1)//2 , (self.n_y-1)//2 ]
        self.f[(self.n_x-1)//2 + 1, (self.n_y-1)//2 + 1] = 1e-12 # np.inf #
#        self.f[(self.n_x-1)//2 , (self.n_y-1)//2 ] = 1e-12 # np.inf

        # TODO: the max is here 1. / in MotionClouds we use Nyquist
        self.f_norm = self.f / np.max((self.n_x, self.n_y))

    def full_url(self, url_database, basedir='database'):
        import os
        return os.path.join(basedir, url_database)

    def list_database(self, url_database, basedir='database'):
        import os
        try:
            filelist = os.listdir(self.full_url(url_database))
        except:
            print('failed opening database %s' % url_database)
        #TODO
        for garbage in ['.AppleDouble', '.DS_Store']:
            if garbage in filelist:
                filelist.remove(garbage)
        return filelist

    def load_in_database(self, url_database, basedir='database', i_image=None, filename=None, verbose=True):
        """
        Loads a random image from directory url_database

        """
        filelist = self.list_database(url_database=url_database)

        if filename is None:
            if i_image is None:
                i_image = np.random.randint(0, len(filelist))
            else:
                i_image = i_image % len(filelist)

            if verbose: print 'Using image ', filelist[i_image]
            filename = filelist[i_image]

        from pylab import imread
        import os
        image = imread(os.path.join(self.full_url(url_database), filename)) * 1.
        if image.ndim == 3:
            image = image.sum(axis=2)
        return image, filename

    def patch(self, url_database, i_image=None, filename=None, croparea=None, threshold=0.2, verbose=True):
        """
        takes a subimage of size s (a tuple)

        does not accept if energy is relatively below a threshold (flat image)

        """
#         if not(filename is None):
#             image, filename = self.load_in_database(url_database, filename=filename, verbose=verbose)
#         else:
        image, filename = self.load_in_database(url_database, i_image=i_image, filename=filename, verbose=verbose)

        if (croparea is None):
            image_size_h, image_size_v = image.shape
            if self.n_x > image_size_h or self.n_y > image_size_v:
                raise Exception('Patch size too big for the image in your DB')
            elif self.n_x == image_size_h or self.n_y == image_size_v:
                return image, filename, [0, self.n_x, 0, self.n_y]
            else:
                energy = image[:].std()
                energy_ = 0

                while energy_ < threshold*energy:
                    #if energy_ > 0: print 'dropped patch'
                    x_rand = int(np.ceil((image_size_h-self.n_x)*np.random.rand()))
                    y_rand = int(np.ceil((image_size_v-self.n_y)*np.random.rand()))
                    image_ = image[(x_rand):(x_rand+self.n_x), (y_rand):(y_rand+self.n_y)]
                    energy_ = image_[:].std()

                if verbose: print 'Cropping @ [l,r,b,t]: ', [x_rand, x_rand+self.n_x, y_rand, y_rand+self.n_y]

                croparea = [x_rand, x_rand+self.n_x, y_rand, y_rand+self.n_y]
        image_ = image[croparea[0]:croparea[1], croparea[2]:croparea[3]]
        image_ -= image_.mean()
        return image_, filename, croparea

    def energy(self, image):
        #       see http://fseoane.net/blog/2011/computing-the-vector-norm/
        return  np.mean(image.ravel()**2)

    def normalize(self, image, center=True, use_max=True):
        image_ = image.copy()
        if center: image_ -= np.mean(image_.ravel())
        if use_max:
            if np.max(np.abs(image_.ravel()))>0: image_ /= np.max(np.abs(image_.ravel()))
        else:
            if self.energy(image_)>0: image_ /= self.energy(image_)**.5
        return image_

    #### filter definition
    def fourier_grid(self):
        """
            use that function to define a reference frame for envelopes in Fourier space.
            In general, it is more efficient to define dimensions as powers of 2.

        """

        # From the numpy doc:
        # (see http://docs.scipy.org/doc/numpy/reference/routines.fft.html#module-numpy.fft )
        # The values in the result follow so-called “standard” order: If A = fft(a, n),
        # then A[0] contains the zero-frequency term (the mean of the signal), which
        # is always purely real for real inputs. Then A[1:n/2] contains the positive-frequency
        # terms, and A[n/2+1:] contains the negative-frequency terms, in order of
        # decreasingly negative frequency. For an even number of input points, A[n/2]
        # represents both positive and negative Nyquist frequency, and is also purely
        # real for real input. For an odd number of input points, A[(n-1)/2] contains
        # the largest positive frequency, while A[(n+1)/2] contains the largest negative
        # frequency. The routine np.fft.fftfreq(A) returns an array giving the frequencies
        # of corresponding elements in the output. The routine np.fft.fftshift(A) shifts
        # transforms and their frequencies to put the zero-frequency components in the
        # middle, and np.fft.ifftshift(A) undoes that shift.
        #
        return np.mgrid[(-self.n_x//2):((self.n_x-1)//2 + 1), (-self.n_y//2):((self.n_y-1)//2 + 1)]

#     def expand_complex(self, FT, hue=False):
#         if hue:
#             image_temp = np.zeros((FT.shape[0], FT.shape[1], 4))
#             import matplotlib.cm as cm
#             angle = np.angle(FT)/2./np.pi+.5
#             print 'angle ', angle.min(), angle.max()
#             alpha = np.abs(FT)
#             alpha /= alpha.max()
#             print 'alpha ', alpha.min(), alpha.max()
#             image_temp = cm.hsv(angle)#, alpha=alpha)
#             print image_temp.shape, image_temp.min(), image_temp.max()
#         else:
#             image_temp = 0.5 * np.ones((FT.shape[0], FT.shape[1], 3))
#             FT_ = self.normalize(FT)
#             print 'real ', FT_.real.min(), FT_.real.max()
#             print 'imag ', FT_.imag.min(), FT_.imag.max()
#             image_temp[:,:,0] = 0.5 + 0.5 * FT_.real # * (FT_.real>0) #np.angle(FT)/2./np.pi+.5 #
# #            alpha = np.abs(FT)
# #            alpha /= alpha.max()
#             image_temp[:,:,1] = 0.5
#             image_temp[:,:,2] = 0.5 + 0.5 * FT_.imag #  * (FT_.imag>0)  #alpha
#         return image_temp

    def show_FT(self, FT, axis=False):#,, phase=0. do_complex=False
        N_X, N_Y = FT.shape
        image_temp = self.invert(FT)#, phase=phase)
        import pylab
#         origin : [‘upper’ | ‘lower’], optional, default: None
#         Place the [0,0] index of the array in the upper left or lower left corner of the axes. If None, default to rc image.origin.
#         extent : scalars (left, right, bottom, top), optional, default: None
#         Data limits for the axes. The default assigns zero-based row, column indices to the x, y centers of the pixels.
        fig = pylab.figure(figsize=(12,6))
        a1 = fig.add_subplot(121)
        a2 = fig.add_subplot(122)
        a1.imshow(np.absolute(FT), cmap=pylab.cm.hsv, origin='upper')
        a2.imshow(image_temp/np.abs(image_temp).max(), vmin=-1, vmax=1, cmap=pylab.cm.gray, origin='upper')
        if not(axis):
            pylab.setp(a1, xticks=[], yticks=[])
            pylab.setp(a2, xticks=[], yticks=[])
        a1.axis([0, N_X, N_Y, 0])
        a2.axis([0, N_X, N_Y, 0])
        return fig, a1, a2

    def convert(self, image):
        return fftshift(fft2(image))

    def invert(self, FT_image, full=False):#, phase=np.pi/2):
        if full:
            return ifft2(ifftshift(FT_image)) # *np.exp(1j*phase)
        else:
            return ifft2(ifftshift(FT_image)).real

    def FTfilter(self, image, FT_filter, full=False):
        FT_image = self.convert(image) * FT_filter
        return self.invert(FT_image, full=full)

    def trans(self, u, v):
#        return np.exp(-1j*2*np.pi*(u*self.f_x + v*self.f_y))
        return np.exp(-1j*2*np.pi*(u/self.n_x*self.f_x + v/self.n_y*self.f_y))

    def translate(self, image, vec, preshift=False):
        """
        Translate image by vec (in pixels)

        """
        u, v = vec
        u, v = u * 1., v * 1.

        if preshift:
            # first translate by the integer value
            image = np.roll(np.roll(image, np.int(u), axis=0), np.int(v), axis=1)
            u -= np.int(u)
            v -= np.int(v)

        # sub-pixel translation
        return self.FTfilter(image, self.trans(u, v))
#
#     def coco(self, image, filter_, normalize=True):
#         """
#         Returns the correlation coefficient
#
#         """
#         from scipy.signal import correlate2d
#         coco = correlate2d(image, filter_, mode='same')
#         if normalize:
#             coco /= np.sqrt(self.energy(image)*self.energy(filter_))
#
#         return coco

    def olshausen_whitening_filt(self, f_0=.16, alpha=1.4, N=0.01):
        """
        Returns the whitening filter used by (Olshausen, 98)

        f_0 = 200 / 512

        /!\ you will have some problems at dewhitening without a low-pass

        """

        K_ols = (N**2 + self.f_norm**2)**.5 * self.low_pass(f_0=f_0, alpha=alpha)
        K_ols /= np.max(K_ols)

        return  K_ols

    def low_pass(self, f_0, alpha):
        """
        Returns the low_pass filter used by (Olshausen, 98)

        parameters from Atick (p.240)
        f_0 = 22 c/deg in primates: the full image is approx 45 deg
        alpha makes the aspect change (1=diamond on the vert and hor, 2 = anisotropic)

        """
        return np.exp(-(self.f_norm/f_0)**alpha)

    def whitening_filt(self, size=(512, 512),
                             url_database='Yelmo',
                             n_learning=400,
                             N=1.,
                             f_0=.8, alpha=1.4,
                             N_0=.5,
                             recompute=False):
        """
        Returns the average correlation filter in FT space.

        Computes the average power spectrum = FT of cross-correlation, the mean decorrelation
        is given for instance by (Attick, 92).

        """
        #import shelve # http://docs.python.org/library/shelve.html
        #results = shelve.open('white'+ str(size[0]) + '-' + str(size[1]) + '.mat')
        try:
            K = np.load('white'+ str(size[0]) + '-' + str(size[1]) + '.npy')
            if recompute:
                print('Recomputing the whitening filter')
#             boingk
            #    results.remove('K')
            #K = results['K']

        except:
            print ' Learning the whitening filter'
            power_spectrum = 0. # power spectrum
            for i_learning in range(n_learning):
                image, filename, croparea = self.patch(url_database, verbose=False)
                image = self.normalize(image) #TODO : is this fine?
                power_spectrum += np.abs(fft2(image))**2

            power_spectrum = fftshift(power_spectrum)

    ##        from scipy import mgrid
    ##        fx, fy = mgrid[-1:1:1j*size[0],-1:1:1j*size[1]]
    ##        rho = np.sqrt(fx**2+fy**2)
    ##        power_spectrum = N**2 / (rho**2 + N**2)

            power_spectrum /= np.mean(power_spectrum)
            # formula from Attick:
            M = np.sqrt(power_spectrum / (N**2 + power_spectrum)) * self.low_pass(f_0=f_0, alpha=alpha)
            K = M / np.sqrt(M**2 * (N**2 + power_spectrum) + N_0**2)
     #       K = 1 / np.sqrt( N**2 + power_spectrum)  * low_pass(f_0 = f_0, alpha = alpha)
            K /= np.max(K) # normalize energy :  DC is one <=> xcorr(0) = 1

            np.save('white'+ str(size[0]) + '-' + str(size[1]) + '.npy', K)

            #results['K'] = K
            #results['power_spectrum'] = power_spectrum
        #results.close()

        return K

    def whitening(self, image):
        """
        Returns the whitened image
        """
        K = self.whitening_filt(size=image.shape, recompute=False)

        if not(K.shape == image.shape):
            K = self.whitening_filt(size=image.shape, recompute=True)

        return self.FTfilter(image, K)

    def dewhitening(self, white):
        """
        Returns the dewhitened image

        """
        K = self.whitening_filt(white.shape)

        return self.FTfilter(white, 1./K)


    def retina(self, image):
        """
        A dummy retina processsing


        """

#        TODO: log-polar transform with openCV
        white = self.whitening(image)
        white = self.normalize(white) # mean = 0, std = 1
        return white

class LogGabor:
    """
    defines a LogGabor transform.

    Its envelope is equivalent to a log-normal probability distribution on the frequncy axis, and von-mises on the radial axis.


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
#         TODO : this is only ok for Motion Clouds..
#        env = 1./self.f*np.exp(-.5*(np.log(self.f/sf_0)**2)/(np.log((sf_0+B_sf)/sf_0)**2))
##        one should find that env[(self.n_x-1)//2 + 1, (self.n_y-1)//2 + 1] = 0.
#        print ' Debug: ' , env[(self.n_x-1)//2 + 1, (self.n_y-1)//2 + 1], env[(self.n_x-1)//2 , (self.n_y-1)//2 ]
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

class MatchingPursuit:
    """
    defines a MatchingPursuit algorithm

    """
    def __init__(self, lg):
        """
        initializes the LogGabor structure

        """
        self.pe = lg.pe
        self.MP_alpha = self.pe.MP_alpha

        self.lg = lg
        self.im = lg.im
        self.n_x = lg.n_x
        self.n_y = lg.n_y

        self.base_levels = self.pe.base_levels
        self.n_levels = int(np.log(np.max((self.n_x, self.n_y)))/np.log(self.base_levels)) #  self.pe.n_levels
        self.MP_alpha = self.pe.MP_alpha

        self.sf_0 = lg.n_x / np.logspace(1, self.n_levels, self.n_levels, base=self.base_levels)

        self.n_theta = self.pe.n_theta
        self.theta_ = np.linspace(0., np.pi, self.n_theta, endpoint=False)
        self.B_theta = self.pe.B_theta
        self.B_sf = self.pe.B_sf
        self.N = self.pe.N
        self.do_whitening = self.pe.do_whitening
        self.do_mask = self.pe.do_mask
        if self.do_mask:
            X, Y = np.mgrid[-1:1:1j*self.n_x, -1:1:1j*self.n_y]
            self.mask = (X**2 + Y**2) < 1.

    def run(self, image, verbose=False):
        edges = np.zeros((5, self.N), dtype=np.complex)
        image_ = image.copy()
        if self.do_whitening: image_ = self.im.whitening(image_)
        C = self.init(image_)
        for i_edge in range(self.N):
            # MATCHING
            ind_edge_star = self.argmax(C)
            # recording
            if verbose: print 'Max activity  : ', np.absolute(C[ind_edge_star]), ' phase= ', np.angle(C[ind_edge_star], deg=True), ' deg,  @ ', ind_edge_star
            edges[:, i_edge] = np.array([ind_edge_star[0]*1., ind_edge_star[1]*1., self.theta_[ind_edge_star[2]], self.sf_0[ind_edge_star[3]], self.MP_alpha * C[ind_edge_star]])
            # PURSUIT
            C = self.backprop(C, ind_edge_star)
#            if verbose: print 'Residual activity : ',  C[ind_edge_star]
        return edges, C
#
    def init(self, image):
        C = np.empty((self.n_x, self.n_y, self.n_theta, self.n_levels), dtype=np.complex)
        for i_sf_0, sf_0 in enumerate(self.sf_0):
            for i_theta, theta in enumerate(self.theta_):
                FT_lg = self.lg.loggabor(0, 0, sf_0=sf_0, B_sf=self.B_sf,
                                    theta=theta, B_theta=self.B_theta)
                C[:, :, i_theta, i_sf_0] = self.im.FTfilter(image, FT_lg, full=True)
                if self.do_mask: C[:, :, i_theta, i_sf_0] *= self.mask
        return C

    def reconstruct(self, edges):
#        Fimage = np.zeros((self.n_x, self.n_y), dtype=np.complex)
        image = np.zeros((self.n_x, self.n_y))
#        print edges.shape, edges[:, 0]
        for i_edge in range(edges.shape[1]):#self.N):
            # TODO : check that it is correct when we remove alpha when making new MP
            image += self.im.invert(edges[4, i_edge] * self.lg.loggabor(
                                                                        edges[0, i_edge].real, edges[1, i_edge].real,
                                                                        theta=edges[2, i_edge].real, B_theta=self.B_theta,
                                                                        sf_0=edges[3, i_edge].real, B_sf=self.B_sf,
                                                                        ),
                                    full=False)
        return image

    def argmax(self, C):
        """
        Returns the ArgMax from C by returning the
        (x_pos, y_pos, theta, scale)  tuple

        >>> C = np.random.randn(10, 10, 5, 4)
        >>> C[x_pos][y_pos][level][level] = C.max()

        """
        ind = np.absolute(C).argmax()
        return np.unravel_index(ind, C.shape)

    def backprop(self, C, edge_star):
        """
        Removes edge_star from the activity

        """
        C_star = self.MP_alpha * C[edge_star]
        FT_lg_star = self.lg.loggabor(edge_star[0]*1., edge_star[1]*1., sf_0=self.sf_0[edge_star[3]],
                         B_sf=self.B_sf,#_ratio*self.sf_0[edge_star[3]],
                    theta= self.theta_[edge_star[2]], B_theta=self.B_theta)
        lg_star = self.im.invert(C_star*FT_lg_star, full=False)

        for i_sf_0, sf_0 in enumerate(self.sf_0):
            for i_theta, theta in enumerate(self.theta_):
                FT_lg = self.lg.loggabor(0, 0, sf_0=sf_0, B_sf=self.B_sf, theta=theta, B_theta=self.B_theta)
                C[:, :, i_theta, i_sf_0] -= self.im.FTfilter(lg_star, FT_lg, full=True)
                if self.do_mask: C[:, :, i_theta, i_sf_0] *= self.mask
        return C

    def adapt(self, edges):
        # TODO : implement a COMP adaptation of the thetas and scales tesselation of Fourier space
        pass

    def show_edges(self, edges, fig=None, a=None, image=None, norm=True,
                   color='auto', v_min=-1., v_max=1., show_phase=False, gamma=1., pedestal=.2, mappable=False):
        """
        Shows the quiver plot of a set of edges, optionally associated to an image.

        """
        import pylab
        import matplotlib.cm as cm
        if fig==None:
            fig = pylab.figure(figsize=(self.pe.figsize_edges, self.pe.figsize_edges))
        if a==None:
            border = 0.0
            a = fig.add_axes((border, border, 1.-2*border, 1.-2*border), axisbg='w')
        a.axis(c='b', lw=0)

        if color == 'black' or color == 'redblue' or color in['brown', 'green', 'blue']: #cocir or chevrons
            linewidth = self.pe.line_width_chevrons
            scale = self.pe.scale_chevrons
        else:
            linewidth = self.pe.line_width
            scale = self.pe.scale

        opts= {'extent': (0, self.n_x, self.n_y, 0),
               'cmap': cm.gray,
               'vmin':v_min, 'vmax':v_max, 'interpolation':'nearest', 'origin':'upper'}
#         origin : [‘upper’ | ‘lower’], optional, default: None
#         Place the [0,0] index of the array in the upper left or lower left corner of the axes. If None, default to rc image.origin.
#         extent : scalars (left, right, bottom, top), optional, default: None
#         Data limits for the axes. The default assigns zero-based row, column indices to the x, y centers of the pixels.
        if not(image == None):
#             if image.ndim==2: opts['cmap'] = cm.gray
            if norm: image = self.im.normalize(image, center=True, use_max=True)
            a.imshow(image, **opts)
        else:
            a.imshow([[v_max]], **opts)
        if edges.shape[1] > 0:
            from matplotlib.collections import LineCollection#, EllipseCollection
            import matplotlib.patches as patches
            # draw the segments
            segments, colors, linewidths = list(), list(), list()

            X, Y, Theta, Sf_0 = edges[1, :].real+.5, edges[0, :].real+.5, np.pi -  edges[2, :].real, edges[3, :].real
            weights = edges[4, :]

            #show_phase, pedestal = False, .2 # color edges according to phase or hue? pedestal value for alpha when weights= 0

    #        print X, Y, Theta, Sf_0, weights, scale_
    #        print 'Min theta ', Theta.min(), ' Max theta ', Theta.max()
#            weights = np.absolute(weights)/(np.abs(weights)).max()
            weights = weights/(np.abs(weights)).max()

            for x, y, theta, sf_0, weight in zip(X, Y, Theta, Sf_0, weights):
                u_, v_ = np.cos(theta)*scale/sf_0*self.n_x, np.sin(theta)*scale/sf_0*self.n_y
                segment = [(x - u_, y - v_), (x + u_, y + v_)]
                segments.append(segment)
                if color=='auto':
                    if show_phase:
                        #colors.append(cm.hsv(np.angle(weight), alpha=pedestal + (1. - pedestal)*weight**gamma))#))
                        colors.append(cm.hsv(0., alpha=pedestal + (1. - pedestal)*weight**gamma))#)) # HACK
                    else: colors.append(cm.hsv((theta % np.pi)/np.pi, alpha=pedestal + (1. - pedestal)*weight))#))
                elif color == 'black':
                    colors.append((0, 0, 0, 1))# black
                elif color == 'green': # figure 1DE
                    colors.append((0.05, 0.5, 0.05, np.abs(weight)**gamma))
                elif color == 'blue': # figure 1DE
                    colors.append((0.05, 0.05, 0.5, np.abs(weight)**gamma))
                elif color == 'brown': # figure 1DE
                    colors.append((0.5, 0.05, 0.05, np.abs(weight)**gamma))
                else: # geisler maps etc...
                    colors.append(((np.sign(weight)+1)/2, 0, (1-np.sign(weight))/2, np.abs(weight)**gamma))#weight*(1-weight)))# between red and blue
                linewidths.append(linewidth) # *weight thinning byalpha...

            # TODO : put circle in front
            n_ = np.sqrt(self.n_x**2+self.n_y**2)
            for x, y, theta, sf_0, weight in zip(X, Y, Theta, Sf_0, weights):
                if color=='auto':
                    if show_phase:
                        #fc = cm.hsv(np.angle(weight), alpha=pedestal + (1. - pedestal)*weight**gamma)
                        fc = cm.hsv(0., alpha=pedestal + (1. - pedestal)*weight**gamma) # HACK
                    else:
                        fc = cm.hsv((theta % np.pi)/np.pi, alpha=pedestal + (1. - pedestal)*weight**gamma)
                elif color == 'black':
                    fc = (0, 0, 0, 1)# black
                elif color == 'green': # figure 1DE
                    fc = (0.05, 0.5, 0.05, np.abs(weight)**gamma)
                elif color == 'blue': # figure 1DE
                    fc = (0.05, 0.05, 0.5, np.abs(weight)**gamma)
                elif color == 'brown': # figure 1DE
                    fc = (0.5, 0.05, 0.05, np.abs(weight)**gamma)
                else:
                    fc = ((np.sign(weight)+1)/2, 0, (1-np.sign(weight))/2, np.abs(weight)**gamma)
                # http://matplotlib.sourceforge.net/users/transforms_tutorial.html
                circ = patches.Circle((x,y), self.pe.scale_circle*scale/sf_0*n_, facecolor=fc, edgecolor='none')#, alpha=0.5*weight)
                # (0.5, 0.5), 0.25, transform=ax.transAxes, facecolor='yellow', alpha=0.5)
                a.add_patch(circ)

            line_segments = LineCollection(segments, linewidths=linewidths, colors=colors, linestyles='solid')
            a.add_collection(line_segments)

        if not(color=='auto'):# chevrons maps etc...
            pylab.setp(a, xticks=[])
            pylab.setp(a, yticks=[])

        a.axis([0, self.n_x, self.n_y, 0])
        pylab.draw()
        if mappable:
            return fig, a, line_segments
        else:
            return fig, a


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


