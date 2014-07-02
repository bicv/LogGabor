from LogGabor import LogGabor

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
