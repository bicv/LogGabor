{
# COMPUTATIONAL PARAMETERS
#'ncpus' : None, # for a SMP machine
#'ncpus' : 16, # Cluster
'ncpus' : 1, # on the cluster we can run many batches - no need for pp
# 'n_jobs' : 1, # stops after one job so that we do not squat the cluster
# use n_jobs=-1 to occupy all CPUs from a SMP see http://scikit-learn.org/0.13/modules/generated/sklearn.grid_search.GridSearchCV.html
'svm_n_jobs' : 1, #
# Image
'N_X' : 256, # size of images
# Log-Gabor
#'base_levels' : 2.,
'base_levels' : 1.618,
'n_theta' : 24, # number of (unoriented) angles between 0. radians (included) and np.pi radians (excluded)
'B_sf' : 1.5, # 1.5 in Geisler
'B_theta' : 3.14159/8.,
# Matching Pursuit
# TODO : use 1 ??
'alpha' : .0, # exponent of the color envelope
'MP_alpha' : .5, # ratio of inhibition in alpha-Matching Pursuit
# 'N' : 512 # number of edges extracted
'N' : 2**11,
'do_whitening'  : True, # = self.pe.do_whitening
'do_mask'  : True, # self.pe.do_mask
#do_real=False # do we consider log-gabors with a complex part?
'figpath' : 'figures/',
'datapath' : 'database/',
'ext' : '.pdf',
'scale' : .2,
'scale_circle' : 0.08, # relativesize of segments and pivot
'scale_chevrons' : 2.5,
'line_width': 1.,
'line_width_chevrons': .75,
'edge_scale_chevrons': 180.,
'figsize_edges' : 6,
# EdgeFactory PARAMETERS
'd_width' : 45., # Geisler 1.23 deg (full image = 45deg)
'd_min' : .25, # Geisler 1.23 deg (full image = 45deg)
'd_max' : 2., # Geisler 1.23 deg (full image = 45deg)
'N_r' : 6, #
'N_Dtheta' : 24, # equal to n_theta : 24 to avoid artifacts
'N_phi' : 12, #
'N_scale' : 5, #
'loglevel_max': 7,
'N_image' : None, #use all images in the folder 200, #None
'noise' : 0.5, #
}
