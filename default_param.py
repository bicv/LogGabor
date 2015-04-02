# -*- coding: utf8 -*-
{
# Image
# 'N_image' : None, #use all images in the folder
'N_image' : 100, #use 100 images in the folder
# 'N_image' : 10, #use 4 images in the folder
'seed' : None, # a seed for the Random Number Generator (RNG) for picking images in databases, set to None xor set to a given number to freeze the RNG
'N_X' : 256, # size of images
'N_Y' : 256, # size of images
# 'N_X' : 64, # size of images
# 'N_Y' : 64, # size of images
'noise' : 0.1, # level of noise when we use some
'do_mask'  : True, # self.pe.do_mask
# whitening parameters:
'do_whitening'  : True, # = self.pe.do_whitening
'white_name_database' : 'serre07_distractors',
'white_n_learning' : 0,
'white_N' : .07,
'white_N_0' : .0, # olshausen = 0.
'white_f_0' : .4, # olshausen = 0.2
'white_alpha' : 1.4,
'white_steepness' : 4.,
'white_recompute' : False,
# Log-Gabor
#'base_levels' : 2.,
'base_levels' : 1.618,
'n_theta' : 24, # number of (unoriented) angles between 0. radians (included) and np.pi radians (excluded)
'B_sf' : .4, # 1.5 in Geisler
'B_theta' : 3.14159/18.,
# EdgeFactory PARAMETERS
'd_width' : 45., # Geisler 1.23 deg (full image = 45deg)
'd_min' : .25, # Geisler 1.23 deg (full image = 45deg)
'd_max' : 2., # Geisler 1.23 deg (full image = 45deg)
'N_r' : 6, #
'N_Dtheta' : 24, # equal to n_theta : 24 to avoid artifacts
'N_phi' : 12, #
'N_scale' : 5, #
'loglevel_max': 7,
# PATHS
'figpath' : 'figures/',
'edgefigpath' : 'figures/edges/',
'matpath' : 'mat/',
'edgematpath' : 'mat/edges/',
'datapath' : 'database/',
'ext' : '.pdf',
}
