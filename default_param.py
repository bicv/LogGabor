{
# Image
'N_X' : 256, # size of images
# Log-Gabor
#'base_levels' : 2.,
'base_levels' : 1.618,
'n_theta' : 24, # number of (unoriented) angles between 0. radians (included) and np.pi radians (excluded)
'B_sf' : 1.5, # 1.5 in Geisler
'B_theta' : 3.14159/8.,
'figpath' : 'figures/',
'datapath' : 'database/',
'ext' : '.pdf',
# EdgeFactory PARAMETERS
'd_width' : 45., # Geisler 1.23 deg (full image = 45deg)
'd_min' : .25, # Geisler 1.23 deg (full image = 45deg)
'd_max' : 2., # Geisler 1.23 deg (full image = 45deg)
'N_r' : 6, #
'N_Dtheta' : 24, # equal to n_theta : 24 to avoid artifacts
'N_phi' : 12, #
'N_scale' : 5, #
'loglevel_max': 7,
'alpha' : .0, # exponent of the color envelope
'N_image' : None, #use all images in the folder 200, #None
'noise' : 0.5, #
}
