### diskchef fitting procedures

This is a template for diskchef fitting pipeline. NOEMA/PRODIGE data of DN Tau are supplied. 
The final pieces of code will be added to the main diskchef repository

delta-v fixed to 0.41? km/s
r_in fixed to ?
alpha_T fixed to  0.55
midplane_T constrain should be very narrow [9, 15]
mass constrain should be narrow

READ UV file
UV_MAP
CLEAN
WRITE CELAN savefilename
FITS "imaged.fits" FROM imaged.lmv-clean /OVERWRITE
