import numpy as np
import ompy as om
from scipy.interpolate import interp1d

# response for CACTUS
#             Eg    FWHM_rel
NaI2012 = [[ 360 , 1.8465110],
           [ 847 , 1.2905956],
           [1238 , 1.0261709],
           [1779 , 0.9133949],
           [2839 , 0.7804907],
           [3089 , 0.7651453],
           [4497 , 0.6569752],
           [6130 , 0.5542545],
           [9900 , 0.5073242],
           [15000, 0.4333146]]

# Experimental relative FWHM at 1.33 MeV of resulting array
fwhm_abs = 6.8/100 * 1.330  # (90/1330 = 6.8%)

resp = {"Eg": np.array(NaI2012)[:, 0]/1e3,  # convert to MeV
        "fwhm_rel": np.array(NaI2012)[:, 1]}


def interpolate(x, y, fill_value="extrapolate"):
    return interp1d(x, y,
                    kind="linear", bounds_error=False,
                    fill_value=fill_value)


fwhm_rel_1330 = (fwhm_abs / 1.330 * 100)
f_fwhm_rel_perCent = interpolate(resp['Eg'], resp['fwhm_rel'] * fwhm_rel_1330)


def f_fwhm_abs(E):
    return E * f_fwhm_rel_perCent(E)/100
