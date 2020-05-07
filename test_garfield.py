"""
Minimal test interface for using gf from GAMBIT.
"""

from garfield import Interface
from collections import OrderedDict
from pathlib import Path
import os


#
# Global initialization
#
import numpy as np


# Do any initialization stuff that we only
# want to do once per scan, e.g. read experiment
# data from file to memory.
# abs_dir = os.path.dirname(os.path.abspath(__file__))
# data_path = os.path.join(abs_dir, 'some_data_file.dat')
data_path = Path("data")
gf = Interface(data_path)
# gf.init(data_path)
gf.Sn = 8.
gf.nld_pars = {"T": 0.7, "Eshift": -0.4,
               "Ecrit": 2.0}
# gf.gsf_pars = {}
gsf_pars = {}

gsf_pars['p1'], gsf_pars['p2'], gsf_pars['p3']  = np.array([12.68, 236., 3.])  # noqa
gsf_pars['p4'], gsf_pars['p5'], gsf_pars['p6']  = np.array([15.2,  175., 2.2]) # noqa
gsf_pars['p7'], gsf_pars['p8'], gsf_pars['p9'] = np.array([6.33, 4.3, 1.9])
gsf_pars['p10'], gsf_pars['p11'], gsf_pars['p12'] = np.array([10.6, 30., 5.])
gsf_pars['p13'], gsf_pars['p14'], gsf_pars['p15']   = np.array([2.86, 0.69, 0.69]) # noqa
gsf_pars['p20'] = 0.6

gf.gsf_pars = gsf_pars

gf.run()

#
# Interface functions
#

def set_model_pars(pars):
    """
    Communicate model parameters from GAMBIT to gf

    Args:
        pars: Dictionary with GAMBIT parameter names and values
    """

    # Split input parameters into gsf and nld parameters
    nld_pars = OrderedDict()
    gsf_pars = OrderedDict()

    for par_name, par_val in pars.items():

        if par_name[0:3] == "nld":
            nld_pars[par_name] = par_val

        elif par_name[0:3] == "gsf":
            gsf_pars[par_name] = par_val

    gf.set_nld_pars(nld_pars)
    gf.set_gsf_pars(gsf_pars)


def run(settings):
    """
    Do the gf work

    Args:
        settings: A dictionary with any parameter-point-specific
                  settings for fumpy
    Returns:
        success: Boolean indicating whether all went well
    """

    # Extract whatever per-parameter-point settings gf might need
    gf.set_setting_A(settings["setting_A"])
    gf.set_setting_B(settings["setting_B"])

    # Do the work
    success = gf.run()

    return success


def get_results():
    """
    Communicate results from gf back to GAMBIT

    Returns:
        results: A dictionary (str-double) with results for GAMBIT
    """

    # Get log-likelihoods and any other numbers we want to save
    results = {}
    results["loglike"] = gf.get_loglike()
    results["some_other_number"] = 3.1415

    return results

