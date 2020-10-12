"""
Minimal test interface for using GLEDELi from GAMBIT.
"""
from collections import OrderedDict
import numpy as np
import logging
from io import StringIO
import re

from settings import ParametrizedInterface


logger = logging.getLogger(__name__)
log_stream = StringIO()
fmtstring = '{name} {levelname}: {message}'
logging.basicConfig(stream=log_stream,
                    format=fmtstring, style="{", level=logging.INFO)

#
# Global initialization
#

# Do any initialization stuff that we only
# want to do once per scan, e.g. read experiment
# data from file to memory.

# initialize as this is not done from withing GAMBIT
glede = ParametrizedInterface().set_settings()


def set_model_names(model_names):
    """
    Dummy function that demonstrates how to get the names
    of active models from GAMBIT
    """
    for name in model_names:
        print("gledeliBE: Got model name:", name)


def set_model_pars(pars):
    """
    Communicate model parameters from GAMBIT to GLEDELi

    Args:
        pars: Dictionary with GAMBIT parameter names and values
    """

    # Split input parameters into gsf and nld parameters
    nld_pars = OrderedDict()
    gsf_pars = OrderedDict()

    for par_name, par_val in pars.items():

        # Sort and rename parameters obtained from GAMBIT
        if par_name[0:4] == "nld_":
            nld_pars[par_name.replace("nld_", "")] = par_val

        elif par_name[0:4] == "gsf_":
            gsf_pars[par_name.replace("gsf_", "")] = par_val

    glede.nld_pars = nld_pars
    glede.gsf_pars = gsf_pars


def run(settings):
    """
    Do the GLEDELi work

    Args:
        settings: A dictionary with any parameter-point-specific
                  settings for fumpy
    Returns:
        success: Boolean indicating whether all went well
    """
    try:
        glede.run()
        success = True
    except Exception as inst:
        is_cutoff = re.match("lnlike: [+-]*[\d.]+(?:e[+-]?\d+)* below cutoff",
                             inst.args[0])
        if is_cutoff is None:
            raise inst
        else:
            logger.info(f"{inst.__class__.__name__}: {inst}")
            success = False

    # Construct a dict with info on how things went
    run_info = {}
    run_info['log_msg'] = log_stream.getvalue()
    log_stream.seek(0)
    log_stream.truncate(0)
    run_info['success'] = success

    return run_info


def get_results():
    """
    Communicate results from GLEDELi back to GAMBIT

    Returns:
        results: A dictionary (str-double) with results for GAMBIT
    """

    # Get log-likelihoods and any other numbers we want to save
    results = {}
    results["loglike"] = glede.get_loglike()
    results["D0_model"] = glede.D0_model
    results["Gg_model"] = glede.Gg_model

    return results


if __name__ == "__main__":
    # run with these parameters
    # glede.nld_pars = {"T": 0.61, "Eshift": -1.02,
    #                   "Ecrit": 2.3}
    glede.nld_pars = {"NLDa": 17, "Eshift": 0.30,
                      "Ecrit": 1.98}
    gsf_pars = {}
    gsf_pars['p1'], gsf_pars['p2'], gsf_pars['p3'] = np.array([12.68, 236., 3.])  # noqa
    gsf_pars['p4'], gsf_pars['p5'], gsf_pars['p6'] = np.array([15.2,  175., 2.2]) # noqa
    gsf_pars['p7'], gsf_pars['p8'], gsf_pars['p9'] = np.array([6.42, 4.2, 1.9])
    gsf_pars['p10'], gsf_pars['p11'], gsf_pars['p12'] = np.array([10.6, 30., 4.9])
    gsf_pars['p13'], gsf_pars['p14'], gsf_pars['p15'] = np.array([2.81, 0.54, 0.76]) # noqa
    gsf_pars['p20'] = 0.61

    glede.gsf_pars = gsf_pars

    try:
        glede.run()
        success = True
    except Exception as inst:
        is_cutoff = re.match("lnlike: [+-]*[\d.]+(?:e[+-]?\d+)* below cutoff",
                             inst.args[0])
        print("asdasd", is_cutoff)
        if is_cutoff is None:
            raise inst
        else:
            logger.info(f"{inst.__class__.__name__}: {inst}")
            success = False

    print(log_stream.getvalue())
    log_stream.seek(0)
    log_stream.truncate(0)

    print("\n Give me a break")
    try:
        glede.run()
        success = True
    except Exception as inst:
        is_cutoff = re.match("lnlike: [+-]*[\d.]+(?:e[+-]?\d+)* below cutoff",
                             inst.args[0])
        print("asdasd", is_cutoff)
        if is_cutoff is None:
            raise inst
        else:
            logger.info(f"{inst.__class__.__name__}: {inst}")
            success = False


    import matplotlib.pyplot as plt
    for name, lnlike in glede._lnlikefgs.items():
        lnlike.matrix.plot(vmin=1e-3, vmax=1e-1, scale="log")
        lnlike.create().plot(vmin=1e-3, vmax=1e-1, scale="log")
        print(name)
        print(lnlike.matrix.Ex[0], lnlike.matrix.Ex[-1], lnlike.matrix.Eg[0],
              lnlike.matrix.Eg[-1])

    # data = glede._lnlikegsf_exp.data
    # data["rel_err"] = data["yerr"]/data["y"]
    # print(data)

    print("results:", get_results())


    print(log_stream.getvalue())

    plt.show()
