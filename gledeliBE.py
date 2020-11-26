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
# `set_model_names` will run in GAMBIT initialization


def set_model_names(model_names):
    """
    Set model according to active models from GAMBIT
    """
    logger.debug(f"Attempt to set models in gledeli from gambit.")
    if "NLDModelCT_and_discretes" in model_names:
        model_in_gledeli = "ct_and_discrete"
        glede._nld.model = model_in_gledeli
        logger.debug(f"Set gledeli nld model to {model_in_gledeli}")
    elif "NLDModelBSFG_and_discretes" in model_names:
        model_in_gledeli = "bsfg_and_discrete"
        glede._nld.model = model_in_gledeli
        logger.debug(f"Set gledeli nld model to {model_in_gledeli}")
    else:
        raise NotImplementedError("Selected NLD model unknown")

    if any(k in model_names for k in ["GSFModel20", "GSF_GLOModel20"]):
        model_in_gledeli = "GLO"
        glede._gsf.gdr_model = model_in_gledeli
        logger.debug(f"Set gledeli nld model to {model_in_gledeli}")
    elif "GSF_EGLOModel20" in model_names:
        model_in_gledeli = "EGLO"
        glede._gsf.gdr_model = model_in_gledeli
        logger.debug(f"Set gledeli nld model to {model_in_gledeli}")
    else:
        raise NotImplementedError("Selected GSF model unknown")


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
