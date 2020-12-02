import numpy as np
import copy
from typing import Dict
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
from ompy import normalize_rows

from pathlib import Path
file_path = Path(__file__).resolve()
import sys  # noqa
sys.path.append(str(file_path.parents[2]))

from gledeliBE import glede  # noqa
from gledeliBE import set_model_names, get_results, log_stream # noqa

from gledeli.lnlike_D0 import LnlikeD0
from gledeli.lnlike_Gg import LnlikeGg


def define_nld_pars(model) -> Dict:
    # set up different model parameters
    model_nameCT = "NLDModelCT_and_discretes"
    model_nameBSFG = "NLDModelBSFG_and_discretes"

    if model == model_nameCT:
        return {"T": 0.61, "Eshift": -0.9, "Ecrit": 2.3}
    elif model == model_nameBSFG:
        return {"NLDa": 17, "Eshift": 0.30, "Ecrit": 1.98}
    else:
        raise NotImplementedError(f"{model} unknown")


def _base_gsf_model() -> Dict:
    gsf_pars = {}
    gsf_pars['p1'], gsf_pars['p2'], gsf_pars['p3'] = np.array([12.68, 236., 3.])  # noqa
    gsf_pars['p4'], gsf_pars['p5'], gsf_pars['p6'] = np.array([15.2,  175., 2.2]) # noqa
    gsf_pars['p7'], gsf_pars['p8'], gsf_pars['p9'] = np.array([6.42, 4.2, 1.9])
    gsf_pars['p10'], gsf_pars['p11'], gsf_pars['p12'] = np.array([10.6, 30., 4.9])
    gsf_pars['p13'], gsf_pars['p14'], gsf_pars['p15'] = np.array([2.81, 0.54, 0.76]) # noqa
    gsf_pars['T'] = 0.61
    return gsf_pars


def define_gsf_pars(model) -> Dict:
    model_nameGLO = "GSF_GLOModel20"
    model_nameEGLO = "GSF_EGLOModel20"

    if model == model_nameGLO:
        gsf_pars = _base_gsf_model()
        return gsf_pars
    elif model == model_nameEGLO:
        gsf_pars = _base_gsf_model()
        gsf_pars['epsilon_0'] = 5.
        gsf_pars['k'] = 2.
        return gsf_pars
    else:
        raise NotImplementedError(f"{model} unknown")


def init_glede(model_nameNLD, model_nameGSF, nld_pars, gsf_pars, cutoff):
    glede.lnlike_cutoff = cutoff
    model_names = [model_nameNLD, model_nameGSF]
    set_model_names(model_names)
    glede.nld_pars = nld_pars
    glede.gsf_pars = gsf_pars


def perturbed_matrix(model, exp_std):
    mu = model.values
    sigma = exp_std.values

    perturbed = model.copy()
    perturbed.values = np.random.normal(mu, sigma)

    # redraws to avoid negative numbers
    while np.any(perturbed.values < 0):
        negative_entries = perturbed.values < 0
        perturbed.values[negative_entries] \
            = np.random.normal(mu[negative_entries],
                               sigma[negative_entries])
    perturbed.values = normalize_rows(perturbed.values)
    return perturbed


def create_fg(glede):
    for name, lnlike in glede._lnlikefgs.items():
        # lnlike.matrix.plot(vmin=1e-3, vmax=1e-1, scale="log")
        model = lnlike.create()

        fig, axes = plt.subplots(2, 2)
        for ax in axes.flatten():
            perturbed = perturbed_matrix(model, lnlike.matrix_std)
            perturbed.plot(ax=ax, vmin=1e-3, vmax=1e-1, scale="log")

        # save the last one
        perturbed.save(outdir / f"1Gen_{name}.m")
        lnlike.matrix_std.save(outdir / f"1Gen_{name}_std.m")


def create_gsf(glede):
    fig, ax = plt.subplots()
    data = glede._lnlikegsf_exp.data
    perturbed = data.copy()

    for name, group in data.groupby("kind"):
        y_model = glede._gsf.create(group["x"], kind=name).values
        y_model_perturbed = np.random.normal(y_model, group["yerr"])
        perturbed["y"] = y_model_perturbed
        ax.plot(group["x"], y_model, marker=">", ls="None",
                label="model")
        np.savetxt(outdir / f"gsf_{name}.dat", perturbed.values)
    data.plot(ax=ax, x="x", y="y", marker="o", ls="None", label="data")
    perturbed.plot(ax=ax, x="x", y="y", marker="<", ls="None",
                   label="model; perturbed")
    ax.legend()


def create_D0_Gg(glede):
    nldSn = glede._nld.create(glede.norm_pars.Sn[0])
    lnlikeD0 = LnlikeD0()
    D0_model = lnlikeD0.D0_from_nldSn(nldSn, **glede.norm_pars.asdict())
    print(f"D0_model: {D0_model}")
    D0_model_perturbed = np.random.normal(D0_model, glede.norm_pars.D0[1])
    print(f"D0_model perturbed: {D0_model_perturbed}")

    lnlikeGg = LnlikeGg()
    lnlikeGg.norm_pars = glede.norm_pars
    Gg_model = lnlikeGg.Gg_standard(glede._nld, glede._gsf, D0_model)
    print(f"Gg_model: {Gg_model}")
    Gg_model_perturbed = np.random.normal(Gg_model, glede.norm_pars.Gg[1])
    print(f"Gg_model perturbed: {Gg_model_perturbed}")


if __name__ == "__main__":
    np.random.seed(123555543)

    outdir = Path("export")
    outdir.mkdir(exist_ok=True)

    model_nameNLD = "NLDModelCT_and_discretes"
    model_nameGSF = "GSF_GLOModel20"
    nld_pars = define_nld_pars(model_nameNLD)
    gsf_pars = define_gsf_pars(model_nameGSF)
    cutoff = -5e7
    init_glede(model_nameNLD, model_nameGSF, nld_pars, gsf_pars, cutoff)
    glede.run()

    create_fg(glede)
    create_gsf(glede)
    create_D0_Gg(glede)

    plt.show()
