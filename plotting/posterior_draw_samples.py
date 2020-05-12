"""
Example script that reads the content of a
GAMBIT hdf5 output file and plots the posterior distribution
of one dataset
"""
import h5py
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import sys
from typing import Dict, Union, Optional
from pathlib import Path
from tqdm import tqdm

gledelig_path = '../Backends/installed/gledeli/1.0'
gledelig_path = Path(__file__).parent / gledelig_path
sys.path.insert(0, str(gledelig_path.resolve()))
from gledeli.create_nld import CreateNLD  # noqa
from gledeli.create_gsf import CreateGSF  # noqa

MIN_PYTHON = (3, 7)
if sys.version_info < MIN_PYTHON:
    sys.exit("Python %s.%s or later is required.\n" % MIN_PYTHON)


class HDFLoader:
    def __init__(self):
        pass

    def load(self, fname: Union[str, Path]) -> Dict[str, pd.DataFrame]:
        """ loads results from hdf5 file as pd.DataFrame

        Args:
            fname: Filename for GAMBIT's hdf5 output

        Returns:
            data: dict with keys ["gsf, "nld", "lnlike"] containing parameters
                for those. Note that data["lnlike"] also contains the
                posterior weights.
        """
        file = h5py.File(fname, 'r')
        group = file['/scan_output/']

        data = {}
        data["gsf"] = self.hdf5_gsf_pars(group)
        data["nld"] = self.hdf5_nld_pars(group)
        data["lnlike"] = self.hdf5_lnlike_pars(group)

        # Remove bad points:
        mask = data["lnlike"]['LogLike_isvalid']
        for key, value in data.items():
            for key2, value2 in value.items():
                data[key][key2] = data[key][key2][mask]
        # Solved: For some reason loglike False survives?
        # TODO: Beatify

        # convert to dataframe
        for key, value in data.items():
            data[key] = pd.DataFrame.from_dict(value)
        return data

    @staticmethod
    def hdf5_gsf_pars(group: h5py._hl.group.Group) -> dict:
        gsf_data = {}
        basename = '#GSFModel20_parameters @GSFModel20::primary_parameters::' \
                   'gsf_p{i}'
        for i in range(1, 21):
            gsf_data[f'p{i}'] = np.array(group[basename.format(i=i)])
        return gsf_data

    @staticmethod
    def hdf5_nld_pars(group: h5py._hl.group.Group) -> dict:
        nld_data = {}
        basename = '#NLDModelCT_and_discretes_parameters '\
                   '@NLDModelCT_and_discretes::primary_parameters::nld_{key}'
        keys = ["Ecrit", "T", "Eshift"]
        for key in keys:
            nld_data[f'{key}'] = np.array(group[basename.format(key=key)])
        return nld_data

    @staticmethod
    def hdf5_lnlike_pars(group: h5py._hl.group.Group) -> dict:
        data = {}
        data['posterior_weights'] = np.array(group["Posterior"])
        data['LogLike'] = np.array(group["LogLike"])
        data['LogLike_isvalid'] = np.array(group["LogLike_isvalid"],
                                           dtype=bool)
        return data


def gsf_plot(df: pd.DataFrame, sort_by: Optional[str] = "posterior_weights",
             sort_ascending: bool = False,
             weights: Optional[str] = None,
             n_samples: int = 100,
             x: Optional[np.array] = None, ax=None):
    """ Plots gsf samples

    Args:
        df: dataframe
        sort_by: key to sort by, Defaults to "posterior_weights". Use
            `None` to keep the original sorting.
        sort_ascending: Sort ascending vs. descending. Specify list for
            multiple sort orders. If this is a list of bools, must match the
            length of the by.
        weights: Transparency for plotting. Defaults to choosing an weights
            based on the `sort_by`.
        n_samples: how many samples to plot
        x: Energy grid for the plot. Chooses a default if `None`
        ax (matplotlib axis, optional): The axis to plot onto. If not
            provided, a new figure is created

    Returns:
        The figure and axis used.
    """
    fig, ax = plt.subplots() if ax is None else (ax.figure, ax)

    weights = sort_by if weights is None else weights
    if sort_by is not None:
        df = df.sort_values(sort_by, ascending=sort_ascending)
    if weights is not None:
        weights_max = df[weights][:n_samples].max()
        weights_min = df[weights][:n_samples].min()
        weights_range = weights_max - weights_min

    x = np.linspace(0.1, 15, 100) if x is None else x
    for i, row in enumerate(df.iterrows()):
        dic = row[1].to_dict()
        if weights is None:
            alpha = 1/n_samples
        else:
            alpha = (dic[weights]-weights_min)/weights_range / 5  # arb 1/5
            if alpha < 0:
                assert alpha > -0.01
                alpha = 0

        ax.plot(x, CreateGSF.model(x, dic), '-',
                color='k', alpha=alpha)
        ax.plot(x, CreateGSF.model_E1(x, dic), '--',
                color='b', alpha=alpha)
        ax.plot(x, CreateGSF.model_M1(x, dic), '--',
                color='g', alpha=alpha)
        if i == n_samples:
            break

    ax.plot([], [], '-', color='k', label='samples, sum')
    ax.plot([], [], '--', color='b', label='samples, E1')
    ax.plot([], [], '--', color='g', label='samples, M1')
    ax.legend()

    ax.set_yscale('log')
    ax.set_xlabel(rf"$\gamma$-ray energy $E_\gamma$~[MeV]")
    ax.set_ylabel(rf"$\gamma$-SF f($E_\gamma$) [MeV$^{{-3}}$]")
    return fig, ax


def nld_plot(df: pd.DataFrame,
             data_path: Union[str, Path],
             sort_by: Optional[str] = "posterior_weights",
             sort_ascending: bool = False,
             weights: Optional[str] = None,
             n_samples: int = 100,
             x: Optional[np.array] = None,
             ax=None):
    """ Plots nld samples

    Note:
        As of now, the direcrete levels are binned with the same as the
        x binning generally given here. This might be a too fine binning?

    Args:
        df: dataframe
        data_path: Path to folder containing discrete levels
        sort_by: key to sort by, Defaults to "posterior_weights". Use
            `None` to keep the original sorting.
        sort_ascending: Sort ascending vs. descending. Specify list for
            multiple sort orders. If this is a list of bools, must match the
            length of the by.
        weights: Transparency for plotting. Defaults to choosing an weights
            based on the `sort_by`.
        n_samples: how many samples to plot
        x: Energy grid for the plot. Chooses a default if `None`
        ax (matplotlib axis, optional): The axis to plot onto. If not
            provided, a new figure is created

    Returns:
        The figure and axis used.
    """
    fig, ax = plt.subplots() if ax is None else (ax.figure, ax)

    weights = sort_by if weights is None else weights
    if sort_by is not None:
        df = df.sort_values(sort_by, ascending=sort_ascending)
    if weights is not None:
        weights_max = df[weights][:n_samples].max()
        weights_min = df[weights][:n_samples].min()
        weights_range = weights_max - weights_min

    x = np.linspace(0.1, 8, 100) if x is None else x
    createNLD = CreateNLD(pars=None, energy=x,
                          data_path=data_path)
    for i, row in enumerate(df.iterrows()):
        dic = row[1].to_dict()
        if weights is None:
            alpha = 1/n_samples
        else:
            alpha = (dic[weights]-weights_min)/weights_range / 5  # arb 1/5
            if alpha < 0:
                assert alpha > -0.01
                alpha = 0

        createNLD.pars = dic
        nld = createNLD.create()
        nld.plot(ax=ax, color="b", alpha=alpha)

        if i == n_samples:
            break

    createNLD.pars["Ecrit"] = x[-1]
    bin_edges_discrete, ibin_last = createNLD.bin_edges_discrete()
    discrete = createNLD.load_discrete(bin_edges=bin_edges_discrete)
    discrete.plot(ax=ax, c="k", label="discrete")

    ax.plot([], [], '-', color='b', label='samples')
    ax.legend()

    ax.set_yscale('log')
    ax.set_ylabel(r"Level density $\rho(E_x)~[\mathrm{MeV}^{-1}]$")
    ax.set_xlabel(r"Excitation energy $E_x~[\mathrm{MeV}]$")

    ax.set_ylim(bottom=1)
    return fig, ax


def plot_posterior_marginals(data: pd.DataFrame, key: str, ax=None, **kwargs):
    """ Plots marginalized posteriors

    Args:
        data: samples
        key: key to make histogram of
        ax (matplotlib axis, optional): The axis to plot onto. If not
        provided, a new figure is created
        kwargs: Additional kwargs for plotting

    Returns:
        The figure and axis used.
    """
    if ax is None:
        fig, ax = plt.subplots(constrained_layout=True)
    else:
        fig, ax = ax.figure, ax

    x = data[key]
    weights = data['posterior_weights']
    hist, bin_edges = np.histogram(x, bins=100, weights=weights)
    bin_width = np.diff(bin_edges)
    hist /= hist.max()
    ax.bar(bin_edges[:-1], hist, width=bin_width, align="edge",
           label="posterior", color="tab:blue", **kwargs)

    # profile likelihood (per bin)
    groups = pd.cut(x, bin_edges, precision=6)
    grouped = data.groupby(groups)
    profile = np.zeros(len(grouped))
    for i, (name, group) in enumerate(grouped):
        profile[i] = group["LogLike"].max()

    norm = data["LogLike"].max()
    ax.bar(bin_edges[:-1], np.exp(profile-norm), width=bin_width, align="edge",
           label="profile likelihood", color="tab:orange", **kwargs)

    # add marker for max posterior and max likelihood
    max_post_sample = data.iloc[data['posterior_weights'].idxmax()]
    ax.plot(max_post_sample[key], 0.08, "ks",
            label="max posterior", markerfacecolor="None")
    max_lnlike_sample = data.iloc[data['LogLike'].idxmax()]
    ax.plot(max_lnlike_sample[key], 0.05, "kd",
            label="max Lnlike", markerfacecolor="None")

    # lines to read of 1, 2, 3 sigma uncertainty (Wilk's theorem)
    ax.axhline(np.exp(-0.5), color="k", linestyle="--", linewidth=1)
    ax.axhline(np.exp(-2), color="k", linestyle="--", linewidth=1)
    ax.axhline(np.exp(-4), color="k", linestyle="--", linewidth=1)

    ax.legend(fontsize="xx-small", markerscale=0.5)
    ax.set_xlabel(key)
    ax.set_ylabel("posterior ratio $p/p_{max}$\n"
                  "profile Likelihood ratio $L/L_{max}$")
    return fig, ax


if __name__ == "__main__":
    data_path = (gledelig_path / "data").resolve()
    assert data_path.exists(), f"data path {data_path} does not exists"

    # Open the hdf5 file and access the group 'scan_output'
    try:
        h5py_filepath = sys.argv[1]
    except IndexError:
        h5py_filepath = "../runs/OMBIT_demo/samples/results.hdf5"
    results_file = Path(__file__).parent / h5py_filepath
    results_file.resolve()
    assert results_file.exists(), f'results_file {results_file} ' \
                                  'does not exists'

    figdir = Path("figs/")
    figdir.mkdir(exist_ok=True)

    hdfloader = HDFLoader()
    data = hdfloader.load(results_file)

    res_gsf = pd.concat([data["gsf"], data["lnlike"]], axis=1)
    res_nld = pd.concat([data["nld"], data["lnlike"]], axis=1)

    fig, ax = gsf_plot(res_gsf)
    fig.suptitle("sampels sorted by posterior_weight")
    fig.savefig(figdir/"gsf_sampels sorted by posterior_weight")

    fig, ax = gsf_plot(res_gsf, sort_by="LogLike")
    fig.suptitle("sampels sorted by likelihood")
    fig.savefig(figdir/"gsf_sampels sorted by likelihood")

    res_gsf_equal = res_gsf.sample(n=100, weights="posterior_weights",
                                   random_state=6548)
    fig, ax = gsf_plot(res_gsf_equal, sort_by=None)
    fig.suptitle("random samples, equally weighted")
    fig.savefig(figdir/"gsf_random samples_eqweight")

    def wnld_plot(*args, **kwargs):
        """ wrapper """
        return nld_plot(*args, **kwargs, data_path=gledelig_path/"data")

    fig, ax = wnld_plot(res_nld)
    fig.suptitle("sampels sorted by posterior_weight")
    fig.savefig(figdir/"nld_sampels sorted by posterior_weight")

    fig, ax = wnld_plot(res_nld, sort_by="LogLike")
    fig.suptitle("sampels sorted by likelihood")
    fig.savefig(figdir/"nld_sampels sorted by likelihood")

    res_nld_equal = res_nld.sample(n=100, weights="posterior_weights",
                                   random_state=6548)
    fig, ax = wnld_plot(res_nld_equal, sort_by=None)
    fig.suptitle("random samples, equally weighted")
    fig.savefig(figdir/"nld_random samples eqweight")

    # plt.show()

    for key in tqdm(res_gsf.keys()):
        if key in data["lnlike"].keys():
            continue
        fig, _ = plot_posterior_marginals(res_gsf, key, alpha=0.5)
        fig.savefig(figdir / ('gsf_' + key + '_posterior_hist.png'))

        # fig.close()

    for key in tqdm(res_nld.keys()):
        if key in data["lnlike"].keys():
            continue
        fig, _ = plot_posterior_marginals(res_nld, key, alpha=0.5)
        fig.savefig(figdir / ('nld_' + key + '_posterior_hist.png'))
        # fig.close()
