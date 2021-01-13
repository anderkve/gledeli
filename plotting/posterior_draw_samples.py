"""
Example script that reads the content of a
GAMBIT hdf5 output file and plots the posterior distribution
of one dataset
"""
import h5py
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
import re
import copy
from typing import Dict, Union, Optional, Tuple, Sequence
from pathlib import Path
from tqdm import tqdm
from scipy.integrate import trapz

gledelig_path = '../'
gledelig_path = Path(__file__).parent / gledelig_path
sys.path.insert(0, str(gledelig_path.resolve()))

# get instance of gledeli with parameters (spincut ...)
from gledeliBE import glede  # load first to mock pymultinest # noqa

from gledeli.create_nld import CreateNLD  # noqa
from gledeli.create_gsf import CreateGSF  # noqa


MIN_PYTHON = (3, 7)
if sys.version_info < MIN_PYTHON:
    sys.exit("Python %s.%s or later is required.\n" % MIN_PYTHON)


class HDFLoader:
    """ Load results from GAMBIT search

    Attributes:
        nld_model: name of nld model (run `load` or `get_nld_model` to set it)
        gsf_model: name of gsf model (run `load` or `get_gsf_model` to set it)
    """

    def __init__(self):
        # list of model names [GAMBIT, gledeli]
        self._nld_model_names = \
            [["NLDModelBSFG_and_discretes", "bsfg_and_discrete"],
             ["NLDModelCT_and_discretes", "ct_and_discrete"]]
        self._gsf_model_names = \
            [["GSFModel20", "GLO-CT"],
             ["GSF_GLO_CT_Model20", "GLO-CT"],
             ["GSF_EGLO_CT_Model20", "EGLO-CT"],
             ["GSF_MGLO_CT_Model20", "MGLO-CT"],
             ["GSF_GH_CT_Model20", "GH-CT"],
             #["GSF_constantM1","SP"]
             ]
        self.nld_model: Optional[str] = None
        self.gsf_model: Optional[str] = None

    def load(self, fname: Union[str, Path]) -> Tuple[pd.DataFrame, Dict]:
        """ loads results from hdf5 file as pd.DataFrame

        Args:
            fname: Filename for GAMBIT's hdf5 output

        Returns:
            tuple[data, names]:
                data is nld, gsf and likelihood and posterior parameterers,
                and names is a dict with the names of the parameters per model
        """
        file = h5py.File(fname, 'r')
        group = file['/scan_output/']

        nld_model = self.get_models(group, "nld")
        gsf_model = self.get_models(group, "gsf")

        data = {}
        data["gsf"] = self.hdf5_gsf_pars(group, gsf_model)
        data["nld"] = self.hdf5_nld_pars(group, nld_model)
        data["other"] = self.hdf5_other_pars(group)
        data["results"] = self.hdf5_results_pars(group)

        # Remove bad points:
        mask = data["results"]['LogLike_isvalid']
        for key, value in data.items():
            for key2, value2 in value.items():
                data[key][key2] = data[key][key2][mask]
        # TODO: Beatify code above?

        # convert to dataframe
        names = {}
        for key, value in data.items():
            data[key] = pd.DataFrame.from_dict(value)
            names[key] = data[key].columns.values

        # verify_integrity ensures that we don't have pars with same name
        data = pd.concat([data["nld"], data["gsf"], data["other"],
                         data["results"]],
                         axis=1, verify_integrity=True)

        return data, names

    def get_models(self, group: h5py._hl.group.Group,
                   which: str ="nld") -> dict:
        if which == "nld":
            known_models = self._nld_model_names
        elif which == "gsf":
            known_models = self._gsf_model_names
        else:
            raise NotImplementedError('Can only find models in ["gsf", "nld"]',
                                      f'but which is set to {which}')

        model = self._get_model(group, known_models)

        if which == "nld":
            self.nld_model = model["gledeli"]
        elif which == "gsf":
            self.gsf_model = model["gledeli"]
        return model

    @staticmethod
    def _get_model(group, known_models) -> dict:
        keys = group.keys()
        for pair in known_models:
            if any(re.match(".*"+pair[0]+".*", key) for key in keys):
                names = pair
                break
        try:
            model = {"gambit": names[0], "gledeli": names[1]}
        except NameError:
            raise NotImplementedError(f"No known model name in {keys}")
        return model

    @staticmethod
    def hdf5_gsf_pars(group: h5py._hl.group.Group, model_name: dict) -> dict:
        gsf_data = {}
        basename = f'#{model_name["gambit"]}_parameters '\
                   f'@{model_name["gambit"]}''::primary_parameters::gsf_{key}'
        for i in range(1, 20):
            gambit_name = basename.format(key=f"p{i}")
            gsf_data[f'gsf_p{i}'] = np.array(group[gambit_name])

        if model_name["gledeli"] in ["MGLO-CT", "EGLO-CT"]:
            p_extra = ["T", "epsilon_0", "k"]
        elif model_name["gledeli"] == ["GH-CT"]:
            p_extra = ["T", "k"]
        elif model_name["gledeli"] == ["GLO-CT"]:
            p_extra = ["T"]

        for par in p_extra:
            gsf_data[f"gsf_{par}"] = np.array(group[basename.format(key=par)])

        # TODO: clean up after gledeli/gambit_np refactoring
        try:
            model_name = ["GSF_constantM1", "constantM1"]
            basename = f'#{model_name[0]}_parameters '\
                       f'@{model_name[0]}''::primary_parameters::gsf_{key}'
            par = model_name[1]
            gsf_data[f"gsf_{par}"] = np.array(group[basename.format(key=par)])
        except KeyError:
            print("Log: Didn't find constantM1 amonst keys")
            pass

        return gsf_data

    @staticmethod
    def hdf5_nld_pars(group: h5py._hl.group.Group, nld_model: dict) -> dict:
        nld_data = {}
        basename = f'#{nld_model["gambit"]}_parameters '\
                   f'@{nld_model["gambit"]}''::primary_parameters::nld_{key}'
        if nld_model["gledeli"] == "bsfg_and_discrete":
            keys = ["NLDa", "Ecrit", "Eshift"]
        elif nld_model["gledeli"] == "ct_and_discrete":
            keys = ["T", "Ecrit", "Eshift"]
        else:
            raise NotImplementedError()
        for key in keys:
            nld_data[f'nld_{key}'] = np.array(group[basename.format(key=key)])
        return nld_data

    @staticmethod
    def hdf5_other_pars(group: h5py._hl.group.Group) -> dict:
        # quickfix to put here
        other_data = {}
        base = "#gledeliResults @NuclearBit::getGledeliResults::"
        other_data['Gg_model'] = np.array(group[base+"Gg_model"])
        other_data['D0_model'] = np.array(group[base+"D0_model"])
        return other_data

    @staticmethod
    def hdf5_results_pars(group: h5py._hl.group.Group) -> dict:
        data = {}
        data['LogLike'] = np.array(group["LogLike"])
        data['LogLike_isvalid'] = np.array(group["LogLike_isvalid"],
                                           dtype=bool)
        try:
            data['posterior_weights'] = np.array(group["Posterior"])
        except KeyError:
            # TODO: what is the "actual" values it should be filled with?
            print("replacing posterior weights [missing key] with ones")
            data['posterior_weights'] = np.ones_like(data["LogLike"])
        return data


class PosteriorPlotter:
    """ Plot posterior and profile likelihoods

    Args:
        glede: Instance of gledeli with models set (as of Oct 2020:nld model)
    """
    def __init__(self, glede):
        self.glede = glede
        self._nld = self.glede._nld
        self._gsf = self.glede._gsf

    def gsf_plot(self, *args, **kwargs):
        """ wrapper for _gsf_plot """
        if len(args) > 0:
            df = args[0]
            args = list(args)
            renamed = rename_gsf_columns(df)
            args[0] = renamed
        elif "df" in kwargs.keys():
            df = kwargs["df"]
            renamed = rename_gsf_columns(df)
            kwargs["df"] = renamed
        return self._gsf_plot(*args, **kwargs)

    def _gsf_plot(self,
                  df: pd.DataFrame,
                  sort_by: Optional[str] = "posterior_weights",
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

            creategsf = CreateGSF(pars=dic)
            ax.plot(x, creategsf.create(x).values, '-',
                    color='k', alpha=alpha)
            ax.plot(x, creategsf.create(x, kind="E1").values, '--',
                    color='b', alpha=alpha)
            ax.plot(x, creategsf.create(x, kind="M1").values, '--',
                    color='g', alpha=alpha)
            if i == n_samples:
                break

        data = self.glede._lnlikegsf_exp.data
        for name, group in data.groupby("label"):
            ax.errorbar(x=group["x"], y=group["y"], yerr=group["yerr"],
                        fmt="o",
                        label=name, mfc="None", ms=5)

        ax.plot([], [], '-', color='k', label='samples, sum')
        ax.plot([], [], '--', color='b', label='samples, E1')
        ax.plot([], [], '--', color='g', label='samples, M1')

        try:
            self.plot_analyzed_om_result(ax,
                gledelig_path/"data/162Dy_oslo/old_analysis/gsf.txt") # noqa
        except OSError:
            print("No old Oslo data available")

        ax.legend()

        ax.set_yscale('log')
        ax.set_xlabel(rf"$\gamma$-ray energy $E_\gamma$~[MeV]")
        ax.set_ylabel(rf"$\gamma$-SF f($E_\gamma$) [MeV$^{{-3}}$]")
        return fig, ax

    @staticmethod
    def plot_analyzed_om_result(ax, file, **kwargs):
        """ Plot analyzed oslo method results on top

        Assumed format: [E, values, err, err_tot]
            where we will use the columns with err_tot, including
            normalization error
         """
        data = np.loadtxt(file, usecols=[0, 1, 3])
        kwargs.setdefault("alpha", 0.2)
        kwargs.setdefault("fmt", "<")
        kwargs.setdefault("label", "OM analysis")
        ax.errorbar(data[:, 0], data[:, 1], data[:, 2], **kwargs)
        ax.legend()

    def nld_plot(self, *args, **kwargs):
        """ wrapper for _nld_plot """
        if len(args) > 0:
            df = args[0]
            args = list(args)
            renamed = rename_nld_columns(df)
            args[0] = renamed
        elif "df" in kwargs.keys():
            df = kwargs["df"]
            renamed = rename_nld_columns(df)
            kwargs["df"] = renamed
        return self._nld_plot(*args, **kwargs)

    def _nld_plot(self,
                  df: pd.DataFrame,
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
        createNLD = copy.deepcopy(self._nld)
        createNLD.energy = x

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

        try:
            self.plot_analyzed_om_result(ax,
                gledelig_path/"data/162Dy_oslo/old_analysis/nld.txt") # noqa
        except OSError:
            print("No old Oslo data available")

        ax.set_yscale('log')
        ax.set_ylabel(r"Level density $\rho(E_x)~[\mathrm{MeV}^{-1}]$")
        ax.set_xlabel(r"Excitation energy $E_x~[\mathrm{MeV}]$")

        ax.set_ylim(bottom=1)
        return fig, ax

    def compare_firstgen(self,
                         pars: pd.Series, ax=None):
        """ Compare input first generation spectrum to a fit

        Args:
            nld_pars: one row of DataFrame containing parmeters
            ax (matplotlib axis, optional): The axis to plot onto. If not
                provided, a new figure is created

        Returns:
            The figure and axis used.
        """
        if ax is None:
            nexp = len(glede._lnlikefgs)
            nexp=2
            fig, ax = plt.subplots(2, nexp, constrained_layout=True)
            fig.set_size_inches(4*nexp, 6)
        else:
            fig, ax = ax.figure, ax

        nld = copy.deepcopy(self._nld)
        pars_renamed1 = rename_nld_columns(pars)

        nld.pars = pars_renamed1.to_dict()
        pars_renamed2 = rename_gsf_columns(pars)
        gsf = CreateGSF(pars=pars_renamed2.to_dict())

        # nld = glede._nld
        # gsf = glede._gsf
        # nld.pars = pars.to_dict()
        # gsf.pas = pars.to_dict()

        for i, (name, fg_creator) in enumerate(glede._lnlikefgs.items()):
            fg_creator.nld = nld
            fg_creator.gsf = gsf
            exp = fg_creator.matrix

            model = fg_creator.create()

            exp.plot(ax=ax[i, 0], scale="log", vmin=1e-3, vmax=1e-1)
            model.plot(ax=ax[i, 1], scale="log", vmin=1e-3, vmax=1e-1)

            x = np.linspace(*ax[i, 0].get_ylim())
            ax[i, 0].plot(x, x, "r--", label="E_x = E_g")
            ax[i, 1].plot(x, x, "r--", label="E_x = E_g")

            ax[i, 0].text(0.05, 0.05, rf"(input: {name} exp.)",
                          fontsize=plt.rcParams["axes.labelsize"],
                          transform=ax[i, 0].transAxes)

            ax[i, 1].text(0.05, 0.05, r"(model)",
                          fontsize=plt.rcParams["axes.labelsize"],
                          transform=ax[i, 1].transAxes)
        return fig, ax

    def plot_posterior_marginals(self,
                                 data: pd.DataFrame, key: str, ax=None,
                                 histrange=None, qs=0.999, **kwargs):
        """ Plots marginalized posteriors

        Args:
            data: samples
            key: key to make histogram of
            ax (matplotlib axis, optional): The axis to plot onto. If not
                provided, a new figure is created
            histrange: range for histogram. If not provided (default),
                calculates it from the fraction `qs` of sample to include
            qs: fraction of sample to include in bound, defaults to 0.999.
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
        if histrange is None:
            q = [0.5 - 0.5*qs, 0.5 + 0.5*qs]
            histrange = self.quantile(x, q, weights=weights)
        hist, bin_edges = np.histogram(x, bins=100, weights=weights,
                                       range=histrange)
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

    @staticmethod
    def quantile(x, q, weights=None):
        """
        Compute sample quantiles with support for weighted samples.
        Note
        ----
        When ``weights`` is ``None``, this method simply calls numpy's percentile
        function with the values of ``q`` multiplied by 100.
        Parameters
        ----------
        x : array_like[nsamples,]
           The samples.
        q : array_like[nquantiles,]
           The list of quantiles to compute. These should all be in the range
           ``[0, 1]``.
        weights : Optional[array_like[nsamples,]]
            An optional weight corresponding to each sample. These
        Returns
        -------
        quantiles : array_like[nquantiles,]
            The sample quantiles computed at ``q``.
        Raises
        ------
        ValueError
            For invalid quantiles; ``q`` not in ``[0, 1]`` or dimension mismatch
            between ``x`` and ``weights``.
        """
        x = np.atleast_1d(x)
        q = np.atleast_1d(q)

        if np.any(q < 0.0) or np.any(q > 1.0):
            raise ValueError("Quantiles must be between 0 and 1")

        if weights is None:
            return np.percentile(x, list(100.0 * q))
        else:
            weights = np.atleast_1d(weights)
            if len(x) != len(weights):
                raise ValueError("Dimension mismatch: len(weights) != len(x)")
            idx = np.argsort(x)
            sw = weights[idx]
            cdf = np.cumsum(sw)[:-1]
            cdf /= cdf[-1]
            cdf = np.append(0, cdf)
            return np.interp(q, cdf, x[idx]).tolist()


def bm1(df: Union[pd.DataFrame, Dict],
        x: np.array = np.arange(0, 20, 0.005)) -> float:
    """BM1 strength from gsf
    see eq19 in PRC 98, 054310 (2018)

    Note: Use trapz as integration as it is quite simple (less time consuming)

    Args:
        df: parameters
        x: values to calculate samples for the integral

    Returns:
        B(M1) in μ_N^2
    """
    def fM1(x):
        return CreateGSF.model_M1(x, df)
    const = 2.5980e8  # 27(ħc)³/16π  in μ_N^2 MeV²
    integral = trapz(fM1(x), x=x)
    return const*integral


def corner_plot(results: pd.DataFrame, n_samples: int = 10000,
                exclude: Sequence[str] = None, **kwargs):
    """ create cornerplot

    Args:
        results: parameters before eq. weighting
        n_samples: number of sampels for corner plot (quickly becomes
            very high memory consuption from matplotlib)
        exclude : keys to exclude from cornerplot. Defaults to
            (see sourcecode)
    """
    from corner import corner
    if exclude is None:
        exclude = ["gsf_p16", "gsf_p17", "gsf_p18", "gsf_p19",
                   'posterior_weights', 'LogLike', 'LogLike_isvalid']
    results_equal = results.sample(n=n_samples, weights="posterior_weights",
                                   random_state=6548)
    for item in exclude:
        results_equal.pop(item)

    kwargs.setdefault("quantiles", [0.16, 0.5, 0.84])
    kwargs.setdefault("show_titles", True)
    kwargs.setdefault("title_kwargs", {"fontsize": 12})
    kwargs.setdefault("title_fmt", '.3f')
    kwargs.setdefault("max_n_ticks", 4)
    fig = corner(results_equal, labels=results_equal.columns, **kwargs)
    return fig


def rename_gsf_columns(df: Union[pd.DataFrame, pd.Series],
                       copy=False, inplace=False,
                       **kwargs) -> pd.DataFrame:
    if isinstance(df, pd.DataFrame):
        current_names = df.columns
        axis = "columns"
    elif isinstance(df, pd.Series):
        current_names = df.index
        axis = "index"
    mapper = {col: col[4:] for col in current_names if col[:4] == 'gsf_'}
    renamed = df.rename(mapper, axis=axis, copy=copy, inplace=inplace,
                        **kwargs)
    return renamed


def rename_nld_columns(df: Union[pd.DataFrame, pd.Series],
                       copy=False, inplace=False,
                       **kwargs) -> pd.DataFrame:
    if isinstance(df, pd.DataFrame):
        current_names = df.columns
        axis = "columns"
    elif isinstance(df, pd.Series):
        current_names = df.index
        axis = "index"
    mapper = {col: col[4:] for col in current_names if col[:4] == 'nld_'}
    renamed = df.rename(mapper, axis=axis, copy=copy, inplace=inplace,
                        **kwargs)
    return renamed


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

    print("loading results")
    hdfloader = HDFLoader()
    results, names = hdfloader.load(results_file)
    glede._nld.model = hdfloader.nld_model
    print("finished")

    results_gsf = rename_gsf_columns(results)
    results["B(M1)"] = results_gsf.apply(bm1, axis=1)
    print("calculated BM1")

    # create equally weighted samples
    results_equal = results.sample(n=100, weights="posterior_weights",
                                   random_state=6548)
    results_equal["posterior_weights"] = 1

    # gsf plots
    pp = PosteriorPlotter(glede)
    fig, ax = pp.gsf_plot(results)
    fig.suptitle("sampels sorted by posterior_weight")
    fig.savefig(figdir/"gsf_sampels sorted by posterior_weight")

    fig, ax = pp.gsf_plot(results, sort_by="LogLike")
    fig.suptitle("sampels sorted by likelihood")
    fig.savefig(figdir/"gsf_sampels sorted by likelihood")

    fig, ax = pp.gsf_plot(results_equal, sort_by=None)
    fig.suptitle("random samples, equally weighted")
    fig.savefig(figdir/"gsf_random samples_eqweight")

    # nld plots
    def wnld_plot(*args, **kwargs):
        """ wrapper """
        return pp.nld_plot(*args, **kwargs, data_path=gledelig_path/"data")

    fig, ax = wnld_plot(results)
    fig.suptitle("sampels sorted by posterior_weight")
    fig.savefig(figdir/"nld_sampels sorted by posterior_weight")

    fig, ax = wnld_plot(results, sort_by="LogLike")
    fig.suptitle("sampels sorted by likelihood")
    fig.savefig(figdir/"nld_sampels sorted by likelihood")

    fig, ax = wnld_plot(results_equal, sort_by=None)
    fig.suptitle("random samples, equally weighted")
    fig.savefig(figdir/"nld_random samples eqweight")

    # firstgen plots
    fig, ax = \
        pp.compare_firstgen(results.iloc[
                                results['posterior_weights'].idxmax()])
    fig.savefig(figdir/"fg_posterior_max")
    fig, ax = \
        pp.compare_firstgen(results.iloc[
                                results['LogLike'].idxmax()])
    fig.savefig(figdir/"fg_loglike_max")

    for i, (_, row) in enumerate(results_equal.iterrows()):
        fig, ax = \
            pp.compare_firstgen(row)
        fig.savefig(figdir/f"fg_random_{i}")
        plt.close(fig)
        if i > 10:
            break

    plt.show()

    for key in tqdm(results.keys()):
        if key in names["results"]:
            continue
        elif key in names["nld"]:
            base = ""
        elif key in names["gsf"]:
            base = ""
        else:
            base = "dependent_"
        fig, _ = pp.plot_posterior_marginals(results, key, alpha=0.5)
        fig.savefig(figdir / (f'{base}{key}_posterior_hist.png'))
        plt.close(fig)

    for key in ["D0_model", "Gg_model", "B(M1)"]:
        base = "dependent_"
        if key == "D0_model":
            histrange = [0, 10]
        elif key == "Gg_model":
            histrange = [0, 500]
        else:
            histrange = [0, 50]
        fig, _ = pp.plot_posterior_marginals(results, key, alpha=0.5,
                                             histrange=histrange)
        fig.savefig(figdir / (f'{base}{key}_posterior_hist_zzom.png'))
        plt.show()
        plt.close(fig)

    print("prepare corner plot")
    fig = corner_plot(results)
    fig.suptitle("random samples, equally weighted")
    fig.savefig(figdir/"corner")
    plt.close(fig)
