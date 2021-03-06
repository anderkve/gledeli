
"""
Any additional settings for running GLEDELi from GAMBIT.
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Union, Optional
import numpy as np
import pandas as pd
import logging

# Mock importing pymultinest  -- it's not needed but may cause problems
import sys
import mock
MOCK_MODULES = ['pymultinest']
for mod_name in MOCK_MODULES:
    sys.modules[mod_name] = mock.Mock()
from ompy import NormalizationParameters

from data.resolutionEg import f_fwhm_abs
from gledeli import Interface
from gledeli import LnlikeFirstGen

logger = logging.getLogger(__name__)


class ParametrizedInterface:
    """ helper class to create a GELDELI interface with additional settings

    Attributes:
        glede: Inface of GLEDELi
    """

    def __init__(self):
        """
        TODO:
            - call from parrent module?
        """
        current_file_path = Path(__file__).parent
        self._data_path = Path(current_file_path / "data")
        self.glede = Interface(self._data_path)

        self._norm_pars: NormalizationParameters = None

    def set_settings(self):
        # set spincut parameters & exp. D0 / Gg values
        norm_pars = NormalizationParameters(name="162Dy")
        norm_pars.Sn = [8.197, 0.001]  # MeV -- uncertainty not processed (?)
        self.glede.norm_pars = norm_pars
        self.set_exp_data_D0_Gg()
        self.set_spincut_parameters()

        # get experimental gsf data
        self.set_gsf_experiments()

        # set response from oslo-method experiment
        self.set_firstgen_experiments()

        return self.glede

    def set_exp_data_D0_Gg(self):
        """ Set observables D0 and Gg (and target spin"""
        norm_pars = self.glede.norm_pars
        norm_pars.D0 = [2.4, 0.2]  # eV
        norm_pars.Gg = [112., 10.]  # meV
        norm_pars.Jtarget = 5 / 2  # A-1 nucleus

    def set_spincut_parameters(self):
        """ Set spincut paramters for oslo-method type analysis """
        norm_pars = self.glede.norm_pars
        assert norm_pars is not None, "Should be called after assigning Sn"
        norm_pars.spincutModel = 'Disc_and_EB05'
        # workaround in mass to get rigid moment of innertia like in Renst18
        norm_pars.spincutPars = {"mass": 162*(0.9)**(3/5), "NLDa": 18.50,
                                 "Eshift": 0.39,
                                 "Sn": norm_pars.Sn[0],
                                 "sigma2_disc": [1.5, 3.7**2]}
        norm_pars.steps = 100  # number of integration steps for Gg

        self.glede._nld.spin_pars = norm_pars

    def set_firstgen_experiments(self):
        """ Set experimental first generation matrices """
        base = self._data_path / "162Dy_oslo/export"

        self.glede._lnlikefgs = {}

        lnlikefg = LnlikeFirstGen()
        lnlikefg.load_exp(fnmat=base / "1Gen_3He.m",
                          fnstd=base / "1Gen_3He_std.m")
        lnlikefg.resolutionEg = f_fwhm_abs(lnlikefg.matrix.Eg)
        self.glede._lnlikefgs["3He"] = lnlikefg

        lnlikefg = LnlikeFirstGen()
        lnlikefg.load_exp(fnmat=base / "1Gen_4He.m",
                          fnstd=base / "1Gen_4He_std.m")
        lnlikefg.resolutionEg = f_fwhm_abs(lnlikefg.matrix.Eg)
        self.glede._lnlikefgs["4He"] = lnlikefg

    def set_gsf_experiments(self):
        """ Set experimental gsf strength files """
        base = self._data_path
        renstr_label = "Renstrøm2018 et al. > 8.5 MeV"
        gsf_data = \
            [GSFFile(base / "gsf" / "fe1_exp_066_162_photoneut_2018Ren.dat",
                     kind="E1", label=renstr_label),
             # GSFFile(base / "gsf" / "fe1_exp_067_165_photoabs_1981Gur.dat",
             #         kind="E1"),
             # GSFFile(base / "gsf" / "fe1_exp_067_165_photoneut_1966Axe.dat",
             #         kind="E1"),
             # GSFFile(base / "gsf" / "fe1_exp_067_165_photoneut_1969Be8.dat",
             #         kind="E1"),
             # GSFFile(base / "gsf" / "fe1_exp_067_165_photoneut_1976Gor.dat",
             #         kind="E1"),
             GSFFile(base / "gsf" / "fe1_exp_067_165_photoneut_2019Sub.dat",
                     kind="E1"),
             ]
        dfs = [gsf.load() for gsf in gsf_data]
        df = pd.concat(dfs)
        df.set_index(["kind", "label"], inplace=True)
        if any(df["xerr"] != 0):
            raise NotImplementedError()

        df_renstr = df.loc[("E1", renstr_label)]
        df_renstr[df_renstr.x < 8.5] = np.nan
        df.dropna(inplace=True)

        self.glede._lnlikegsf_exp.data = df


@dataclass
class GSFFile:
    """ Class for keeping track of an item in inventory. """
    fname: Union[str, Path]
    kind: str
    label: Optional[str] = None

    def __post_init__(self):
        self.fname = Path(self.fname)
        if self.label is None:
            self.label = self.fname.name

    def load(self):
        data = np.loadtxt(self.fname)
        df = pd.DataFrame(data, columns=["x", "xerr", "y", "yerr"])
        df["kind"] = self.kind
        df["label"] = self.label
        logger.info(f"Loading {self.kind}-gsf dataset {self.label}")
        return df
