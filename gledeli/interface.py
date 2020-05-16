import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Union, Optional
from ompy import NormalizationParameters
import logging

from .dataloader import DataLoader
from .create_nld import CreateNLD
from .create_gsf import CreateGSF
from .lnlike_D0 import LnlikeD0
from .lnlike_Gg import LnlikeGg
from .lnlike_firstgen import LnlikeFirstGen

logger = logging.getLogger(__name__)


class Interface:
    """
    Main interface exposing methods to load data, set parameters and calculate
    the likelihood

    Note:
        Several routines will assume an equally spaced energy grid for the
        experimental Ex-Eg matrix

    Attributes:
        data_path: Path to load experimental data
        nld_pars: Level density parameters
        gsf_pars: Gamma-ray strength function parameters
        A_val: Dummy setting
        B_val: Dummy setting
        Gg: Experimental Gg to compare to, formated as [value, 1 sigma]
        D0: Experimental D0 to compare to, formated as [value, 1 sigma]
        Sn: Neutron binding energy in MeV, needed e.g. to calculate Gg
        norm_pars: Normalization parameters like spin-cut model
        lnlike_cutoff: If lnlike is below this value below this value,
            the calculation will be aborted


    """

    def __init__(self, data_path: Union[str, Path]):
        self.data_path = Path(data_path)

        self._nld_pars: Optional[Dict] = None
        self._gsf_pars: Optional[Dict] = None
        self.A_val = None
        self.B_val = None

        self.Gg: Optional[Tuple[float, float]] = None
        self.D0: Optional[Tuple[float, float]] = None

        self.norm_pars: Optional[NormalizationParameters] = None

        self.lnlike_cutoff: float = None

        # save/calculate observation and models here
        self._matrix = None
        self._matrix_std = None
        self._lnlike: dict = None

        self.load_data(self.data_path)
        self._nld = CreateNLD(data_path=self.data_path, pars=None)
        self._gsf = CreateGSF(pars=None)
        self._lnlikefg = LnlikeFirstGen(nld=None, gsf=None,
                                        matrix=self._matrix,
                                        matrix_std=self._matrix_std)

    def load_data(self, data_path):
        """ Load experimental data from datapath """
        dataloader = DataLoader(data_path)
        self._matrix = dataloader.matrix
        self._matrix_std = dataloader.matrix_std

        # check that it's equally spaced
        dEx = np.diff(self._matrix.Ex)
        dEg = np.diff(self._matrix.Eg)
        if not (np.allclose(dEx, dEx[0]) and np.allclose(dEg, dEg[0])):
            raise NotImplementedError()

        assert self._matrix_std.has_equal_binning(self._matrix)

        if abs(self._matrix.values.sum() - len(self._matrix.Ex)) > 0.1:
            raise NotImplementedError("Input matrix does not seem normalized "
                                      "per Ex row. The current implementation "
                                      "relies on this.")

    def run(self):
        """ Run calculations """
        # set parameters
        self._nld.pars = self.nld_pars
        self._gsf.pars = self.gsf_pars

        # provide nld and gsf model to first generation
        self._lnlikefg.nld = self._nld
        self._lnlikefg.gsf = self._gsf

        assert_string = "lnlike below cutoff for {}"
        self._lnlike = {}

        nldSn = self._nld.model_nld(self.norm_pars.Sn[0])
        lnlikeD0 = LnlikeD0()
        D0_model = lnlikeD0.D0_from_nldSn(nldSn, **self.norm_pars.asdict())
        lnlike = lnlikeD0.lnlike(self.norm_pars.D0)
        assert self.lnlike_above_cutoff(lnlike), assert_string.format("D0")
        self._lnlike["D0"] = lnlike

        lnlikeGg = LnlikeGg()
        lnlikeGg.norm_pars = self.norm_pars
        Gg_model = lnlikeGg.Gg_standard(self._nld, self._gsf, D0_model)
        lnlike = lnlikeGg.lnlike(self.norm_pars.Gg)
        assert self.lnlike_above_cutoff(lnlike), assert_string.format("Gg")
        self._lnlike["Gg"] = lnlike

        lnlike = self._lnlikefg.lnlike()
        assert self.lnlike_above_cutoff(lnlike), assert_string.format("matrix")
        self._lnlike["matrix"] = lnlike

        logger.debug(f"D0_model: {D0_model}")
        logger.debug(f"Gg_model: {Gg_model}")

    def lnlike_above_cutoff(self, lnlike: float) -> bool:
        """ Check if lnlike is above `self.lnlike_cutoff`
        Args:
            lnlike: likelihood to check

        Return:
            bool: True if it is above self.lnlike_cutoff, or self.lnlike_cutoff
                is None, otherwise False
        """
        if self.lnlike_cutoff is None:
            return True
        elif lnlike > self.lnlike_cutoff:
            return True
        else:
            return False

    def get_loglike(self):
        """ Return the loglike for the prediction vs experiment comparison """
        logger.debug([f"loglike of {key} is {val}"
                      for key, val in self._lnlike.items()])
        loglike = sum(self._lnlike.values())
        logger.info(f"This point will get a loglike of {loglike}")
        return loglike

    @property
    def nld_pars(self) -> Optional[Dict]:
        return self._nld_pars

    @nld_pars.setter
    def nld_pars(self, pars) -> None:
        self._nld_pars = pars
        logger.debug(f"Setting nld_pars to: {pars}")

    @property
    def gsf_pars(self) -> Optional[Dict]:
        return self._gsf_pars

    @gsf_pars.setter
    def gsf_pars(self, pars) -> None:
        self._gsf_pars = pars
        logger.debug(f"Setting gsf_pars to: {pars}")


