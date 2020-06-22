import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Union, Optional
from ompy import NormalizationParameters
import logging

from .create_nld import CreateNLD
from .create_gsf import CreateGSF
from .lnlike_D0 import LnlikeD0
from .lnlike_Gg import LnlikeGg
from .lnlike_firstgen import LnlikeFirstGen
from .lnlike_gsf_exp import LnlikeGSFexp

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
        self._lnlike: dict = None

        self._nld = CreateNLD(data_path=self.data_path, pars=None)
        self._gsf = CreateGSF(pars=None)
        self._lnlikegsf_exp = LnlikeGSFexp()

        # Needs to be initiated (and data loaded) before self.run()
        self._lnlikefgs: Dict[str, LnlikeFirstGen] = None

    def run(self):
        """ Run calculations """
        # set parameters
        self._nld.pars = self.nld_pars
        self._gsf.pars = self.gsf_pars

        err_msg = "lnlike: {:.2e} below cutoff for {}"
        self._lnlike = {}

        nldSn = self._nld.model_nld(self.norm_pars.Sn[0])
        lnlikeD0 = LnlikeD0()
        D0_model = lnlikeD0.D0_from_nldSn(nldSn, **self.norm_pars.asdict())
        lnlike = lnlikeD0.lnlike(self.norm_pars.D0)
        assert self.lnlike_above_cutoff(lnlike), err_msg.format(lnlike, "D0")
        self._lnlike["D0"] = lnlike

        lnlikeGg = LnlikeGg()
        lnlikeGg.norm_pars = self.norm_pars
        Gg_model = lnlikeGg.Gg_standard(self._nld, self._gsf, D0_model)
        lnlike = lnlikeGg.lnlike(self.norm_pars.Gg)
        assert self.lnlike_above_cutoff(lnlike), err_msg.format(lnlike, "Gg")
        self._lnlike["Gg"] = lnlike

        assert self._lnlikefgs is not None, "Need to load fg matrix(es)"
        for name, lnlikefg in self._lnlikefgs.items():
            lnlikefg.set_pars(nld=self._nld, gsf=self._gsf)
            lnlike = lnlikefg.lnlike()
            assert self.lnlike_above_cutoff(lnlike), \
                err_msg.format(lnlike, f"{name} fg matrix")
            self._lnlike[f"{name} matrix"] = lnlike

        self._lnlikegsf_exp.gsf = self._gsf
        lnlike = self._lnlikegsf_exp.lnlike()
        assert self.lnlike_above_cutoff(lnlike), \
            err_msg.format(lnlike, "experimental gsf data")
        self._lnlike["gsf_exp"] = lnlike

        logger.debug(f"D0_model: {D0_model}")
        logger.debug(f"Gg_model: {Gg_model}")
        self.D0_model = D0_model
        self.Gg_model = Gg_model

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


