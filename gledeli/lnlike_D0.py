from typing import Tuple, Optional
from ompy.normalizer_nld import NormalizerNLD
import numpy as np

class LnlikeD0:
    """ Calculate lnlike for D0 or nld(Sn)

    Attributes:
        D0_model: model D0 """
    def __init__(self):
        self.D0_model: Optional[float] = None

    def D0_from_nldSn(self, nldSn: float, **pars) -> float:
        """ Wrapper for ompy.NormalizerNLD.D0_from_nldSn with given nld(Sn)

        Args:
            nldSn: Level density at Sn
            pars: Normalization parameters (spin cut model ...).
                Can be provided through ompy.NormalizationParameters.asdict().

        Returns:
            D0: Average s-wave resonance spacing
        """

        def nld_model_dummy(x):
            """ Callable that return nld(Sn) in MeV^-1 """
            return nldSn

        D0 = NormalizerNLD.D0_from_nldSn(nld_model_dummy, **pars)
        self.D0_model = D0
        return D0

    def lnlike(self, exp: Tuple[float, float],
               model: Optional[float] = None) -> float:
        """ Loglikelihood

        Args:
            exp: Experimental value as [value, 1sigma]
            model: Model D0. Defaults to self.D0_model.

        Returns:
            loglikelohood
        """
        model = self.D0_model if model is None else model

        diff = (model - exp[0])/exp[1]
        return (-0.5 * diff**2).item()
