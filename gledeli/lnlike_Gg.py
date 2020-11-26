from typing import Tuple, Optional, Callable
from ompy import NormalizerGSF, Vector
import numpy as np


class LnlikeGg(NormalizerGSF):
    """ Calculate lnlike for Gg

    Attributes:
        Gg_model: model Gg
    """

    def Gg_standard(self, nld: Callable[[float], Vector],
                    gsf: Callable[[float], Vector],
                    D0: float) -> float:
        """ Compute normalization from <Γγ> (Gg) integral; overwrites parent

        See NormalizerGSF.Gg_standard, however, now just using the given
        nld and gsf; no extra model

        Args:
            nld: function providing the nld at the an input energy E
            gsf: function providing the gsf at the an input energy E
            D0: D0 to use with Gg
        """
        def integrate() -> np.ndarray:
            Eg, stepsize = self.norm_pars.E_grid()
            Ex = self.norm_pars.Sn[0] - Eg
            integral = (np.power(Eg, 3) * gsf(Eg).values * nld_warpper(Ex)
                        * self.SpinSum(Ex, self.norm_pars.Jtarget))
            integral = np.sum(integral) * stepsize
            return integral

        def nld_warpper(Ex) -> np.ndarray:
            """ flips Ex for increasing energy grid and flipps back result """
            return nld(Ex[::-1])[::-1]

        integral = integrate()

        # factor of 2 because of equi-parity `spinsum` instead of `spinsum(π)`,
        # see above
        integral /= 2
        Gg = integral * D0 * 1e3  # [eV] * 1e3 -> [meV]
        self._Gg_model = Gg
        return Gg

    def lnlike(self, exp: Tuple[float, float],
               model: Optional[float] = None) -> float:
        """ Loglikelihood

        Args:
            exp: Experimental value as [value, 1sigma]
            model: Model Gg. Defaults to self.Gg_model.

        Returns:
            loglikelohood

        """
        model = self._Gg_model if model is None else model
        assert exp is not None
        assert model is not None

        diff = (model - exp[0])/exp[1]
        return (-0.5 * diff**2).item()
