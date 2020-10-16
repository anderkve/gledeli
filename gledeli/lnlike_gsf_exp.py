from typing import Tuple, Optional, Callable
import numpy as np
import pandas as pd
from ompy import Vector


class LnlikeGSFexp:
    """ Calculate lnlike for experimental GSF data

    Attributes:
        data: Experimental gsf data
        gsf: function providing the gsf at the an input energy E
    """
    def __init__(self, gsf: Optional[Callable[[float], Vector]] = None):
        self.gsf = gsf
        self.data: Optional[pd.DataFrame] = None

    def lnlike(self, data: Optional[pd.DataFrame] = None) -> float:
        """ Loglikelihood

        Args:
            data: Experimental data. Defaults to self.data

        Returns:
            loglikelihood

        """
        chi2 = 0
        data = self.data if data is None else data
        kinds = data.index.unique(level="kind")
        for kind in kinds:
            exp = data.loc[kind]
            model = lambda E: self.gsf(E, kind=kind).values
            assert exp is not None
            assert model is not None
            diff = (model(exp["x"]) - exp["y"])/exp["yerr"]
            chi2 += np.sum(diff**2)
        return np.asscalar(-0.5 * chi2)
