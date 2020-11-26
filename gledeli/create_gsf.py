import numpy as np
from typing import Dict, Optional, Union
from numpy import pi
import logging

from ompy import Vector

logger = logging.getLogger(__name__)


class CreateGSF:
    """Create gamma-ray strength function from model

    Attributes:
        energy (np.ndarray):
            Energy grid on which the gsf shall be calculated, in MeV
        pars (Dict):
            GSF parameters
        GDR_model (str):
            Model for GDR, must be in ["GLO", "EGLO"]
    """

    def __init__(self, pars: Dict, energy: Optional[np.ndarray] = None):
        self.pars = pars
        self.energy = energy

        self.gdr_model: str = "GLO"

        self._gsf = None

    def __call__(self, energy: Optional[np.ndarray] = None, **kwargs):
        """Wrapper for self.create

        Args:
            energy (Optional[np.ndarray], optional): Description
            **kwargs: Description

        Returns:
            TYPE: Description
        """
        return self.create(energy, **kwargs)

    def create(self, energy: Optional[np.ndarray] = None,
               kind: str = "total") -> Vector:
        """Create the model

        Args:
            energy (Optional[np.ndarray], optional):
                Gamma-ray energy
            kind (str, optional):
                Elmagentic type.
                Has to be in ["total", "tot", "sum", "E1", "M1"]. The default
                is "total", and gives total/summed strength function. Following
                kinds are equavalent: ["tot", "total", "sum"].

        Returns:
            Vector: gsf
        """
        if kind in ["total", "tot", "sum"]:
            model = self.model
        elif kind == "E1":
            model = self.model_E1
        elif kind == "M1":
            model = self.model_M1

        self.energy = self.energy if energy is None else energy
        values = model(x=self.energy, pars=self.pars)
        gsf = Vector(values=values, E=self.energy)
        self._gsf = gsf
        return gsf

    def model(self, x, pars):
        """
        The model prediction y(x; pars) for the input point x,
        given the model parameters in the pars dict

        Args:
            x (float):
                The input x value
            pars (dictionary {string: float}):
                The parameter names and values for the given parameter point

        Returns:
            TYPE: Description
        """
        # Sum of E1 and M1 contribution
        y = self.model_E1(x, pars) + self.model_M1(x, pars)
        return y

    @staticmethod
    def model_E1_GLO(x, pars):
        """
        The model prediction for the E1 component of the data with GLO GDR,
        y(x; pars) for the input point x, given the model parameters
        in the pars dict

        Args:
            x (float):
                The input x value
            pars (dictionary {string: float}):
                The parameter names and values for the given parameter point

        Returns:
            TYPE: Description
        """
        y = (fGLO_CT(x, pars['p1'], pars['p2'], pars['p3'], pars['T'])
             + fGLO_CT(x, pars['p4'], pars['p5'], pars['p6'], pars['T'])
             + fSLO(x, pars['p7'], pars['p8'], pars['p9'])
             + fSLO(x, pars['p10'], pars['p11'], pars['p12'])
             )
        return y

    @staticmethod
    def model_E1_EGLO(x, pars):
        """
        The model prediction for the E1 component of the data with EGLO GDR,
        y(x; pars) for the input point x, given the model parameters
        in the pars dict

        Args:
            x (float):
                The input x value
            pars (dictionary {string: float}):
                The parameter names and values for the given parameter point

        Returns:
            TYPE: Description
        """
        y = (fEGLO_CT(x, pars['p1'], pars['p2'], pars['p3'], pars['T'],
                      pars['epsilon_0'], pars['k'])
             + fEGLO_CT(x, pars['p4'], pars['p5'], pars['p6'], pars['T'],
                        pars['epsilon_0'], pars['k'])
             + fSLO(x, pars['p7'], pars['p8'], pars['p9'])
             + fSLO(x, pars['p10'], pars['p11'], pars['p12'])
             )
        return y

    @staticmethod
    def model_M1(x, pars):
        """
        The model prediction for the M1 component of the data,
        y(x; pars) for the input point x, given the model parameters
        in the pars dict

        Args:
            x (float):
                The input x value
            pars (dictionary {string: float}):
                The parameter names and values for the given parameter point

        Returns:
            TYPE: Description
        """

        # Single SLO
        y = fSLO(x, pars['p13'], pars['p14'], pars['p15'])
        return y

    @property
    def model_E1(self):
        return self.__model_E1

    @property
    def gdr_model(self):
        return self.__gdr_model

    @gdr_model.setter
    def gdr_model(self, model):
        logger.debug(f"setting gdr model to {model}")
        self.__gdr_model = model
        if model.lower() == "GLO".lower():
            self.__model_E1 = self.model_E1_GLO
        elif model.lower() == "EGLO".lower():
            self.__model_E1 = self.model_E1_EGLO
        else:
            raise NotImplementedError(f"GDR model {model} unknown")


# commonly used const. strength_factor, convert in mb^(-1) MeV^(-2)
strength_factor = 8.6737E-08


def fSLO(E: Union[np.ndarray, float],
         E0: float, sigma0: float, Gamma0: float
         ) -> Union[np.ndarray, float]:
    """
    Standard Lorentzian function f(E; E0, sigma0, Gamma0)
    adapted from Kopecky & Uhl (1989) eq. (2.1)

    Parameters:
        E (Union[np.ndarray, float]):
            Input E value
        E0 (float):
            Location parameter
        sigma0 (float):
            Scale factor
        Gamma0 (float):
            Width parameter

    Returns:
        Union[np.ndarray, float]: strength function
    """
    f = strength_factor * sigma0 * E * Gamma0**2 / \
        ((E**2 - E0**2)**2 + E**2 * Gamma0**2)
    return f


def fGLO_CT(E: Union[np.ndarray, float],
            E0: float, sigma0: float, Gamma0: float,
            T: float) -> Union[np.ndarray, float]:
    """Generalized Lorentzian with constant temperature of final states
    adapted from Kopecky & Uhl (1989) eq. (2.3-2.4)

    Note: Modified to have constant temperature of final states

    Parameters:
        E (Union[np.ndarray, float]):
            Input E value
        E0 (float):
            Location parameter
        sigma0 (float):
            Scale factor
        Gamma0 (float):
            Width parameter
        T (float):
            Temperature parameter

    Returns:
        Union[np.ndarray, float]: strength function
    """
    def Gamma_c(E, T=T):
        return Gamma0 / E0**2 * (E**2 + (2. * pi * T)**2)
    f1 = (E * Gamma_c(E)) / ((E**2 - E0**2)**2 + E**2 * Gamma_c(E)**2)
    f2 = 0.7 * Gamma_c(0) / E0**3

    f = strength_factor * sigma0 * Gamma0 * (f1 + f2)
    return f


def fEGLO_CT(E: Union[np.ndarray, float],
             E0: float, sigma0: float, Gamma0: float,
             T: float, epsilon_0: float = 0, k: float = 1
             ) -> Union[np.ndarray, float]:
    """Enhanced Generalized Lorentzian with CT Temperature

    Modified to have with constant temperature of final states

    adapted from
    - J. Kopecky and M. Uhl, in Capture Gamma Ray Spectroscopy,
      Proceedings of the Seventh International Symposium on Capture Gamma-ray
      Spectroscopy and Related Topics, edited by R. W. Ho6; AIP Conf. Proc. No.
      238 (AIP, New York, 1991), p. 607. DOI: 10.1063/1.41227

    See also:
        - S.G. Kadmenskii, V.P. Markushev, and V.I. Furman. Radiative width of
          neutron reson- ances. Giant dipole resonances. Sov. J. Nucl. Phys.
        - J. Kopecky, M. Uhl and R.E. Chrien, Phys. Rev. C47, 312 (1993)
          DOI: 10.1103/physrevc.47.312
        - RIPL3

    Note:
        - This was modified for a constant temperature dependece
        - RIPL3 provides emperical parametrization of k and epsilon_0,
          but as stated in the original work, they may be changed

    Parameters:
        E (Union[np.ndarray, float]):
            Input E value
        E0 (float):
            Location parameter
        sigma0 (float):
            Scale factor
        Gamma0 (float):
            Width parameter
        T (float):
            Temperature parameter
        epsilon_0 (float, optional):
            reference/"critical" energy for enhanced width. Free parameter
        k (float, optional):
            enhancement factor; free parameter of e.g. from Fermi gas.
            For the default value, k=1 , this is equivalent to GLO

    Deleted Parameters:
        A: int
            mass number

    Returns:
        Union[np.ndarray, float]: strength function
    """

    # # (MeV); adopted from RIPL, "depends on model for state density"
    # epsilon_0 = 4.5
    # if A < 148:
    #     k = 1.0
    # if(A >= 148):
    #     k = 1. + 0.09 * (A - 148)**2 * np.exp(-0.18 * (A - 148))

    chi = k + (1.0 - k) * (E - epsilon_0) / (E0 - epsilon_0)

    def Gamma_k(E, T=T):
        return chi * Gamma0 / E0**2 * (E**2 + (2. * pi * T)**2)

    f1 = (E * Gamma_k(E)) / ((E**2 - E0**2)**2 + E**2 * Gamma_k(E)**2)
    f2 = 0.7 * Gamma_k(E=0) / E0**3

    f = strength_factor * sigma0 * Gamma0 * (f1 + f2)

    return f
