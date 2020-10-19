import numpy as np
from typing import Dict, Optional
from numpy import pi
from ompy import Vector


class CreateGSF:
    """ Create gamma-ray strength function from model

    Attributes:
        energy: Energy grid on which the gsf shall be calculated, in MeV
        pars: GSF parameters
    """

    def __init__(self, pars: Dict, energy: Optional[np.ndarray] = None):
        self.pars = pars
        self.energy = energy

        self._gsf = None

    def __call__(self, energy: Optional[np.ndarray] = None, **kwargs):
        """ Wrapper for self.create """
        return self.create(energy, **kwargs)

    def create(self, energy: Optional[np.ndarray] = None,
               kind: str = "total") -> Vector:
        """Create the model

        Args:
            energy (optional): bla
            kind: Elmag. type. Has to be in ["total", "tot", "sum", "E1",
                "M1"]. The default is "total", and gives total/summed strength
                function. Following kinds are equavalent: ["tot", "total",
                "sum"].

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

    @staticmethod
    def model(x, pars):
        """
        The model prediction y(x; pars) for the input point x,
        given the model parameters in the pars dict

        Parameters
        ----------
        x : float
            The input x value
        pars : dictionary {string:float}
            The parameter names and values for the given parameter point

        Returns
        -------
        TYPE
            Description
        """
        # Sum of E1 and M1 contribution
        y = CreateGSF.model_E1(x, pars) + CreateGSF.model_M1(x, pars)
        return y

    @staticmethod
    def model_E1(x, pars):
        """
        The model prediction for the E1 component of the data,
        y(x; pars) for the input point x, given the model parameters
        in the pars dict

        Parameters
        ----------
        x : float
            The input x value
        pars : dictionary {string:float}
            The parameter names and values for the given parameter point
        """
        y = (fGLO(x, pars['p1'], pars['p2'], pars['p3'], pars['p20'])
             + fGLO(x, pars['p4'], pars['p5'], pars['p6'], pars['p20'])
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

        Parameters
        ----------
        x : float
            The input x value
        pars : dictionary {string:float}
            The parameter names and values for the given parameter point
        """

        # Single SLO
        y = fSLO(x, pars['p13'], pars['p14'], pars['p15'])
        return y


# commonly used const. strength_factor, convert in mb^(-1) MeV^(-2)
strength_factor = 8.6737E-08


def fSLO(E, E0, sigma0, Gamma0):
    """
    Standard Lorentzian function f(E; E0, sigma0, Gamma0)
    adapted from Kopecky & Uhl (1989) eq. (2.1)

    Parameters:
    E : float
        Input E value
    E0 : float
        Location parameter
    Gamma0 : float
        Width parameter
    sigma0 : float
        Scale factor
    """
    f = strength_factor * sigma0 * E * Gamma0**2 / \
        ((E**2 - E0**2)**2 + E**2 * Gamma0**2)
    return f


def fGLO(E, E0, sigma0, Gamma0, T):
    """
    Generalized Lorentzian f(E; E0, sigma0, Gamma0, T)
    adapted from Kopecky & Uhl (1989) eq. (2.3-2.4)

    Parameters:
    E : float
        Input E value
    E0 : float
        Location parameter
    Gamma0 : float
        Width parameter
    sigma0 : float
        Scale factor
    T : float
        Temperature parameter
    """
    Gamma = Gamma0 * (E**2 + 4 * pi**2 * T**2) / E0**2
    f1 = (E * Gamma) / ((E**2 - E0**2)**2 + E**2 * Gamma**2)
    f2 = 0.7 * Gamma0 * 4 * pi**2 * T**2 / E0**5

    f = strength_factor * sigma0 * Gamma0 * (f1 + f2)
    return f


def fEGLO(E, E0, sigma0, Gamma0, T, A):
    """
    Enhanced Generalized Lorentzian f(E; E0, sigma0, Gamma0, T)
    -- modified CT: see below
    adapted from
    [92] S.G. Kadmenskii, V.P. Markushev, and V.I. Furman. Radiative width of neutron reson-
    ances. Giant dipole resonances. Sov. J. Nucl. Phys., 37:165, 1983.
    [91] J. Kopecky and R.E. Chrien. Observation of the M 1 giant resonance by resonance
    averaging in 106 P d. Nuclear Physics A, 468(2):285-300, 1987. ISSN 0375-9474. doi:
    10.1016/0375-9474(87)90518-5.
    and RIPL3 documentation
    Note: This was modi but constant temperature dependece!

    Parameters:
    E : float
        Input E value
    E0 : float
        Location parameter
    Gamma0 : float
        Width parameter
    sigma0 : float
        Scale factor
    T : float
        Temperature parameter
    A : int
        mass number
    """

    # (MeV); adopted from RIPL: However, potentially depends on model for state density
    epsilon_0 = 4.5

    # also k is adopted from RIPL: However,"depends on model for state
    # density! (they assume Fermi gas!)
    if A < 148:
        k = 1.0
    if(A >= 148):
        k = 1. + 0.09 * (A - 148) * (A - 148) * np.exp(-0.18 * (A - 148))

    Kappa = k + (1.0 - k) * (E - epsilon_0) / (E0 - epsilon_0)
    Kappa_0 = k + (k - 1.) * (epsilon_0) / (E0 - epsilon_0)
    Gamma_k = Kappa * Gamma0 * (E**2 + (2.0 * pi * T)**2) / E0**2
    Gamma_k0 = Kappa_0 * Gamma0 * (2. * pi * T)**2 / E0**2
    denominator = (E**2 - E0**2)**2 + E**2 * E0**2

    f = strength_factor * sigma0 * Gamma0 * \
        ((E * Gamma_k) / denominator + 0.7 * Gamma_k0 / E0**3)
    return f
