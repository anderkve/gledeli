from typing import Union
from numpy import pi
import numpy as np

# commonly used const. strength_factor, convert in mb^(-1) MeV^(-2)
strength_factor = 8.6737E-08


def fSP(value: float) -> float:
    """
    Single particle strength (constant strength function)

    Parameters:
        value (float]):
            Single particle strength

    Returns:
        float: strength function
    """
    return value


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
    def _Gamma_c(E):
        return Gamma_GLO(E, E0=E0, Gamma0=Gamma0, T=T)

    f1 = (E * _Gamma_c(E)) / ((E**2 - E0**2)**2 + E**2 * _Gamma_c(E)**2)
    f2 = 0.7 * _Gamma_c(0) / E0**3

    f = strength_factor * sigma0 * Gamma0 * (f1 + f2)
    return f


def Gamma_GLO(E: Union[np.ndarray, float], E0: float, Gamma0: float,
              T: float) -> Union[np.ndarray, float]:
    """ Width parameter in GLO model

    Note:
        See GLO for documentation

    Parameters:
        E (Union[np.ndarray, float]):
            Input E value
        E0 (float):
            Location parameter
        Gamma0 (float):
            Width parameter
        T (float):
            Temperature parameter
    """
    return Gamma0 / E0**2 * (E**2 + (2. * pi * T)**2)


def Gamma_EGLO(E: Union[np.ndarray, float], E0: float, Gamma0: float,
               T: float,
               epsilon_0: float, k: float) -> Union[np.ndarray, float]:
    """ Width parameter in GLO model

    Note:
        See EGLO and GLO for documentation

    Parameters:
        E (Union[np.ndarray, float]):
            Input E value
        E0 (float):
            Location parameter
        Gamma0 (float):
            Width parameter
        T (float):
            Temperature parameter
        epsilon_0 (float):
            reference/"critical" energy for enhanced width. Free parameter
        k (float):
            enhancement factor; free parameter of e.g. from Fermi gas.
            For k=1 , this is equivalent to Gamma_GLO.
    """
    chi = k + (1.0 - k) * (E - epsilon_0) / (E0 - epsilon_0)
    return chi * Gamma_GLO(E=E, E0=E0, Gamma0=Gamma0, T=T)


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

    def _Gamma_k(E):
        return Gamma_EGLO(E, E0=E0, Gamma0=Gamma0, T=T,
                          epsilon_0=epsilon_0, k=k)

    f1 = (E * _Gamma_k(E)) / ((E**2 - E0**2)**2 + E**2 * _Gamma_k(E)**2)
    f2 = 0.7 * _Gamma_k(E=0) / E0**3

    f = strength_factor * sigma0 * Gamma0 * (f1 + f2)

    return f


def fMGLO_CT(E: Union[np.ndarray, float],
             E0: float, sigma0: float, Gamma0: float,
             T: float, epsilon_0: float = 0, k: float = 1
             ) -> Union[np.ndarray, float]:
    """Modified Generalized Lorentzian with CT Temperature

    Modified to have with constant temperature of final states

    adapted from
    - J. Kroll et al., “Strength of the scissors mode in odd-mass Gd isotopes
      from the radiative capture of resonance neutrons,” Physical Review C,
      vol. 88, no. 3, Art. no. 3, Sep. 2013, doi: 10.1103/physrevc.88.034317.

    Note:
        - This was modified for a constant temperature dependece

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

    def _Gamma_GLO(E):
        return Gamma_GLO(E, E0=E0, Gamma0=Gamma0, T=T)

    def _Gamma_EGLO(E, k=k):
        return Gamma_EGLO(E, E0=E0, Gamma0=Gamma0, T=T,
                          epsilon_0=epsilon_0, k=k)

    f1 = (E * _Gamma_EGLO(E)) / ((E**2 - E0**2)**2 + E**2 * _Gamma_EGLO(E)**2)
    f2 = 0.7 * _Gamma_GLO(E=0) / E0**3

    f = strength_factor * sigma0 * Gamma0 * (f1 + f2)

    return f


def fGH_CT(E: Union[np.ndarray, float],
           E0: float, sigma0: float, Gamma0: float,
           T: float, k: float = 0.63) -> Union[np.ndarray, float]:
    """Goriely's Hybrid model, but with constant temperature of final states

    adapted from:
        - S. Goriely, Radiative neutron captures by neutron-rich nuclei and
        the r-process nucleosynthesis, Physics Letters B, Elsevier BV, 1998,
        436, 10-18 DOI: 10.1016/s0370-2693(98)00907-1
        - RIPL3 eq. (144-145)

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
        k (float, optional):
            enhancement factor

    Returns:
        Union[np.ndarray, float]: strength function
    """
    Gamma = k * Gamma0 * (E**2 + 4 * pi**2 * T**2) / (E*E0)
    f1 = (E * Gamma) / ((E**2 - E0**2)**2 + E**2 * Gamma**2)

    f = strength_factor * sigma0 * Gamma0 * f1
    return f


# def fMLO_Template_CT(E: Union[np.ndarray, float],
#                      E0: float, sigma0: float, Gamma0: float,
#                      T: float, Gamma_MLO: Union[np.ndarray, float]
#                      ) -> Union[np.ndarray, float]:
#     """Modified Lorentzian for a given modified width and constant Temp.

#     Adapted from:
#         - [original reference]
#         - RIPL3 eq. (151-145)

#     Note: Modified to have constant temperature of final states

#     Parameters:
#         E (Union[np.ndarray, float]):
#             Input E value
#         E0 (float):
#             Location parameter
#         sigma0 (float):
#             Scale factor
#         Gamma0 (float):
#             Width parameter
#         T (float):
#             Temperature parameter
#         Gamma_MLO (Union[np.ndarray, float]):
#             Modified width

#     Returns:
#         Union[np.ndarray, float]: strength function
#     """
#     f1 = (E * Gamma_MLO) / ((E**2 - E0**2)**2 + E**2 * Gamma_MLO**2)
#     Lambda = 1 / (1 - np.exp(-E0/T))

#     f = strength_factor * Lambda * sigma0 * Gamma0 * f1
#     return f
