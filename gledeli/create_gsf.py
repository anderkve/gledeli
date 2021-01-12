import numpy as np
from typing import Dict, Optional
import logging

from ompy import Vector

from .gsf_models import fSP, fSLO, fGLO_CT, fEGLO_CT, fMGLO_CT, fGH_CT

logger = logging.getLogger(__name__)


class CreateGSF:
    """Create gamma-ray strength function from model

    Attributes:
        energy (np.ndarray):
            Energy grid on which the gsf shall be calculated, in MeV
        pars (Dict):
            GSF parameters
        GDR_model (str):
            Model for GDR
    """

    def __init__(self, pars: Dict, energy: Optional[np.ndarray] = None):
        self.pars = pars
        self.energy = energy

        self.gdr_model: str = "GLO-CT"

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
    def model_E1_GLO_CT(x, pars):
        """
        The model prediction for the E1 component of the data with GLO-CT GDR,
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
    def model_E1_EGLO_CT(x, pars):
        """
        The model prediction for the E1 component of the data with EGLO-CT GDR,
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
    def model_E1_MGLO_CT(x, pars):
        """
        The model prediction for the E1 component of the data with MGLO-CT GDR,
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
        y = (fMGLO_CT(x, pars['p1'], pars['p2'], pars['p3'], pars['T'],
                      pars['epsilon_0'], pars['k'])
             + fMGLO_CT(x, pars['p4'], pars['p5'], pars['p6'], pars['T'],
                        pars['epsilon_0'], pars['k'])
             + fSLO(x, pars['p7'], pars['p8'], pars['p9'])
             + fSLO(x, pars['p10'], pars['p11'], pars['p12'])
             )
        return y

    @staticmethod
    def model_E1_GH_CT(x, pars):
        """
        The model prediction for the E1 component of the data with GH-CT GDR,
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
        y = (fGH_CT(x, pars['p1'], pars['p2'], pars['p3'], pars['T'],
                    pars['KGH'])
             + fGH_CT(x, pars['p4'], pars['p5'], pars['p6'], pars['T'],
                      pars['KGH'])
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

        # TODO: improove interface here; see also #28
        try:
            y += fSP(pars["constantM1"])
        except KeyError:
            pass

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
        if model.lower() == "GLO-CT".lower():
            self.__model_E1 = self.model_E1_GLO_CT
        elif model.lower() == "EGLO-CT".lower():
            self.__model_E1 = self.model_E1_EGLO_CT
        elif model.lower() == "MGLO-CT".lower():
            self.__model_E1 = self.model_E1_MGLO_CT
        elif model.lower() == "GH-CT".lower():
            self.__model_E1 = self.model_E1_GH_CT
        else:
            raise NotImplementedError(f"GDR model {model} unknown")
