
"""
Any additional settings for running GLEDELi from GAMBIT.
"""
from pathlib import Path

# Mock importing pymultinest  -- it's not needed but may cause problems
import sys
import mock
MOCK_MODULES = ['pymultinest']
for mod_name in MOCK_MODULES:
    sys.modules[mod_name] = mock.Mock()
from ompy import NormalizationParameters

from data.resolutionEg import f_fwhm_abs
from gledeli import Interface

class ParametrizedInterface:
    """ helper class to create a GELDELI interface with additional settings """

    def __init__(self):
        """
        TODO:
            - call from parrent module?
        """
        current_file_path = Path(__file__).parent
        data_path = Path(current_file_path / "data")
        self.glede = Interface(data_path)

        self._norm_pars: NormalizationParameters = None

    def set_settings(self):
        # set spincut parameters & exp. D0 / Gg values
        norm_pars = NormalizationParameters(name="164Dy")
        norm_pars.Sn = [7.658, 0.001]  # MeV
        self.glede.norm_pars = norm_pars
        self.set_exp_data_D0_Gg()
        self.set_spincut_parameters()

        # set response from experiment
        assert self.glede._matrix is not None, "Data has to be loaded upfront"
        self.glede._lnlikefg.resolutionEg = f_fwhm_abs(self.glede._matrix.Eg)

        # cutoff for calculations (TODO: grab from gambit)
        self.glede.lnlike_cutoff = -5e5
        return self.glede

    def set_spincut_parameters(self):
        norm_pars = self.glede.norm_pars
        assert norm_pars is not None, "Should be called after assigning Sn"
        norm_pars.spincutModel = 'Disc_and_EB05'
        norm_pars.spincutPars = {"mass": 164, "NLDa": 18.12, "Eshift": 0.31,
                                 "Sn": norm_pars.Sn[0],
                                 "sigma2_disc": [1.5, 3.6]}
        norm_pars.steps = 100  # number of integration steps for Gg

    def set_exp_data_D0_Gg(self):
        norm_pars = self.glede.norm_pars
        norm_pars.D0 = [6.8, 0.6]  # eV
        norm_pars.Gg = [113., 13.]  # meV
        norm_pars.Jtarget = 5/2  # A-1 nucleus
