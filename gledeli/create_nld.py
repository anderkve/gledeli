import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Union, Optional
from ompy.normalizer_nld import NormalizerNLD
from ompy.spinfunctions import SpinFunctions
from ompy import Vector
import logging

logger = logging.getLogger(__name__)


class CreateNLD:
    """ Model nld by a combination of discrete states and nld model

    Note:
        - Think about whether there should be a smoother continuation between
        the discretes and the model. See also github issue #4.

    Attributes:
        data_path: Path to load experimental data
        discrete_level_energies: energies of the discrete levels in MeV
        energy: Energy grid on which the nld shall be calculated, in MeV
        model: Model for nld. Has to be one of
            ["ct_and_discrete", "bsfg_and_discrete"]
        pars: Level density parameters
        spin_pars: parameters for spin distribution, necessary for bsfg model
    """

    discrete_level_energies: Optional[np.ndarray] = None

    def __init__(self, data_path: Union[str, Path], pars: Dict,
                 energy: Optional[np.ndarray] = None,
                 model: str = "ct_and_discrete"):
        self.data_path = Path(data_path)
        self.pars = pars
        self.energy = energy

        self.model = model
        self.spin_pars = None

        self._nld = None

        if type(self).discrete_level_energies is None:
            self.load_discrete_file(self.data_path)

    def __call__(self, energy: Optional[np.ndarray] = None):
        """ Wrapper for self.create """
        return self.create(energy)

    def create(self, energy: Optional[np.ndarray] = None
               ) -> Union[float, Vector]:
        """ Create the nld combination

        Args:
            energy(optional): Energy grid on which the nld shall be
                calculated, in MeV

        Returns:
            model_nld: float if energy is float, else Vector
        """
        if self.model.lower() in ["ct_and_discrete", "bsfg_and_discrete"]:
            nld = self.model_and_discrete(energy)
        else:
            NotImplementedError(f"model {self.model} not known")
        return nld

    def model_and_discrete(self,
                           energy: Optional[np.ndarray] = None
                           ) -> Union[float, Vector]:
        """ Create the nld from the selected model and discrete states

        Calcualtes discretes up to Ecric. Then calculates the model
        nld above Ecrit. Appends both.

        Args:
            energy(optional): Energy grid on which the nld shall be
                calculated, in MeV

        Returns:
            model_nld: float if energy is float, else Vector
        """
        self.energy = self.energy if energy is None else energy
        energy = self.energy

        if isinstance(energy, np.ndarray):
            # calclate bin edges appart from the last one
            bin_edges_discrete, ibin_last = self.bin_edges_discrete()
            discrete = self.load_discrete(bin_edges=bin_edges_discrete)

            bin_mids_model = self.energy[ibin_last:]
            model = self.model_nld(bin_mids_model)

            # find combined value in Ecrit bin
            floor_to_Ecrit = self.pars['Ecrit'] - bin_edges_discrete[ibin_last]
            levels_dists = discrete[-1] * floor_to_Ecrit
            width_Ecrit = self.energy[ibin_last+1] - self.energy[ibin_last]
            levels_model = model[0] * width_Ecrit * (1-floor_to_Ecrit)
            if bin_edges_discrete[-1] == self.pars['Ecrit']:
                density_Ecrit = levels_model / width_Ecrit
            else:
                density_Ecrit = (levels_dists + levels_model) / width_Ecrit

            # combine replacing Ecrit by the found value above
            combined = np.zeros_like(self.energy)
            combined[:ibin_last+1] = discrete.values  # +1 as we append Ecrit
            combined[ibin_last:] = model.values
            combined[ibin_last] = density_Ecrit

            nld = Vector(values=combined, E=self.energy)

        elif isinstance(energy, float):
            if energy < self.pars['Ecrit']:
                NotImplementedError(f"Energy {energy} is below Ecrit"
                                    f"{self.pars['Ecrit']}. Binning for nld "
                                    f"from discretes not determined.")
            nld = self.model_nld(energy)
        else:
            TypeError(f"energy must be float or np.ndarray")

        self._nld = nld
        return nld

    def model_nld(self,
                  energy: Union[float, np.ndarray]) -> Union[float, Vector]:
        """ Create model nld

        Note:
            We oversample the energy grid and average/rebin afterwards
            to get a better estimate of the mean value in each bin

        Args:
            energy: energy grid on which the nld model is calculated

        Returns:
            model_nld: float if energy is float, else Vector
        """
        if self.model.lower() == "ct_and_discrete":
            def model_call(E):
                return NormalizerNLD.const_temperature(E=E,
                            T=self.pars['T'], Eshift=self.pars['Eshift'])# noqa
        elif self.model.lower() == "bsfg_and_discrete":
            def model_call(E):
                return self.model_bsfg(energy=E, NLDa=self.pars['NLDa'],
                                       Eshift=self.pars['Eshift'])
        else:
            NotImplementedError(f"model {self.model} not known")

        if isinstance(energy, np.ndarray):
            oversample_factor: int = 4
            energy_oversampled = np.linspace(energy[0], energy[-1],
                                             num=len(energy)*oversample_factor)
            model = model_call(energy_oversampled)
            # take mean, returning to the initial energy grid
            model = np.mean(model.reshape(-1, oversample_factor), axis=1)
            model_nld = Vector(values=model, E=energy)
        elif isinstance(energy, float):
            model_nld = model_call(energy)
        else:
            TypeError(f"energy must be float or np.ndarray")
        return model_nld

    def model_bsfg(self, energy: Union[float, np.ndarray],
                   NLDa: float, Eshift: float) -> Union[float, Vector]:
        """ Create model nld from BSFG model, with low energy fix

        Low energy fix from EB05, see Eq. (3).
        Alternative from is presented eg. in RIPL3, see eq. (48)

        Args:
            energy: energy grid on which the nld model is calculated
            NLDa: Level density parameter
            Eshift: Backshift parameter

        Returns:
            model_nld: float if energy is float, else Vector

        ToDo: Reloading of spin cut to save computation time
        """
        spin_pars = self.spin_pars
        assert spin_pars is not None, "spin_pars need to be set for bsfg"
        assert all(key in spin_pars.spincutPars for key in ["NLDa", "Eshift"])

        spin_pars.spincutPars.update({"NLDa": NLDa, "Eshift": Eshift})

        sigma2 = SpinFunctions(Ex=energy, J=None,
                               model=spin_pars.spincutModel,
                               pars=spin_pars.spincutPars).get_sigma2()
        sigma = np.sqrt(sigma2)

        return self._model_bsfg(energy, NLDa, Eshift, sigma)

    @staticmethod
    def _model_bsfg(energy: Union[float, np.ndarray],
                    a: float, Eshift: float,
                    sigma: Union[float, np.ndarray]) -> Union[float, Vector]:
        """ Create model nld from BSFG model, with low energy fix

        Low energy fix from EB05, see Eq. (3) and text below it.
        Alternative from is presented eg. in RIPL3, see eq. (48).

        Note that we use a different convention for U and the excitation energy
        than in EB05.

        Args:
            energy: energy grid on which the nld model is calculated
            a: Level density parameter
            Eshift: Backshift parameter
            sigma: Spincut parameter (has to be same type as energy)

        Returns:
            model_nld: float if energy is float, else Vector
        """
        energy = np.atleast_1d(energy)
        U = energy - Eshift
        # fix for low energies
        U = np.where(U < (25/16)/a, (25/16)/a, U)

        nom = np.exp(2*np.sqrt(a*U))
        denom = 12 * np.sqrt(2) * sigma * a**(1/4) * U**(5/4)
        nld = nom / denom

        assert np.isfinite(nld).all(), \
            f"Some values in nld are not finite: {nld}"
        return nld

    def bin_edges_discrete(self):
        """ Get bin edges for the discrete state hist

        Returns:
             bin_edges_discrete, ibin_last: bin edges for the discrete
                states, and the last bin below Ecrit
        """
        bin_edges = self.energy[:-1] - np.diff(self.energy)/2
        ibin_last = np.nonzero(bin_edges < self.pars['Ecrit'])[0][-1]
        assert 0 <= ibin_last <= len(self.energy)

        bin_edges_discrete = bin_edges[:ibin_last+1]
        bin_edges_discrete = np.append(bin_edges_discrete, self.pars['Ecrit'])
        return bin_edges_discrete, ibin_last

    def load_discrete(self, bin_edges: np.ndarray,
                      data_path: Optional[Union[str, Path]] = None) \
                      -> Tuple[np.ndarray, np.ndarray]:  # noqa
        """ Load discrete levels and bin

        Args:
            energy (ndarray): The binning to use, as **binedges**, in MeV
            data_path (Union[str, Path], optional): The file to load

        Returns:
            Tuple[ndarray, ndarray]
        """
        if data_path is not None:
            self.data_path = data_path
            self.load_discrete_file()
        hist, _ = np.histogram(self.discrete_level_energies, bins=bin_edges)
        binsize = np.diff(bin_edges)
        hist = hist.astype(float) / binsize  # convert to levels/MeV
        mid_bins = bin_edges[:-1] + binsize/2
        result = Vector(values=hist, E=mid_bins)
        return result

    def load_discrete_file(self,
                           data_path: Optional[Union[str, Path]] = None):
        """ Loads energies of discrete levels

        Loads them to a class attribute `discrete_level_energies` such that
        they are available also for new instances.

        Args:
            data_path (Union[str, Path], optional): The file to load
        """
        data_path = self.data_path if data_path is None else Path(data_path)
        logger.info(f"Loading discrete levels from data from {self.data_path}")
        energies = np.loadtxt(data_path / 'discrete_levels.txt')
        energies /= 1e3  # convert to MeV
        if len(energies) > 1:
            assert energies.mean() < 5, "Probably energies are not in keV"
        type(self).discrete_level_energies = energies
