import numpy as np
from typing import Optional, Callable, Union
from scipy.ndimage import gaussian_filter1d
import logging
from pathlib import Path

from ompy.decomposition import chisquare_diagonal
from ompy.gauss_smoothing import gauss_smoothing_matrix_1D
from ompy import Matrix, Vector, nld_T_product

logger = logging.getLogger(__name__)


class LnlikeFirstGen():
    """ Calculate lnlike for first generation matrix

    Note:
        The model observations can be smoothed using the `resolutionEx` and
        `resolutionEg` keywords, smoothing the corresponding axis.
        - The most realistic smoothing may be obtained with a energy dependent
        FWHM (non-static kernel), which is given as a `np.ndarray`, where the
        entry of a given bin is FWHM of the corresponding energy bin.
        - To speed up, we can use provide a sinlge fwhm (static kernel), such
        that we can use `scipy.gaussian_filter1d`, which is about 50 times
        faster.
        - If you provide `None`, no smoothing is applied.

    Attributes:
        nld: function providing the nld at the an input energy E
        gsf: function providing the gsf at the an input energy E
        matrix: experimental fg matrix
        matrix_std: experimental fg matrix uncertainty
        resolutionEx: FWHM resolution of particle detector in (MeV).
        resolutionEg: FWHM resolution of gamma-ray detector in (MeV).
    """

    def __init__(self, nld: Optional[Callable[[float], Vector]] = None,
                 gsf: Optional[Callable[[float], Vector]] = None):

        self.nld = nld
        self.gsf = gsf

        self.matrix: Optional[Matrix] = None
        self.matrix_std: Optional[Matrix] = None

        self._matrix_rowsum: Optional[np.ndarray] = None

        self.resolutionEx: Optional[float] = 0.15
        self.resolutionEg: Optional[float] = 0.1
        self._truncate: float = 2

    def load_exp(self, fnmat: Optional[Union[str, Path]] = 'firstgen.m',
                 fnstd: Optional[Union[str, Path]] = 'firstgen_std.m'):
        """ Load experimental data from disk

        Note:
            The data should already be trimmed to the aread we want to
            compare to and is assumed to be normalized already.

        Args :
            fnmat: Path to first generation matrix
            fnstd: Path to standard deviations matrix

        Raises:
            AssertionError: If fg matrix is not normalized to 1, and
                Ex and Eg need equal binwidths (and linear spaced).

        """
        self.matrix = self.load_matrix(fname=fnmat)
        self.matrix_std = self.load_matrix(fname=fnstd)

        err_msg = "Fg matrix has to (be trimed) and normalized to 1."
        self._matrix_rowsum = np.nansum(self.matrix.values, axis=1)
        np.testing.assert_allclose(self._matrix_rowsum, 1, atol=0.1,
                                   err_msg=err_msg)

        # check that it's equally spaced
        dEx = np.diff(self.matrix.Ex)
        dEg = np.diff(self.matrix.Eg)
        if not (np.allclose(dEx, dEx[0]) and np.allclose(dEg, dEg[0])):
            raise NotImplementedError()

        assert self.matrix_std.has_equal_binning(self.matrix)

    @staticmethod
    def load_matrix(fname: Optional[Union[str, Path]] = 'firstgen.m'):
        """ Load experimental matrix

        Args:
            fname: Filename, defaults to 'firstgen.m'
        """
        mat = Matrix(path=fname)
        if mat.Eg[-1] > 1000:
            logger.debug("Recalibrate matrix.Eg to MeV")
            mat.Eg /= 1e3
        if mat.Ex[-1] > 1000:
            logger.debug("Recalibrate matrix.Ex to MeV")
            mat.Ex /= 1e3
        return mat

    def create(self) -> Matrix:
        """ Create first gen from model nld and gsf

        In the conversion from gsf to transmission coefficient, we assumes
        that the gsf is given as dipole gsf only

        If resolutionEx and or resolutionEg are given, it will smooth the
        resulting matrix. Assumes linear energy grid when smoothing.

        TODO: Where should I normalize, scale down experiment0 (+std?),
              or scale up model?
        """
        matrix = self.matrix

        assert matrix is not None, "Need to load data upfront"
        assert self.nld is not None, "Need to load data upfront"
        assert self.gsf is not None, "Need to load data upfront"
        Enld = matrix.Ex[-1] - matrix.Eg[::-1]
        nld = self.nld(Enld)
        gsf = self.gsf(matrix.Eg)

        T = gsf.values * gsf.E**3 * 2 * np.pi
        values = nld_T_product(nld=nld.values, T=T,
                               resolution=np.zeros_like(matrix.Ex),
                               E_nld=nld.E, Eg=gsf.E, Ex=matrix.Ex)

        values *= self._matrix_rowsum[:, np.newaxis]
        model = Matrix(values=values, Ex=matrix.Ex, Eg=matrix.Eg)

        if self.resolutionEx is not None:
            self.smoothing(model, "Ex")
        if self.resolutionEg is not None:
            self.smoothing(model, "Eg")

        return model

    def smoothing(self, matrix: Matrix, axis: Union[int, str]):
        """ Smoothing of matrix along axis. Operates inplace

        Note:
            see class docstring

        Args:
            matrix: input matrix
            axis: Axis to smooth.
        """
        try:
            axis = axis.lower()
        except AttributeError:
            pass

        if axis in (0, 'ex'):
            axis = "ex"
            axis_int = 0
        elif axis in (1, 'eg'):
            axis = "eg"
            axis_int = 1

        fwhm = self.resolutionEx if axis == "ex" else self.resolutionEg
        E = matrix.Ex if axis == "ex" else matrix.Eg

        def smooth_constant_sigma():
            binsize = E[1] - E[0]
            sigma = fwhm / 2.3548 / binsize
            matrix.values = gaussian_filter1d(matrix.values, sigma,
                                              axis=axis_int,
                                              truncate=self._truncate)

        def smooth_nonconst_sigma():
            matrix.values = \
                gauss_smoothing_matrix_1D(matrix.values, E, fwhm,
                                          axis=axis, truncate=self._truncate)

        if fwhm is None:
            logger.debug(f"{axis} axis not smoothed")
            pass
        elif isinstance(fwhm, float) or isinstance(fwhm, int):
            logger.debug(f"{axis} axis smoothed with {fwhm:.4f}")
            smooth_constant_sigma()
        elif np.all(fwhm[0] == fwhm[:]):
            fwhm = fwhm[0]
            logger.debug(f"{axis} axis smoothed with {fwhm:.4f}")
            smooth_constant_sigma()
        else:
            logger.debug(f"{axis} axis smoothed with {fwhm}")
            smooth_nonconst_sigma()
        return matrix

    def lnlike(self, matrix: Optional[Matrix] = None,
               matrix_std: Optional[Matrix] = None,
               model: Optional[Matrix] = None) -> float:
        """Loglikelihood

        Args:
            matrix (optional): Experimental matrix
            matrix_std (optional): Std. devs of experimental matrix
            model (optional): Modelled matrix. If None, calculates it
                via `self.create`.

        Returns:
            log-likelihood
        """
        matrix = self.matrix if matrix is None else matrix
        matrix_std = self.matrix_std if matrix_std is None else matrix_std
        if model is None:
            model = self.create()

        # resolution: whether to extend beyond Ex=Eg
        resolution = np.zeros_like(matrix.Ex)
        chi = chisquare_diagonal(matrix.values, model.values,
                                 matrix_std.values, resolution, matrix.Eg,
                                 matrix.Ex)
        return -0.5*chi

    def set_pars(self, nld: Callable[[float], Vector],
                 gsf: Callable[[float], Vector]):
        """ Convenience function to set nld and gsf

        Args:
            nld: see instance attributes
            gsf: see instance attributes
        """
        self.nld = nld
        self.gsf = gsf
