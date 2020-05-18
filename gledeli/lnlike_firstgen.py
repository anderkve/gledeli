import numpy as np
from typing import Optional, Callable, Union
from scipy.ndimage import gaussian_filter1d
import logging

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

    def __init__(self, nld: Callable[[float], Vector],
                 gsf: Callable[[float], Vector],
                 matrix: Matrix, matrix_std: Matrix):

        self.nld = nld
        self.gsf = gsf
        self.matrix = matrix
        self.matrix_std = matrix_std

        self.resolutionEx: Optional[float] = 0.15
        self.resolutionEg: Optional[float] = 0.1
        self._truncate: float = 2


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

        values *= matrix.values.sum(axis=1)[:, np.newaxis]
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
