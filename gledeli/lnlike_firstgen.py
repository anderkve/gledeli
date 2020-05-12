import numpy as np
from typing import Optional, Callable
from scipy.ndimage import gaussian_filter1d

from ompy.decomposition import chisquare_diagonal
from ompy import Matrix, Vector, nld_T_product


class LnlikeFirstGen():
    """ Calculate lnlike for first generation matrix

    Attributes:
        nld: function providing the nld at the an input energy E
        gsf: function providing the gsf at the an input energy E
        matrix: experimental fg matrix
        matrix_std: experimental fg matrix uncertainty
        resolutionEx: FWHM resolution of particle detector in (MEV)
        resolutionEg: FWHM resolution of gamma-ray detector in (MEV)
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

        self._model = None
        pass

    def model_firsgen(self) -> Matrix:
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
            binsize = matrix.Ex[1] - matrix.Ex[0]
            sigma = self.resolutionEx / 2.3548 / binsize
            model.values = gaussian_filter1d(model.values, sigma, axis=0,
                                             truncate=self._truncate)
        if self.resolutionEg is not None:
            binsize = matrix.Eg[1] - matrix.Eg[0]
            sigma = self.resolutionEg / 2.3548 / binsize
            model.values = gaussian_filter1d(model.values, sigma, axis=1,
                                             truncate=self._truncate)

        self._model = model
        return model

    def lnlike(self, matrix: Optional[Matrix] = None,
               matrix_std: Optional[Matrix] = None,
               model: Optional[Matrix] = None) -> float:
        """Loglikelihood

        Args:
            matrix (optional): Experimental matrix
            matrix_std (optional): Std. devs of experimental matrix
            model (optional): Modelled matrix.

        Returns:
            log-likelihood
        """
        matrix = self.matrix if matrix is None else matrix
        matrix_std = self.matrix_std if matrix_std is None else matrix_std
        model = self._model if model is None else model
        if model is None:
            model = self.model_firsgen()

        # resolution: whether to extend beyond Ex=Eg
        resolution = np.zeros_like(matrix.Ex)
        chi = chisquare_diagonal(matrix.values, model.values,
                                 matrix_std.values, resolution, matrix.Eg,
                                 matrix.Ex)
        return -0.5*chi
