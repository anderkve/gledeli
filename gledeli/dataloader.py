from pathlib import Path
from typing import Union, Optional
from ompy import Matrix
import logging

logger = logging.getLogger(__name__)


class DataLoader:
    """
    Load experimental data (matrix & uncertainties)

    Attributes:
        data_path: Path to load experimental data
        matrix: Matrix to compare to
        matrix_std: Standard deviation of the matrix
    """

    def __init__(self, data_path: Union[str, Path]):
        self.data_path = Path(data_path)

        logger.info(f"I'll load my experiment data from {data_path}")
        self.matrix = self.load_matrix(fname='firstgen.m')
        self.matrix_std = self.load_matrix(fname='firstgen_std.m')

    def load_matrix(self, data_path: Optional[Union[str, Path]] = None,
                    fname: str = 'firstgen.m'):
        """ Load experimental matrix

        Args:
            data_path: Path to load experimental data
            fname: Filename, defaults to 'firstgen.m'
        """
        data_path = self.data_path if data_path is None else Path(data_path)
        mat = Matrix(path=data_path / fname)
        if mat.Eg[-1] > 1000:
            logger.debug("Recalibrate matrix.Eg to MeV")
            mat.Eg /= 1e3
        if mat.Ex[-1] > 1000:
            logger.debug("Recalibrate matrix.Ex to MeV")
            mat.Ex /= 1e3
        return mat
