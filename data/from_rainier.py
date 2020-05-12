import numpy as np
import ompy as om
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt

def normalize_rows(array: np.ndarray):
    """ Normalize each row to unity """
    norm = array.sum(axis=1)[:, np.newaxis]
    return om.div0(array, norm), norm


def smooth(matrix, resolutionEx=150, resolutionEg=100, _truncate=2):
    if resolutionEx is not None:
        binsize = matrix.Ex[1] - matrix.Ex[0]
        sigma = resolutionEx / 2.3548 / binsize
        matrix.values = gaussian_filter1d(matrix.values, sigma, axis=0,
                                          truncate=_truncate)
    if resolutionEg is not None:
        binsize = matrix.Eg[1] - matrix.Eg[0]
        sigma = resolutionEg / 2.3548 / binsize
        matrix.values = gaussian_filter1d(matrix.values, sigma, axis=1,
                                          truncate=_truncate)


mat = om.Matrix(path="1Gen.m")
smooth(mat)
mat.rebin(axis="Eg", factor=4)
mat.rebin(axis="Ex", factor=4)
# mat.plot()

std = mat.copy()
std.values = np.sqrt(std.values)
mat.values, norm = normalize_rows(mat.values)
std.values = om.div0(std.values, norm)
mat.plot(scale="log")
std.plot(scale="log")

mat.save("firstgen.m")
std.save("firstgen_std.m")

# plt.show()
