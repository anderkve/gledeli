import pytest
import numpy as np
from numpy.testing import assert_equal, assert_almost_equal
import copy

from pathlib import Path
file_path = Path(__file__).resolve()
import sys  # noqa
sys.path.append(str(file_path.parents[1]))

from gledeli.create_gsf import CreateGSF, fGLO_CT, fEGLO_CT  # noqa


@pytest.mark.parametrize(
                "E0,sigma0,Gamma0,T",
                 [(10., 1., 2., 0.5),
                  (15., 100., 20., 1.)]) # noqa
def test_EGLO_to_GLO(E0, sigma0, Gamma0, T):
    E = np.linspace(0.1, 20)
    EGLO = fEGLO_CT(E, E0, sigma0, Gamma0, T)
    GLO = fGLO_CT(E, E0, sigma0, Gamma0, T)
    assert_equal(EGLO, GLO)


@pytest.mark.parametrize(
                "E0,sigma0,Gamma0,T,epsilon_0,k",
                 [(10., 250., 2., 0.5, 4.5, 2.),
                  (15., 300., 3., 1., 4.5, 3.),
                  (15., 300., 4., 1., 5, 1.1)
                  ]) # noqa
def test_EGLO_to_GLO_unequal(E0, sigma0, Gamma0, T, epsilon_0, k):
    E = np.linspace(0.1, 20)
    EGLO = fEGLO_CT(E, E0, sigma0, Gamma0, T, epsilon_0, k)
    GLO = fGLO_CT(E, E0, sigma0, Gamma0, T)

    # strength functions are on the order 10^-7; set to -9 for small k's
    with pytest.raises(AssertionError):
        assert_almost_equal(EGLO, GLO, decimal=9)


# parameters for test below
gsf_pars = {}
gsf_pars['p1'], gsf_pars['p2'], gsf_pars['p3'] = np.array([12.68, 236., 3.])  # noqa
gsf_pars['p4'], gsf_pars['p5'], gsf_pars['p6'] = np.array([15,  0., 1]) # noqa
gsf_pars['p7'], gsf_pars['p8'], gsf_pars['p9'] = np.array([6, 0, 1])
gsf_pars['p10'], gsf_pars['p11'], gsf_pars['p12'] = np.array([6, 0, 1])
gsf_pars['p13'], gsf_pars['p14'], gsf_pars['p15'] = np.array([6, 0, 1]) # noqa
gsf_pars['T'] = 0.61
gsf_pars_GLO = copy.deepcopy(gsf_pars)

gsf_pars_EGLO = copy.deepcopy(gsf_pars)
gsf_pars_EGLO['epsilon_0'] = 5.
gsf_pars_EGLO['k'] = 2.


@pytest.mark.parametrize(
                "gdr_model, gsf_pars",
                 [('GLO', gsf_pars_GLO),
                   ('EGLO', gsf_pars_EGLO)]) # noqa
def test_model_selection(gdr_model, gsf_pars):
    gsf = CreateGSF(pars=gsf_pars)
    gsf.gdr_model = gdr_model

    x = np.linspace(0, 20)
    gsf_create = gsf.create(x).values

    if gdr_model == "GLO":
        function = fGLO_CT(x, gsf_pars['p1'], gsf_pars['p2'],
                           gsf_pars['p3'], gsf_pars['T'])
    elif gdr_model == "EGLO":
        function = fEGLO_CT(x, gsf_pars['p1'], gsf_pars['p2'],
                            gsf_pars['p3'], gsf_pars['T'],
                            gsf_pars['epsilon_0'], gsf_pars_EGLO['k'])
    assert_equal(gsf_create, function)
