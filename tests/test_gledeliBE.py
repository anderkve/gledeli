import pytest
import numpy as np
import copy

from pathlib import Path
file_path = Path(__file__).resolve()
import sys  # noqa
sys.path.append(str(file_path.parents[1]))

from gledeliBE import glede  # noqa
from gledeliBE import set_model_names, get_results, log_stream # noqa

# set up different model parameters
model_nameCT = "NLDModelCT_and_discretes"
nld_parsCT = {"T": 0.61, "Eshift": -1.02, "Ecrit": 2.3}

model_nameBSFG = "NLDModelBSFG_and_discretes"
nld_parsBSFG = {"NLDa": 17, "Eshift": 0.30, "Ecrit": 1.98}

gsf_pars = {}
gsf_pars['p1'], gsf_pars['p2'], gsf_pars['p3'] = np.array([12.68, 236., 3.])  # noqa
gsf_pars['p4'], gsf_pars['p5'], gsf_pars['p6'] = np.array([15.2,  175., 2.2]) # noqa
gsf_pars['p7'], gsf_pars['p8'], gsf_pars['p9'] = np.array([6.42, 4.2, 1.9])
gsf_pars['p10'], gsf_pars['p11'], gsf_pars['p12'] = np.array([10.6, 30., 4.9])
gsf_pars['p13'], gsf_pars['p14'], gsf_pars['p15'] = np.array([2.81, 0.54, 0.76]) # noqa
gsf_pars['T'] = 0.61

model_nameGLO = "GSF_GLO_CT_Model20"
gsf_pars_GLO = copy.deepcopy(gsf_pars)

model_nameEGLO = "GSF_EGLO_CT_Model20"
gsf_pars_EGLO = copy.deepcopy(gsf_pars)
gsf_pars_EGLO['epsilon_0'] = 5.
gsf_pars_EGLO['k'] = 2.


def init_glede(model_nameNLD, model_nameGSF, nld_pars, gsf_pars, cutoff):
    glede.lnlike_cutoff = cutoff
    model_names = [model_nameNLD, model_nameGSF]
    set_model_names(model_names)
    glede.nld_pars = nld_pars
    glede.gsf_pars = gsf_pars


@pytest.mark.parametrize(
                "model_nameNLD,model_nameGSF,nld_pars,gsf_pars,cutoff",
                 [(model_nameCT, model_nameGLO,
                   nld_parsCT, gsf_pars_GLO, -1e8),
                  (model_nameBSFG, model_nameGLO,
                  nld_parsBSFG, gsf_pars_GLO, -1e8),
                  (model_nameCT, model_nameEGLO,
                   nld_parsCT, gsf_pars_EGLO, -1e8),
                  (model_nameBSFG, model_nameEGLO,
                  nld_parsBSFG, gsf_pars_EGLO, -1e8)]) # noqa
def test_normal_run(model_nameNLD, model_nameGSF,
                    nld_pars, gsf_pars, cutoff):
    init_glede(model_nameNLD, model_nameGSF, nld_pars, gsf_pars, cutoff)
    glede.run()
    results = get_results()
    print(results)
    for key in ['loglike', 'D0_model', 'Gg_model']:
        assert key in results
        assert np.isfinite(results[key])


@pytest.mark.parametrize(
                "model_nameNLD,model_nameGSF,nld_pars,gsf_pars,cutoff",
                 [(model_nameCT, model_nameGLO,
                   nld_parsCT, gsf_pars_GLO, -1),
                  (model_nameBSFG, model_nameGLO,
                  nld_parsBSFG, gsf_pars_GLO, -1),
                  (model_nameCT, model_nameEGLO,
                   nld_parsCT, gsf_pars_EGLO, -1),
                  (model_nameBSFG, model_nameEGLO,
                  nld_parsBSFG, gsf_pars_EGLO, -1)]) # noqa
def test_raise_cutoff_error(model_nameNLD, model_nameGSF,
                            nld_pars, gsf_pars, cutoff):
    init_glede(model_nameNLD, model_nameGSF, nld_pars, gsf_pars, cutoff)
    message = r"lnlike: [+-]*[\d.]+(?:e[+-]?\d+)* below cutoff"
    with pytest.raises(AssertionError, match=message):
        glede.run()
