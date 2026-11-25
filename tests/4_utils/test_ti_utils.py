import numpy as np

from foxes.utils.ti_utils import tke2ti


def test_tke2ti_isotropic_factor():
    # isotropic turbulence: TKE = 3/2 * sigma^2  ->  TI = sqrt(2/3 * TKE) / WS
    tke = np.array([1.5, 0.375])
    ws = np.array([10.0, 5.0])
    assert np.allclose(tke2ti(tke, ws), [0.1, 0.1])


def test_tke2ti_max_ti():
    assert np.allclose(tke2ti(np.array([150.0]), np.array([10.0]), max_ti=0.3), [0.3])