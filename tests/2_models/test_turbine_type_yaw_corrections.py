import warnings

import numpy as np

from foxes.core.turbine_type import TurbineType


class _DummyTurbineType(TurbineType):
    def output_farm_vars(self, algo):
        return []

    def calculate(self, algo, mdata, fdata, st_sel):
        return {}

    def needs_rews2(self):
        return False

    def needs_rews3(self):
        return False


def test_yaw_corrections_clip_negative_projection_without_runtime_warning():
    ttype = _DummyTurbineType(
        yawm_corr_P="wind_speed",
        yawm_corr_ct="wind_speed",
        yawm_corr_p_P=1.88,
        yawm_corr_p_ct=1.0,
    )

    rews_p = np.array([10.0], dtype=float)
    rews_ct = np.array([10.0], dtype=float)
    yawm = np.array([120.0], dtype=float)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out_rews_p, out_rews_ct, _, _ = ttype.get_rho_yawm_corrections(
            rews_p,
            rews_ct,
            yawm=yawm,
        )

    msgs = [str(w.message) for w in caught]
    assert not any("invalid value encountered" in m for m in msgs)
    assert out_rews_p[0] == 0.0
    assert out_rews_ct[0] == 0.0
