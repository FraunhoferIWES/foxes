from __future__ import annotations

import numpy as np
from typing import TYPE_CHECKING, cast

from foxes.models.wake_models.axisymmetric import AxisymmetricWakeModel
from foxes.utils.two_circles import calc_area
from foxes.config import config
import foxes.variables as FV
import foxes.constants as FC

from .centre import PartialCentre

if TYPE_CHECKING:
    from foxes.core.algorithm import Algorithm
    from foxes.core.data import FData, MData, TData
    from foxes.core.wake_model import WakeModel


class PartialAxiwake(PartialCentre):
    """
    Partial wake calculation for axial wake models.

    The basic idea is that the x-dependent part of
    the wake model is evaluated only once, and the radial
    part then for `n` radii that cover the target rotor discs.

    The latter results are then weighted according to the overlap
    of radial wake circle area deltas and the target rotor disc area.

    Attributes
    ----------
    n
        The number of radial evaluation points


    """

    def __init__(self, n: int = 6) -> None:
        """
        Constructor.

        Parameters
        ----------
        n
            The number of radial evaluation points

        """
        super().__init__()
        self.n = n

    def __repr__(self) -> str:
        return f"{type(self).__name__}(n={self.n})"

    def check_wmodel(self, wmodel: WakeModel, error: bool = True) -> bool:
        """
        Checks the wake model type

        Parameters
        ----------
        wmodel
            The wake model to be tested
        error
            Flag for raising TypeError

        Returns
        -------
        chk
            True if wake model is compatible

        """
        if not isinstance(wmodel, AxisymmetricWakeModel):
            if error:
                raise TypeError(
                    f"Partial wakes '{self.name}': Cannot be applied to wake model '{wmodel.name}', since not an AxisymmetricWakeModel"
                )
            return False
        return True

    def contribute(
        self,
        algo: Algorithm,
        mdata: MData,
        fdata: FData,
        tdata: TData,
        downwind_index: int,
        wake_deltas: dict[str, np.ndarray],
        wmodel: WakeModel,
    ) -> None:
        """
        Modifies wake deltas at target points by
        contributions from the specified wake source turbines.

        Parameters
        ----------
        algo
            The calculation algorithm
        mdata
            The model data
        fdata
            The farm data
        tdata
            The target point data
        downwind_index
            The index of the wake causing turbine
            in the downwind order
        wake_deltas
            The wake deltas. Key: variable name,
            value
            (n_states, n_targets, n_tpoints, ...)

        """
        # check:
        self.check_wmodel(wmodel, error=True)
        wmodel = cast(AxisymmetricWakeModel, wmodel)

        # prepare:
        n_states = mdata.n_states
        n_targets = tdata.n_targets
        assert n_states is not None and n_targets is not None

        # get D:
        D = self.get_data(
            FV.D,
            FC.STATE_TARGET,
            lookup="w",
            algo=algo,
            fdata=fdata,
            tdata=tdata,
            downwind_index=downwind_index,
            upcast=True,
        )

        # calc coordinates to rotor centres:
        wcoos = algo.wake_frame.get_wake_coos(algo, mdata, fdata, tdata, downwind_index)

        # prepare x and r coordinates:
        x = wcoos[..., 0, 0]
        n = wcoos[..., 0, 1:3]
        R = np.linalg.norm(n, axis=-1)
        r: np.ndarray = np.zeros(
            (n_states, n_targets, self.n), dtype=config.dtype_double
        )
        del wcoos

        # prepare circle section area calculation:
        A: np.ndarray = np.zeros(
            (n_states, n_targets, self.n), dtype=config.dtype_double
        )
        weights = np.zeros_like(A)

        # get normalized 2D vector between rotor and wake centres:
        sel = R > 0
        if np.any(sel):
            n[sel] /= R[sel][:, None]
        if np.any(~sel):
            n[:, :, 0][~sel] = 1

        # case wake centre outside rotor disk:
        sel = (x > 1e-8) & (R > D / 2)
        if np.any(sel):
            n_sel = np.sum(sel)
            Rsel = np.zeros((n_sel, self.n + 1), dtype=config.dtype_double)
            Rsel[:] = R[sel][:, None]
            Dsel = D[sel][:, None]

            # equal delta R2:
            R1 = np.zeros((n_sel, self.n + 1), dtype=config.dtype_double)
            R1[:] = Dsel / 2
            steps = np.linspace(0.0, 1.0, self.n + 1, endpoint=True) - 0.5
            R2 = np.zeros_like(R1)
            R2[:] = Rsel + Dsel * steps[None, :]
            r[sel] = 0.5 * (R2[:, 1:] + R2[:, :-1])

            hA = calc_area(R1, R2, Rsel)
            hA = hA[:, 1:] - hA[:, :-1] + 1e-15

            weights[sel] = hA / np.sum(hA, axis=-1)[:, None]
            del hA, Rsel, Dsel, R1, R2

        # case wake centre inside rotor disk:
        sel = (x > 0) & (R < D / 2)
        if np.any(sel):
            n_sel = np.sum(sel)
            Rsel = np.zeros((n_sel, self.n + 1), dtype=config.dtype_double)
            Rsel[:] = R[sel][:, None]
            Dsel = D[sel][:, None]

            # equal delta R2:
            R1 = np.zeros((n_sel, self.n + 1), dtype=config.dtype_double)
            R1[:, 1:] = Dsel / 2
            R2 = np.zeros_like(R1)
            # R2[:, 1:] = Rsel[:, :-1] + Dsel/2
            # R2[:]    *= np.linspace(0., 1, self.n + 1, endpoint=True)[None, :]
            R2[:, 1:] = (Rsel[:, :-1] + Dsel / 2) / (self.n - 0.5)
            R2[:, 1:] *= (
                0.5 + np.linspace(0.0, self.n - 1, self.n, endpoint=True)[None, :]
            )
            hr = 0.5 * (R2[:, 1:] + R2[:, :-1])
            hr[:, 0] = 0.0
            r[sel] = hr

            hA = calc_area(R1, R2, Rsel)
            hA = hA[:, 1:] - hA[:, :-1]
            weights[sel] = hA / np.sum(hA, axis=-1)[:, None]
            del hA, hr, Rsel, Dsel, R1, R2

        # evaluate wake model:
        wdeltas, st_sel = wmodel.calc_wakes_x_r(
            algo, mdata, fdata, tdata, downwind_index, x, r
        )

        # run superposition models:
        if wmodel.affects_ws and wmodel.has_uv:
            assert wmodel.has_vector_wind_superp, (
                f"{self.name}: Expecting vector wind superposition in wake model '{wmodel.name}', got '{wmodel.wind_superposition}'"
            )
            vec_superp = wmodel.vec_superp
            assert vec_superp is not None
            if FV.WS in wdeltas or FV.UV in wdeltas:
                if FV.UV not in wdeltas:
                    vec_superp.wdeltas_ws2uv(
                        algo, fdata, tdata, downwind_index, wdeltas, st_sel
                    )
                duv = np.einsum("snd,sn->sd", wdeltas.pop(FV.UV), weights[st_sel])
                wake_deltas[FV.UV] = vec_superp.add_wake_vector(
                    algo,
                    mdata,
                    fdata,
                    tdata,
                    downwind_index,
                    st_sel,
                    wake_deltas[FV.UV],
                    duv[:, None],
                )
                del duv
            for v in [FV.WS, FV.WD, FV.UV]:
                if v in wdeltas:
                    del wdeltas[v]

        for v, wdel in wdeltas.items():
            try:
                superp = wmodel.superp[v]
            except KeyError:
                s = {v: m.name for v, m in wmodel.superp.items()}
                raise KeyError(
                    f"Model '{self.name}': Missing wake superposition entry for variable '{v}' in wake model '{wmodel.name}', found {s}"
                )

            d = np.einsum("sn,sn->s", wdel, weights[st_sel])

            wake_deltas[v] = superp.add_wake(
                algo,
                mdata,
                fdata,
                tdata,
                downwind_index,
                st_sel,
                v,
                wake_deltas[v],
                d[:, None],
            )
