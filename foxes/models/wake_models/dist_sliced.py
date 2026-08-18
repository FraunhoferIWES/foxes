from __future__ import annotations

from abc import abstractmethod
import numpy as np
from typing import TYPE_CHECKING

from foxes.core import SingleTurbineWakeModel
from foxes.config import config
import foxes.variables as FV

if TYPE_CHECKING:
    from foxes.core.algorithm import Algorithm
    from foxes.core.data import FData, MData, TData


class DistSlicedWakeModel(SingleTurbineWakeModel):
    """
    Abstract base class for wake models for which
    the x-denpendency can be separated from the
    yz-dependency.

    The multi-yz ability is used by the `PartialDistSlicedWake`
    partial wakes model.


    """

    def new_wake_deltas(
        self, algo: Algorithm, mdata: MData, fdata: FData, tdata: TData
    ) -> dict[str, np.ndarray]:
        """
        Creates new empty wake delta arrays.

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

        Returns
        -------
        wake_deltas
            Key: variable name, value: The zero filled
            wake deltas, shape: (n_states, n_targets, n_tpoints, ...)

        """
        n_states = tdata.n_states
        n_targets = tdata.n_targets
        n_tpoints = tdata.n_tpoints
        assert n_states is not None and n_targets is not None and n_tpoints is not None
        if self.has_uv:
            duv: np.ndarray = np.zeros(
                (n_states, n_targets, n_tpoints, 2),
                dtype=config.dtype_double,
            )
            return {FV.UV: duv}
        else:
            dws: np.ndarray = np.zeros(
                (n_states, n_targets, n_tpoints),
                dtype=config.dtype_double,
            )
            return {FV.WS: dws}

    @abstractmethod
    def calc_wakes_x_yz(
        self,
        algo: Algorithm,
        mdata: MData,
        fdata: FData,
        tdata: TData,
        downwind_index: int,
        x: np.ndarray,
        yz: np.ndarray,
    ) -> tuple[dict[str, np.ndarray], np.ndarray]:
        """
        Calculate wake deltas.

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
            The index in the downwind order
        x
            The x values, shape: (n_states, n_targets)
        yz
            The yz values for each x value, shape:
            (n_states, n_targets, n_yz_per_target, 2)

        Returns
        -------
        wdeltas
            The wake deltas. Key: variable name str,
            value
        st_sel
            The state-target selection, for which the wake
            is non-zero, shape: (n_states, n_targets)

        """
        pass

    def contribute(
        self,
        algo: Algorithm,
        mdata: MData,
        fdata: FData,
        tdata: TData,
        downwind_index: int,
        wake_coos: np.ndarray,
        wake_deltas: dict[str, np.ndarray],
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
        wake_coos
            The wake frame coordinates of the evaluation
            points, shape: (n_states, n_targets, n_tpoints, 3)
        wake_deltas
            The wake deltas. Key: variable name,
            value
            (n_states, n_targets, n_tpoints, ...)

        """
        x = wake_coos[:, :, 0, 0]
        yz = wake_coos[..., 1:3]

        wdeltas, st_sel = self.calc_wakes_x_yz(
            algo, mdata, fdata, tdata, downwind_index, x, yz
        )

        if self.affects_ws and self.has_uv:
            assert self.has_vector_wind_superp, (
                f"Wake model {self.name}: Missing vector wind superposition, got '{self.wind_superposition}'"
            )
            vec_superp = self.vec_superp
            assert vec_superp is not None
            if FV.UV in wdeltas or FV.WS in wdeltas:
                if FV.UV not in wdeltas:
                    vec_superp.wdeltas_ws2uv(
                        algo, fdata, tdata, downwind_index, wdeltas, st_sel
                    )
                wake_deltas[FV.UV] = vec_superp.add_wake_vector(
                    algo,
                    mdata,
                    fdata,
                    tdata,
                    downwind_index,
                    st_sel,
                    wake_deltas[FV.UV],
                    wdeltas.pop(FV.UV),
                )

        for v, hdel in wdeltas.items():
            try:
                superp = self.superp[v]
            except KeyError:
                raise KeyError(
                    f"Model '{self.name}': Missing wake superposition entry for variable '{v}', found {sorted(list(self.superp.keys()))}"
                )

            wake_deltas[v] = superp.add_wake(
                algo,
                mdata,
                fdata,
                tdata,
                downwind_index,
                st_sel,
                v,
                wake_deltas[v],
                hdel,
            )
