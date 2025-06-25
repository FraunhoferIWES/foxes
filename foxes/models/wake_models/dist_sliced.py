from abc import abstractmethod
import numpy as np

from foxes.core import SingleTurbineWakeModel
from foxes.config import config
import foxes.variables as FV

class DistSlicedWakeModel(SingleTurbineWakeModel):
    """
    Abstract base class for wake models for which
    the x-denpendency can be separated from the
    yz-dependency.

    The multi-yz ability is used by the `PartialDistSlicedWake`
    partial wakes model.

    :group: models.wake_models

    """
    
    def new_wake_deltas(self, algo, mdata, fdata, tdata):
        """
        Creates new empty wake delta arrays.

        Parameters
        ----------
        algo: foxes.core.Algorithm
            The calculation algorithm
        mdata: foxes.core.MData
            The model data
        fdata: foxes.core.FData
            The farm data
        tdata: foxes.core.TData
            The target point data

        Returns
        -------
        wake_deltas: dict
            Key: variable name, value: The zero filled
            wake deltas, shape: (n_states, n_targets, n_tpoints, ...)

        """
        if self.has_uv:
            duv = np.zeros(
                (tdata.n_states, tdata.n_targets, tdata.n_tpoints, 2), 
                dtype=config.dtype_double,
            )
            return {FV.UV: duv}
        else:
            dws = np.zeros(
                (tdata.n_states, tdata.n_targets, tdata.n_tpoints), 
                dtype=config.dtype_double,
            )
            return {FV.WS: dws}
    
    @abstractmethod
    def calc_wakes_x_yz(
        self,
        algo,
        mdata,
        fdata,
        tdata,
        downwind_index,
        x,
        yz,
    ):
        """
        Calculate wake deltas.

        Parameters
        ----------
        algo: foxes.core.Algorithm
            The calculation algorithm
        mdata: foxes.core.MData
            The model data
        fdata: foxes.core.FData
            The farm data
        tdata: foxes.core.TData
            The target point data
        downwind_index: int
            The index in the downwind order
        x: numpy.ndarray
            The x values, shape: (n_states, n_targets)
        yz: numpy.ndarray
            The yz values for each x value, shape:
            (n_states, n_targets, n_yz_per_target, 2)

        Returns
        -------
        wdeltas: dict
            The wake deltas. Key: variable name str,
            value: numpy.ndarray, shape: (n_st_sel, n_yz_per_target)
        st_sel: numpy.ndarray of bool
            The state-target selection, for which the wake
            is non-zero, shape: (n_states, n_targets)

        """
        pass

    def contribute(
        self,
        algo,
        mdata,
        fdata,
        tdata,
        downwind_index,
        wake_coos,
        wake_deltas,
    ):
        """
        Modifies wake deltas at target points by
        contributions from the specified wake source turbines.

        Parameters
        ----------
        algo: foxes.core.Algorithm
            The calculation algorithm
        mdata: foxes.core.MData
            The model data
        fdata: foxes.core.FData
            The farm data
        tdata: foxes.core.TData
            The target point data
        downwind_index: int
            The index of the wake causing turbine
            in the downwind order
        wake_coos: numpy.ndarray
            The wake frame coordinates of the evaluation
            points, shape: (n_states, n_targets, n_tpoints, 3)
        wake_deltas: dict
            The wake deltas. Key: variable name,
            value: numpy.ndarray with shape
            (n_states, n_targets, n_tpoints, ...)

        """
        x = wake_coos[:, :, 0, 0]
        yz = wake_coos[..., 1:3]

        wdeltas, st_sel = self.calc_wakes_x_yz(
            algo, mdata, fdata, tdata, downwind_index, x, yz
        )

        if self.affects_ws and self.has_uv:
            assert self.has_vector_wind_superp, f"Wake model {self.name}: Missing vector wind superposition, got '{self.wind_superposition}'"
            if FV.UV in wdeltas or FV.WS in wdeltas:
                if not FV.UV in wdeltas:
                    self.vec_superp.wdeltas_ws2uv(
                        algo, fdata, tdata, downwind_index, wdeltas, st_sel)
                wake_deltas[FV.UV] = self.vec_superp.add_wake_vector(
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
            