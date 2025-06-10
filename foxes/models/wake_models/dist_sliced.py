from abc import abstractmethod
import numpy as np

from foxes.core import WakeModel, WindVectorWakeSuperposition
from foxes.config import config
from foxes.utils import wd2uv, uv2wd, delta_wd
import foxes.variables as FV

class DistSlicedWakeModel(WakeModel):
    """
    Abstract base class for wake models for which
    the x-denpendency can be separated from the
    yz-dependency.

    The multi-yz ability is used by the `PartialDistSlicedWake`
    partial wakes model.

    Attributes
    ----------
    wind_superposition: str
        The wind superposition model name (vector or compenent model),
        will be looked up in model book
    other_superpositions: dict
        The superpositions for other than (ws, wd) variables. 
        Key: variable name str, value: The wake superposition 
        model name, will be looked up in model book
    vec_superp: foxes.core.WindVectorWakeSuperposition or None
        The wind vector wake superposition model
    superp: dict
        The superposition dict, key: variable name str,
        value: `foxes.core.WakeSuperposition`

    :group: models.wake_models

    """

    def __init__(self, wind_superposition, other_superpositions={}):
        """
        Constructor.

        Parameters
        ----------
        wind_superposition: str
            The wind superposition model name (vector or compenent model),
            will be looked up in model book
        other_superpositions: dict
            The superpositions for other than (ws, wd) variables. 
            Key: variable name str, value: The wake superposition 
            model name, will be looked up in model book

        """
        super().__init__()
        self.wind_superposition = wind_superposition
        self.other_superpositions = other_superpositions
        self.vec_superp = None
        self.superp = {}
        self._has_uv = False

        for v in [FV.WS, FV.WD]:
            assert v not in other_superpositions, f"Wake model '{self.name}': Found variable '{v}' among 'other_superposition' keyword, use 'wind_superposition' instead"

    def sub_models(self):
        """
        List of all sub-models

        Returns
        -------
        smdls: list of foxes.core.Model
            Names of all sub models

        """
        w = [self.vec_superp] if self.vec_superp is not None else []
        return w + list(self.superp.values())

    def initialize(self, algo, verbosity=0, force=False):
        """
        Initializes the model.

        Parameters
        ----------
        algo: foxes.core.Algorithm
            The calculation algorithm
        verbosity: int
            The verbosity level, 0 = silent
        force: bool
            Overwrite existing data

        """
        self.superp = {
            v: algo.mbook.wake_superpositions[s] for v, s in self.other_superpositions.items()
        }

        self.vec_superp = algo.mbook.wake_superpositions[self.wind_superposition]
        self.__has_vector_superp = isinstance(self.vec_superp, WindVectorWakeSuperposition)
        if self.__has_vector_superp:
            self._has_uv = True
        else:
            self.superp[FV.WS] = self.vec_superp
            self.vec_superp = None
            
        super().initialize(algo, verbosity, force)
    
    @property
    def has_vector_wind_superp(self):
        """
        This model uses a wind vector superposition
        
        Returns
        -------
        hasv: bool
            Flag for wind vector superposition
        
        """
        return self.__has_vector_superp

    @property
    def has_uv(self):
        """
        This model uses wind vector data
        
        Returns
        -------
        hasuv: bool
            Flag for wind vector data
        
        """
        return self._has_uv

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
        dwd_defl,
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
        dwd_defl: numpy.ndarray or None
            The wind direction change at the target points 
            in radiants due to wake deflection, 
            shape: (n_states, n_targets)
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
        delta_wd_defl,
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
        delta_wd_defl: numpy.ndarray or None
            The wind direction change at the target points 
            in radiants due to wake deflection, 
            shape: (n_states, n_targets, n_tpoints)
        wake_deltas: dict
            The wake deltas. Key: variable name,
            value: numpy.ndarray with shape
            (n_states, n_targets, n_tpoints, ...)

        """
        dwd = delta_wd_defl[:, :, 0] if delta_wd_defl is not None else None
        x = wake_coos[:, :, 0, 0]
        yz = wake_coos[..., 1:3]

        wdeltas, st_sel = self.calc_wakes_x_yz(
            algo, mdata, fdata, tdata, downwind_index, dwd, x, yz
        )

        if self.has_vector_wind_superp:
            self.vec_superp.wdeltas_ws2uv(algo, fdata, tdata, downwind_index, wdeltas, st_sel)
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

    def finalize_wake_deltas(
        self,
        algo,
        mdata,
        fdata,
        amb_results,
        wake_deltas,
    ):
        """
        Finalize the wake calculation.

        Modifies wake_deltas on the fly.

        Parameters
        ----------
        algo: foxes.core.Algorithm
            The calculation algorithm
        mdata: foxes.core.MData
            The model data
        fdata: foxes.core.FData
            The farm data
        amb_results: dict
            The ambient results, key: variable name str,
            values: numpy.ndarray with shape
            (n_states, n_targets, n_tpoints)
        wake_deltas: dict
            The wake deltas object at the selected target
            turbines. Key: variable str, value: numpy.ndarray
            with shape (n_states, n_targets, n_tpoints)

        """
        for v in wake_deltas.keys():
            if v != FV.UV:
                try:
                    wake_deltas[v] = self.superp[v].calc_final_wake_delta(
                        algo, mdata, fdata, v, amb_results[v], wake_deltas[v]
                    )
                except KeyError:
                    raise KeyError(f"Wake model '{self.name}': Variable '{v}' appears to be modified, missing superposition model")

        if self.has_vector_wind_superp:
            dws, dwd = self.vec_superp.calc_final_wake_delta_uv(
                    algo, mdata, fdata, amb_results, wake_deltas.pop(FV.UV)
                )
            
            wake_deltas[FV.WS] = dws
            wake_deltas[FV.WD] = dwd
            