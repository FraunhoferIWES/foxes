from __future__ import annotations

import numpy as np
from scipy.spatial.distance import cdist
from typing import TYPE_CHECKING, Any

from foxes.utils import wd2uv
from foxes.core.data import TData
from foxes.config import config
from foxes.algorithms.sequential import Sequential
import foxes.variables as FV
import foxes.constants as FC

from .farm_order import FarmOrder

if TYPE_CHECKING:
    from foxes.core.algorithm import Algorithm
    from foxes.core.data import FData, MData
    from foxes.core.axial_induction_model import AxialInductionModel
    from foxes.core.model import LoadedData, Model


class SeqDynamicWakes(FarmOrder):
    """
    Dynamic wakes for the sequential algorithm.

    Attributes
    ----------
    cl_ipars: dict
        Interpolation parameters for centre line
        point interpolation
    dt_min: float, optional
        The delta t value in minutes,
        if not from timeseries data
    induction: foxes.core.AxialInductionModel
        The induction model

    :group: models.wake_frames.sequential

    """

    def __init__(
        self,
        cl_ipars: dict[str, Any] | None = None,
        dt_min: float | None = None,
        induction: str = "Madsen",
        **kwargs: Any,
    ) -> None:
        """
        Constructor.

        Parameters
        ----------
        cl_ipars: dict
            Interpolation parameters for centre line
            point interpolation
        dt_min: float, optional
            The delta t value in minutes,
            if not from timeseries data
        induction: foxes.core.AxialInductionModel or str
            The induction model
        kwargs: dict, optional
            Additional parameters for the base class

        """
        super().__init__(**kwargs)
        self.cl_ipars = {} if cl_ipars is None else cl_ipars
        self.dt_min = dt_min
        self.induction: str | AxialInductionModel = induction
        self._dt: np.ndarray | None = None
        self._traces_p: np.ndarray | None = None
        self._traces_v: np.ndarray | None = None
        self._traces_l: np.ndarray | None = None

    def __repr__(self) -> str:
        iname = (
            self.induction if isinstance(self.induction, str) else self.induction.name
        )
        return f"{type(self).__name__}(dt_min={self.dt_min}, induction={iname})"

    def sub_models(self) -> list[Model]:
        """
        List of all sub-models

        Returns
        -------
        smdls: list of foxes.core.Model
            All sub models

        """
        return [] if isinstance(self.induction, str) else [self.induction]

    def initialize(
        self,
        algo: Algorithm,
        loaded_data: LoadedData | None = None,
        force: bool = False,
        verbosity: int = 0,
    ) -> LoadedData:
        """
        Initializes the model.

        Parameters
        ----------
        algo: foxes.core.Algorithm
            The calculation algorithm
        loaded_data: dict, optional
            Data that has already been loaded, to be extended by this function.
            Keys are "coords", a dict with entries `dim_name_str -> dim_array`;
            "data_vars", a dict with entries `name_str -> (dim_tuple, data_ndarray)`;
            and "extra_data", a dict with non-array additional data.
        force: bool
            Overwrite existing data
        verbosity: int
            The verbosity level, 0 = silent

        Returns
        -------
        loaded_data: dict
            The loaded data, containing keys "coords", "data_vars", and "extra_data".
            Keys are "coords", a dict with entries `dim_name_str -> dim_array`;
            "data_vars", a dict with entries `name_str -> (dim_tuple, data_ndarray)`;
            and "extra_data", a dict with non-array additional data.

        """
        if isinstance(self.induction, str):
            self.induction = algo.mbook.axial_induction[self.induction]

        loaded_data = super().initialize(
            algo, loaded_data=loaded_data, force=force, verbosity=verbosity
        )

        if not isinstance(algo, Sequential):
            raise TypeError(
                f"Incompatible algorithm type {type(algo).__name__}, expecting {Sequential.__name__}"
            )

        # determine time step:
        times = np.asarray(algo.states.index())
        if self.dt_min is None:
            if not np.issubdtype(times.dtype, np.datetime64):
                raise TypeError(
                    f"{self.name}: Expecting state index of type np.datetime64, found {times.dtype}"
                )
            elif len(times) == 1:
                raise KeyError(
                    f"{self.name}: Expecting 'dt_min' for single step timeseries"
                )
            self._dt = (
                (times[1:] - times[:-1])
                .astype("timedelta64[s]")
                .astype(config.dtype_int)
            )
        else:
            n = max(len(times) - 1, 1)
            dt = np.timedelta64(int(round(self.dt_min * 60)), "s")
            self._dt = np.full(n, dt, dtype="timedelta64[s]").astype(config.dtype_int)

        # init wake traces data:
        n_states = algo.n_states
        n_turbines = algo.n_turbines
        assert n_states is not None and n_turbines is not None
        self._traces_p = np.zeros(
            (n_states, n_turbines, 3), dtype=config.dtype_double
        )
        self._traces_v = np.zeros(
            (n_states, n_turbines, 3), dtype=config.dtype_double
        )
        self._traces_l = np.full(
            (n_states, n_turbines), np.nan, dtype=config.dtype_double
        )
        return loaded_data

    def calc_order(self, algo: Algorithm, mdata: MData, fdata: FData) -> np.ndarray:
        """
        Calculates the order of turbine evaluation.

        This function is executed on a single chunk of data,
        all computations should be based on numpy arrays.

        Parameters
        ----------
        algo: foxes.core.Algorithm
            The calculation algorithm
        mdata: foxes.core.MData
            The model data
        fdata: foxes.core.FData
            The farm data

        Returns
        -------
        order: numpy.ndarray
            The turbine order, shape: (n_states, n_turbines)

        """
        return super().calc_order(algo, mdata, fdata)

    def get_wake_coos(
        self,
        algo: Algorithm,
        mdata: MData,
        fdata: FData,
        tdata: TData,
        downwind_index: int,
    ) -> np.ndarray:
        """
        Calculate wake coordinates of rotor points.

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

        Returns
        -------
        wake_coos: numpy.ndarray
            The wake frame coordinates of the evaluation
            points, shape: (n_states, n_targets, n_tpoints, 3)

        """
        # prepare:
        n_states = 1
        n_targets = tdata.n_targets
        n_tpoints = tdata.n_tpoints
        assert n_targets is not None and n_tpoints is not None
        n_points = n_targets * n_tpoints
        points = tdata[FC.TARGETS][:1].reshape(n_states, n_points, 3)
        counter = algo.states.counter
        N = counter + 1
        traces_l = self._traces_l
        traces_p = self._traces_p
        traces_v = self._traces_v
        dt_data = self._dt
        assert traces_l is not None and traces_p is not None and traces_v is not None
        assert dt_data is not None

        if np.isnan(traces_l[counter, downwind_index]):
            # new wake starts at turbine:
            traces_p[counter, downwind_index][:] = fdata[FV.TXYH][0, downwind_index]
            traces_l[counter, downwind_index] = 0

            # transport wakes that originate from previous time steps:
            if counter > 0:
                dxyz = traces_v[:counter, downwind_index] * dt_data[counter - 1]
                traces_p[:counter, downwind_index] += dxyz
                traces_l[:counter, downwind_index] += np.linalg.norm(dxyz, axis=-1)
                del dxyz

            # compute wind vectors at wake traces:
            # TODO: dz from U_z is missing here
            svrs = algo.states.output_point_vars(algo)
            hpdata = TData.from_points(
                points=traces_p[None, :N, downwind_index], variables=svrs
            )
            res = algo.states.calculate(algo, mdata, fdata, hpdata)
            wd = res[FV.WD][0, :, 0]
            if FV.YAWM in fdata:
                wdfl = algo.wake_deflection
                assert wdfl is not None, "Wake deflection model not initialized"
                wddef = wdfl.get_yaw_alpha_seq(
                    algo,
                    mdata,
                    fdata,
                    hpdata,
                    downwind_index,
                    traces_l[:N, downwind_index],
                )
                if wddef is not None:
                    wd += wddef
                del wddef
            traces_v[:N, downwind_index, :2] = wd2uv(wd, res[FV.WS][0, :, 0])
            del hpdata, res, svrs, wd

        # find nearest wake point:
        dists = cdist(points[0], traces_p[:N, downwind_index])
        tri = np.argmin(dists, axis=1)
        del dists

        # project:
        wcoos: np.ndarray = np.full(
            (n_states, n_points, 3), 1e20, dtype=config.dtype_double
        )
        wcoos[0, :, 2] = points[0, :, 2] - fdata[FV.TXYH][0, downwind_index, None, 2]
        nx = traces_v[tri, downwind_index, :2]
        mv = np.linalg.norm(nx, axis=-1)
        nx /= mv[:, None]
        delp = points[0, :, :2] - traces_p[tri, downwind_index, :2]
        projx = np.einsum("pd,pd->p", delp, nx)
        dt = dt_data[counter] if counter < len(dt_data) else dt_data[-1]
        dx = mv * dt
        sel = (projx > -dx) & (projx < dx)
        if np.any(sel):
            ny = np.concatenate([-nx[:, 1, None], nx[:, 0, None]], axis=1)
            wcoos[0, sel, 0] = projx[sel] + traces_l[tri[sel], downwind_index]
            wcoos[0, sel, 1] = np.einsum("pd,pd->p", delp, ny)[sel]
            del ny
        del delp, projx, mv, dx, nx, sel

        # turbines that cause wake:
        tdata[FC.STATE_SOURCE_ORDERI] = downwind_index

        # states that cause wake for each target point:
        tdata.add(
            FC.STATES_SEL,
            tri[None, :].reshape(n_states, n_targets, n_tpoints),
            (FC.STATE, FC.TARGET, FC.TPOINT),
        )

        return wcoos.reshape(n_states, n_targets, n_tpoints, 3)

    def get_wake_modelling_data(
        self,
        algo: Algorithm,
        variable: str,
        downwind_index: int,
        fdata: FData,
        tdata: TData,
        target: str,
        states0: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Return data that is required for computing the
        wake from source turbines to evaluation points.

        Parameters
        ----------
        algo: foxes.core.Algorithm, optional
            The algorithm, needed for data from previous iteration
        variable: str
            The variable, serves as data key
        downwind_index: int, optional
            The index in the downwind order
        fdata: foxes.core.FData
            The farm data
        tdata: foxes.core.TData
            The target point data
        target: str, optional
            The dimensions identifier for the output,
            FC.STATE_TARGET, FC.STATE_TARGET_TPOINT
        states0: numpy.ndarray, optional
            The states of wake creation

        Returns
        -------
        data: numpy.ndarray
            Data for wake modelling, shape:
            (n_states, n_turbines) or (n_states, n_target)

        """
        if states0 is None and FC.STATE_SOURCE_ORDERI in tdata:
            # from previous iteration:
            if downwind_index != tdata[FC.STATE_SOURCE_ORDERI]:
                raise ValueError(
                    f"Model '{self.name}': Mismatch of '{FC.STATE_SOURCE_ORDERI}'. Expected {tdata[FC.STATE_SOURCE_ORDERI]}, got {downwind_index}"
                )

            n_states = 1
            n_targets = tdata.n_targets
            n_tpoints = tdata.n_tpoints
            n_points = n_targets * n_tpoints
            counter = algo.states.counter

            s = tdata[FC.STATES_SEL][0].reshape(n_points)
            fresults = algo.farm_results_downwind
            assert fresults is not None, "Missing farm_results_downwind"
            data = fresults[variable].to_numpy()
            data[counter] = fdata[variable][0]
            data = data[s, downwind_index].reshape(n_states, n_targets, n_tpoints)

            if target == FC.STATE_TARGET:
                if n_tpoints == 1:
                    data = data[:, :, 0]
                else:
                    data = np.einsum("stp,p->st", data, tdata[FC.TWEIGHTS])
                return data
            elif target == FC.STATE_TARGET_TPOINT:
                return data
            else:
                raise ValueError(
                    f"Cannot handle target '{target}', choices are {FC.STATE_TARGET}, {FC.STATE_TARGET_TPOINT}"
                )

        else:
            return super().get_wake_modelling_data(
                algo, variable, downwind_index, fdata, tdata, target, states0
            )

    def get_centreline_points(
        self,
        algo: Algorithm,
        mdata: MData,
        fdata: FData,
        downwind_index: int,
        x: np.ndarray,
    ) -> np.ndarray:
        """
        Gets the points along the centreline for given
        values of x.

        Parameters
        ----------
        algo: foxes.core.Algorithm
            The calculation algorithm
        mdata: foxes.core.MData
            The model data
        fdata: foxes.core.FData
            The farm data
        downwind_index: int
            The index in the downwind order
        x: numpy.ndarray
            The wake frame x coordinates, shape: (n_states, n_points)

        Returns
        -------
        points: numpy.ndarray
            The centreline points, shape: (n_states, n_points, 3)

        """
        raise NotImplementedError
