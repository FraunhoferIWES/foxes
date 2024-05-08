import numpy as np
from scipy.spatial.distance import cdist

from foxes.core import WakeFrame
from foxes.utils import wd2uv
from foxes.core.data import TData
import foxes.variables as FV
import foxes.constants as FC
from foxes.algorithms import Sequential


class SeqDynamicWakes(WakeFrame):
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

    :group: models.wake_frames.sequential

    """

    def __init__(self, cl_ipars={}, dt_min=None):
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

        """
        super().__init__()
        self.cl_ipars = cl_ipars
        self.dt_min = dt_min

    def __repr__(self):
        return f"{type(self).__name__}(dt_min={self.dt_min})"

    def initialize(self, algo, verbosity=0):
        """
        Initializes the model.

        Parameters
        ----------
        algo: foxes.core.Algorithm
            The calculation algorithm
        verbosity: int
            The verbosity level, 0 = silent

        """
        super().initialize(algo, verbosity)
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
                (times[1:] - times[:-1]).astype("timedelta64[s]").astype(FC.ITYPE)
            )
        else:
            n = max(len(times) - 1, 1)
            self._dt = np.full(n, self.dt_min * 60, dtype="timedelta64[s]").astype(
                FC.ITYPE
            )

        # init wake traces data:
        self._traces_p = np.zeros((algo.n_states, algo.n_turbines, 3), dtype=FC.DTYPE)
        self._traces_v = np.zeros((algo.n_states, algo.n_turbines, 3), dtype=FC.DTYPE)
        self._traces_l = np.zeros((algo.n_states, algo.n_turbines), dtype=FC.DTYPE)

    def calc_order(self, algo, mdata, fdata):
        """ "
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
        # prepare:
        n_states = fdata.n_states
        n_turbines = algo.n_turbines
        tdata = TData.from_points(points=fdata[FV.TXYH])

        # calculate streamline x coordinates for turbines rotor centre points:
        # n_states, n_turbines_source, n_turbines_target
        coosx = np.zeros((n_states, n_turbines, n_turbines), dtype=FC.DTYPE)
        for ti in range(n_turbines):
            coosx[:, ti, :] = self.get_wake_coos(algo, mdata, fdata, tdata, ti)[
                :, :, 0, 0
            ]

        # derive turbine order:
        # TODO: Remove loop over states
        order = np.zeros((n_states, n_turbines), dtype=FC.ITYPE)
        for si in range(n_states):
            order[si] = np.lexsort(keys=coosx[si])

        return order

    def get_wake_coos(
        self,
        algo,
        mdata,
        fdata,
        tdata,
        downwind_index,
    ):
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
            in the downwnd order

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
        n_points = n_targets * n_tpoints
        points = tdata[FC.TARGETS].reshape(n_states, n_points, 3)
        counter = algo.states.counter
        N = counter + 1

        # new wake starts at turbine:
        self._traces_p[counter, downwind_index] = fdata[FV.TXYH][0, downwind_index]
        self._traces_l[counter, downwind_index] = 0

        # transport wakes that originate from previous time steps:
        if counter > 0:
            dxyz = self._traces_v[:counter, downwind_index] * self._dt[:counter, None]
            self._traces_p[:counter, downwind_index] += dxyz
            self._traces_l[:counter, downwind_index] += np.linalg.norm(dxyz, axis=-1)

        # compute wind vectors at wake traces:
        # TODO: dz from U_z is missing here
        hpdata = {
            v: np.zeros((1, N, 1), dtype=FC.DTYPE)
            for v in algo.states.output_point_vars(algo)
        }
        hpdims = {v: (FC.STATE, FC.TARGET, FC.TPOINT) for v in hpdata.keys()}
        hpdata = TData.from_points(
            points=self._traces_p[None, :N, downwind_index],
            data=hpdata,
            dims=hpdims,
        )
        res = algo.states.calculate(algo, mdata, fdata, hpdata)
        self._traces_v[:N, downwind_index, :2] = wd2uv(
            res[FV.WD][0, :, 0], res[FV.WS][0, :, 0]
        )
        del hpdata, hpdims, res

        # project:
        dists = cdist(points[0], self._traces_p[:N, downwind_index])
        tri = np.argmin(dists, axis=1)
        del dists
        wcoos = np.full((n_states, n_points, 3), 1e20, dtype=FC.DTYPE)
        wcoos[0, :, 2] = points[0, :, 2] - fdata[FV.TXYH][:, downwind_index][0, None, 2]
        delp = points[0, :, :2] - self._traces_p[tri, downwind_index, :2]
        nx = self._traces_v[tri, downwind_index, :2]
        nx /= np.linalg.norm(nx, axis=1)[:, None]
        ny = np.concatenate([-nx[:, 1, None], nx[:, 0, None]], axis=1)
        wcoos[0, :, 0] = (
            np.einsum("pd,pd->p", delp, nx) + self._traces_l[tri, downwind_index]
        )
        wcoos[0, :, 1] = np.einsum("pd,pd->p", delp, ny)

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
        algo,
        variable,
        downwind_index,
        fdata,
        tdata,
        target,
        states0=None,
        upcast=False,
    ):
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
        upcast: bool
            Flag for ensuring targets dimension,
            otherwise dimension 1 is entered

        Returns
        -------
        data: numpy.ndarray
            Data for wake modelling, shape:
            (n_states, n_turbines) or (n_states, n_target)
        dims: tuple
            The data dimensions

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

            s = tdata[FC.STATES_SEL][0].reshape(n_points)
            data = algo.farm_results[variable].to_numpy()
            data = data[s, downwind_index].reshape(n_states, n_targets, n_tpoints)

            if target == FC.STATE_TARGET:
                if n_tpoints == 1:
                    data = data[:, :, 0]
                else:
                    data = np.einsum("stp,p->st", data, tdata[FC.TWEIGHTS])
                return data, (FC.STATE, FC.TARGET)
            elif target == FC.STATE_TARGET_TPOINT:
                return data, (FC.STATE, FC.TARGET, FC.TPOINT)
            else:
                raise ValueError(
                    f"Cannot handle target '{target}', choices are {FC.STATE_TARGET}, {FC.STATE_TARGET_TPOINT}"
                )

        else:
            return super().get_wake_modelling_data(
                algo, variable, downwind_index, fdata, tdata, target, states0, upcast
            )

    def get_centreline_points(self, algo, mdata, fdata, downwind_index, x):
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
