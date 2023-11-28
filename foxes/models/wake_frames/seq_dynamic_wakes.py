import numpy as np
from scipy.spatial.distance import cdist

from foxes.core import WakeFrame
from foxes.utils import wd2uv
from foxes.core.data import Data
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
        mdata: foxes.core.Data
            The model data
        fdata: foxes.core.Data
            The farm data

        Returns
        -------
        order: numpy.ndarray
            The turbine order, shape: (n_states, n_turbines)

        """

        # prepare:
        n_states = fdata.n_states
        n_turbines = algo.n_turbines
        pdata = Data.from_points(points=fdata[FV.TXYH])

        # calculate streamline x coordinates for turbines rotor centre points:
        # n_states, n_turbines_source, n_turbines_target
        coosx = np.zeros((n_states, n_turbines, n_turbines), dtype=FC.DTYPE)
        for ti in range(n_turbines):
            coosx[:, ti, :] = self.get_wake_coos(
                algo, mdata, fdata, pdata, np.full(n_states, ti)
            )[..., 0]

        # derive turbine order:
        # TODO: Remove loop over states
        order = np.zeros((n_states, n_turbines), dtype=FC.ITYPE)
        for si in range(n_states):
            order[si] = np.lexsort(keys=coosx[si])

        return order

    def get_wake_coos(self, algo, mdata, fdata, pdata, states_source_turbine):
        """
        Calculate wake coordinates.

        Parameters
        ----------
        algo: foxes.core.Algorithm
            The calculation algorithm
        mdata: foxes.core.Data
            The model data
        fdata: foxes.core.Data
            The farm data
        pdata: foxes.core.Data
            The evaluation point data
        states_source_turbine: numpy.ndarray
            For each state, one turbine index for the
            wake causing turbine. Shape: (n_states,)

        Returns
        -------
        wake_coos: numpy.ndarray
            The wake frame coordinates of the evaluation
            points, shape: (n_states, n_points, 3)

        """

        # prepare:
        n_states = 1
        n_points = pdata.n_points
        points = pdata[FC.POINTS]
        stsel = (np.arange(n_states), states_source_turbine)
        tindx = states_source_turbine[0]
        counter = algo.states.counter
        N = counter + 1

        # new wake starts at turbine:
        self._traces_p[counter, tindx] = fdata[FV.TXYH][0, tindx]
        self._traces_l[counter, tindx] = 0

        # transport wakes that originate from previous time steps:
        if counter > 0:
            dxyz = self._traces_v[:counter, tindx] * self._dt[:counter, None]
            self._traces_p[:counter, tindx] += dxyz
            self._traces_l[:counter, tindx] += np.linalg.norm(dxyz, axis=-1)

        # compute wind vectors at wake traces:
        # TODO: dz from U_z is missing here
        hpdata = {
            v: np.zeros((1, N), dtype=FC.DTYPE)
            for v in algo.states.output_point_vars(algo)
        }
        hpdata[FC.POINTS] = self._traces_p[None, :N, tindx]
        hpdims = {FC.POINTS: (FC.STATE, FC.POINT, FC.XYH)}
        hpdims.update({v: (FC.STATE, FC.POINT) for v in hpdata.keys()})
        hpdata = Data(hpdata, hpdims, loop_dims=[FC.STATE, FC.POINT])
        res = algo.states.calculate(algo, mdata, fdata, hpdata)
        self._traces_v[:N, tindx, :2] = wd2uv(res[FV.WD][0], res[FV.WS][0])
        del hpdata, hpdims, res

        # project:
        dists = cdist(points[0], self._traces_p[:N, tindx])
        tri = np.argmin(dists, axis=1)
        del dists
        wcoos = np.full((n_states, n_points, 3), 1e20, dtype=FC.DTYPE)
        wcoos[0, :, 2] = points[0, :, 2] - fdata[FV.TXYH][stsel][0, None, 2]
        delp = points[0, :, :2] - self._traces_p[tri, tindx, :2]
        nx = self._traces_v[tri, tindx, :2]
        nx /= np.linalg.norm(nx, axis=1)[:, None]
        ny = np.concatenate([-nx[:, 1, None], nx[:, 0, None]], axis=1)
        wcoos[0, :, 0] = np.einsum("pd,pd->p", delp, nx) + self._traces_l[tri, tindx]
        wcoos[0, :, 1] = np.einsum("pd,pd->p", delp, ny)

        # turbines that cause wake:
        pdata.add(FC.STATE_SOURCE_TURBINE, states_source_turbine, (FC.STATE,))

        # states that cause wake for each target point:
        pdata.add(FC.STATES_SEL, tri[None, :], (FC.STATE, FC.POINT))

        return wcoos

    def get_wake_modelling_data(
        self,
        algo,
        variable,
        states_source_turbine,
        fdata,
        pdata,
        states0=None,
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
        states_source_turbine: numpy.ndarray
            For each state, one turbine index for the
            wake causing turbine. Shape: (n_states,)
        fdata: foxes.core.Data
            The farm data
        pdata: foxes.core.Data
            The evaluation point data
        states0: numpy.ndarray, optional
            The states of wake creation

        """
        if states0 is None:
            # from previous iteration:
            if not np.all(states_source_turbine == pdata[FC.STATE_SOURCE_TURBINE]):
                raise ValueError(
                    f"Model '{self.name}': Mismatch of 'states_source_turbine'. Expected {list(pdata[FC.STATE_SOURCE_TURBINE])}, got {list(states_source_turbine)}"
                )

            s = pdata[FC.STATES_SEL]
            data = algo.farm_results[variable].to_numpy()

            return data[s, states_source_turbine]

        else:
            return super().get_wake_modelling_data(
                algo, variable, states_source_turbine, fdata, pdata, states0
            )

    def get_centreline_points(self, algo, mdata, fdata, states_source_turbine, x):
        """
        Gets the points along the centreline for given
        values of x.

        Parameters
        ----------
        algo: foxes.core.Algorithm
            The calculation algorithm
        mdata: foxes.core.Data
            The model data
        fdata: foxes.core.Data
            The farm data
        states_source_turbine: numpy.ndarray
            For each state, one turbine index for the
            wake causing turbine. Shape: (n_states,)
        x: numpy.ndarray
            The wake frame x coordinates, shape: (n_states, n_points)

        Returns
        -------
        points: numpy.ndarray
            The centreline points, shape: (n_states, n_points, 3)

        """

        raise NotImplementedError
