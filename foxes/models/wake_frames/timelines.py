import numpy as np

from foxes.core import WakeFrame
from foxes.utils import wd2uv
from foxes.core.data import Data
import foxes.variables as FV
import foxes.constants as FC


class Timelines(WakeFrame):
    """
    Dynamic wakes for spatially uniform timeseries states.

    Attributes
    ----------
    max_wake_length: float
        The maximal wake length
    cl_ipars: dict
        Interpolation parameters for centre line
        point interpolation
    dt_min: float, optional
        The delta t value in minutes,
        if not from timeseries data

    :group: models.wake_frames

    """

    def __init__(self, max_wake_length=2e4, cl_ipars={}, dt_min=None):
        """
        Constructor.

        Parameters
        ----------
        max_wake_length: float
            The maximal wake length
        cl_ipars: dict
            Interpolation parameters for centre line
            point interpolation
        dt_min: float, optional
            The delta t value in minutes,
            if not from timeseries data

        """
        super().__init__()
        self.max_wake_length = max_wake_length
        self.cl_ipars = cl_ipars
        self.dt_min = dt_min

    def initialize(self, algo, verbosity=0):
        """
        Initializes the model.

        This includes loading all required data from files. The model
        should return all array type data as part of the idata return
        dictionary (and not store it under self, for memory reasons). This
        data will then be chunked and provided as part of the mdata object
        during calculations.

        Parameters
        ----------
        algo: foxes.core.Algorithm
            The calculation algorithm
        verbosity: int
            The verbosity level, 0 = silent

        Returns
        -------
        idata: dict
            The dict has exactly two entries: `data_vars`,
            a dict with entries `name_str -> (dim_tuple, data_ndarray)`;
            and `coords`, a dict with entries `dim_name_str -> dim_array`

        """
        idata = super().initialize(algo, verbosity)

        if verbosity > 0:
            print(f"{self.name}: Pre-calculating ambient wind vectors")

        # get and check times:
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
            dt = (times[1:] - times[:-1]).astype("timedelta64[s]").astype(FC.ITYPE)
        else:
            n = max(len(times) - 1, 1)
            dt = np.full(n, self.dt_min * 60, dtype="timedelta64[s]").astype(FC.ITYPE)

        # calculate horizontal wind vector in all states:
        self._uv = np.zeros((algo.n_states, 1, 3), dtype=FC.DTYPE)

        # prepare mdata:
        mdata = algo.idata_mem[algo.states.name]["data_vars"]
        mdict = {v: d[1] for v, d in mdata.items()}
        mdims = {v: d[0] for v, d in mdata.items()}
        mdata = Data(mdict, mdims, loop_dims=[FC.STATE])
        del mdict, mdims

        # prepare fdata:
        fdata = Data({}, {}, loop_dims=[FC.STATE])

        # prepare pdata:
        pdata = {
            v: np.zeros((algo.n_states, 1), dtype=FC.DTYPE)
            for v in algo.states.output_point_vars(algo)
        }
        pdata[FC.POINTS] = np.zeros((algo.n_states, 1, 3), dtype=FC.DTYPE)
        pdims = {FC.POINTS: (FC.STATE, FC.POINT, FC.XYH)}
        pdims.update({v: (FC.STATE, FC.POINT) for v in pdata.keys()})
        pdata = Data(pdata, pdims, loop_dims=[FC.STATE, FC.POINT])

        # calculate:
        res = algo.states.calculate(algo, mdata, fdata, pdata)
        if len(dt) == 1:
            self._dxy = wd2uv(res[FV.WD], res[FV.WS])[:, 0, :2] * dt[:, None]
        else:
            self._dxy = wd2uv(res[FV.WD], res[FV.WS])[:-1, 0, :2] * dt[:, None]
            self._dxy = np.insert(self._dxy, 0, self._dxy[0], axis=0)

        """ DEBUG
        import matplotlib.pyplot as plt
        xy = np.array([np.sum(self._dxy[:n], axis=0) for n in range(len(self._dxy))])
        print(xy)
        plt.plot(xy[:, 0], xy[:, 1])
        plt.show()
        quit()
        """

        return idata

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
        n_states = mdata.n_states
        n_points = pdata.n_points
        points = pdata[FC.POINTS]
        stsel = (np.arange(n_states), states_source_turbine)
        rxyz = fdata[FV.TXYH][stsel]

        D = np.zeros((n_states, n_points), dtype=FC.DTYPE)
        D[:] = fdata[FV.D][stsel][:, None]

        i0 = mdata.states_i0(counter=True, algo=algo)
        i1 = i0 + mdata.n_states
        dxy = self._dxy[:i1]

        trace_p = np.zeros((n_states, n_points, 2), dtype=FC.DTYPE)
        trace_p[:] = points[:, :, :2] - rxyz[:, None, :2]
        trace_l = np.zeros((n_states, n_points), dtype=FC.DTYPE)
        trace_d = np.full((n_states, n_points), np.inf, dtype=FC.DTYPE)
        trace_si = np.zeros((n_states, n_points), dtype=FC.ITYPE)
        trace_si[:] = i0 + np.arange(n_states)[:, None] + 1

        wcoos = np.full((n_states, n_points, 3), 1e20, dtype=FC.DTYPE)
        wcoosx = wcoos[:, :, 0]
        wcoosy = wcoos[:, :, 1]
        wcoos[:, :, 2] = points[:, :, 2] - rxyz[:, None, 2]
        del rxyz

        while True:
            sel = (trace_si > 0) & (trace_l < self.max_wake_length)
            if np.any(sel):
                trace_si[sel] -= 1

                delta = dxy[trace_si[sel]]
                dmag = np.linalg.norm(delta, axis=-1)

                trace_p[sel] -= delta
                trace_l[sel] += dmag

                trp = trace_p[sel]
                d0 = trace_d[sel]
                d = np.linalg.norm(trp, axis=-1)
                trace_d[sel] = d

                seln = d <= np.minimum(d0, 2 * dmag)
                if np.any(seln):
                    htrp = trp[seln]
                    raxis = delta[seln]
                    raxis = raxis / np.linalg.norm(raxis, axis=-1)[:, None]
                    saxis = np.concatenate(
                        [-raxis[:, 1, None], raxis[:, 0, None]], axis=1
                    )

                    wcx = wcoosx[sel]
                    wcx[seln] = np.einsum("sd,sd->s", htrp, raxis) + trace_l[sel][seln]
                    wcoosx[sel] = wcx
                    del wcx, raxis

                    wcy = wcoosy[sel]
                    wcy[seln] = np.einsum("sd,sd->s", htrp, saxis)
                    wcoosy[sel] = wcy
                    del wcy, saxis, htrp

            else:
                break

        # turbines that cause wake:
        pdata.add(FC.STATE_SOURCE_TURBINE, states_source_turbine, (FC.STATE,))

        # states that cause wake for each target point:
        pdata.add(FC.STATES_SEL, trace_si, (FC.STATE, FC.POINT))

        return wcoos

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
