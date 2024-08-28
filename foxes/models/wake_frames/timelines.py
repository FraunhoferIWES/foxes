import numpy as np
from xarray import Dataset

from foxes.core import WakeFrame
from foxes.utils import wd2uv
from foxes.core.data import MData, FData, TData
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
            
        # find turbine hub heights:
        t2h = np.zeros(algo.n_turbines, dtype=FC.DTYPE)
        for ti, t in enumerate(algo.farm.turbines):
            t2h[ti] = t.H if t.H is not None else algo.farm_controller.turbine_types[ti].H
        heights = np.unique(t2h)

        # calculate horizontal wind vector in all states:
        self._uv = np.zeros((algo.n_states, 1, 3), dtype=FC.DTYPE)

        # prepare mdata:
        data = algo.get_model_data(algo.states)["coords"]
        mdict = {v: np.array(d) for v, d in data.items()}
        mdims = {v: (v,) for v in data.keys()}
        data = algo.get_model_data(algo.states)["data_vars"]
        mdict.update({v: d[1] for v, d in data.items()})
        mdims.update({v: d[0] for v, d in data.items()})
        mdata = MData(mdict, mdims, loop_dims=[FC.STATE])
        del mdict, mdims, data

        # prepare fdata:
        fdata = FData({}, {}, loop_dims=[FC.STATE])

        # prepare tdata:
        tdata = {
            v: np.zeros((algo.n_states, 1, 1), dtype=FC.DTYPE)
            for v in algo.states.output_point_vars(algo)
        }
        pdims = {v: (FC.STATE, FC.TARGET, FC.TPOINT) for v in tdata.keys()}
        points = np.zeros((algo.n_states, 1, 3), dtype=FC.DTYPE)
        
        # calculate all heights:
        self._dxy = []
        for h in heights:

            if verbosity > 0:
                print(f"  Height: {h} m")
                
            points[..., 2] = h
            tdata = TData.from_points(
                points=points,
                data=tdata,
                dims=pdims,
            )

            res = algo.states.calculate(algo, mdata, fdata, tdata)
            if len(dt) == 1:
                dxy = wd2uv(res[FV.WD], res[FV.WS])[:, 0, 0, :2] * dt[:, None]
            else:
                dxy = wd2uv(res[FV.WD], res[FV.WS])[:-1, 0, 0, :2] * dt[:, None]
                dxy = np.insert(dxy, 0, dxy[0], axis=0)
            self._dxy.append(dxy)
            """ DEBUG
            import matplotlib.pyplot as plt
            xy = np.array([np.sum(self._dxy[h][:n], axis=0) for n in range(len(self._dxy[h]))])
            print(xy)
            plt.plot(xy[:, 0], xy[:, 1])
            plt.title(f"Height {h} m")
            plt.show()
            quit()
            """
        
        self._dxy = Dataset(
            coords={
                FC.STATE: algo.states.index(),
                "height": heights,
            },
            data_vars={
                "dxy": (("height", FC.STATE, "dir"), np.stack(self._dxy, axis=0))
            }
        )

    def set_running(self, algo, large_model_data, sel=None, isel=None, verbosity=0):
        """
        Sets this model status to running, and moves
        all large data to given storage

        Parameters
        ----------
        algo: foxes.core.Algorithm
            The calculation algorithm
        large_model_data: dict
            Large data storage, this function adds data here.
            Key: model name. Value: dict, large model data
        sel: dict, optional
            The subset selection dictionary
        isel: dict, optional
            The index subset selection dictionary
        verbosity: int
            The verbosity level, 0 = silent
            
        """
        super().set_running(algo, large_model_data, sel, isel, verbosity)
        
        if sel is not None or isel is not None:
            large_model_data[self.name]["dxy"] = self._dxy

            if isel is not None:
                self._dxy = self._dxy.isel(isel)
            if sel is not None:
                self._dxy = self._dxy.sel(sel)

    def unset_running(self, algo, large_model_data, sel=None, isel=None, verbosity=0):
        """
        Sets this model status to not running, recovering large data
        
        Parameters
        ----------
        algo: foxes.core.Algorithm
            The calculation algorithm
        large_model_data: dict
            Large data storage, this function pops data from here.
            Key: model name. Value: dict, large model data
        sel: dict, optional
            The subset selection dictionary
        isel: dict, optional
            The index subset selection dictionary
        verbosity: int
            The verbosity level, 0 = silent

        """
        super().unset_running(algo, large_model_data, sel, isel, verbosity)
        
        data = large_model_data[self.name]
        if "dxy" in data:
            self._dxy = data.pop("dxy")
        
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
        targets = tdata[FC.TARGETS]
        n_states, n_targets, n_tpoints = targets.shape[:3]
        n_points = n_targets * n_tpoints
        points = targets.reshape(n_states, n_points, 3)
        rxyz = fdata[FV.TXYH][:, downwind_index]
        theights = fdata[FV.H][:, downwind_index]
        heights = self._dxy["height"].to_numpy()
        data_dxy = self._dxy["dxy"].to_numpy()

        D = np.zeros((n_states, n_points), dtype=FC.DTYPE)
        D[:] = fdata[FV.D][:, downwind_index, None]

        wcoos = np.full((n_states, n_points, 3), 1e20, dtype=FC.DTYPE)
        wcoosx = wcoos[:, :, 0]
        wcoosy = wcoos[:, :, 1]
        wcoos[:, :, 2] = points[:, :, 2] - rxyz[:, None, 2]
        
        i0 = mdata.states_i0(counter=True)
        i1 = i0 + mdata.n_states
        trace_si = np.zeros((n_states, n_points), dtype=FC.ITYPE)
        trace_si[:] = i0 + np.arange(n_states)[:, None] + 1     
        for hi, h in enumerate(heights):
            
            trace_p = np.zeros((n_states, n_points, 2), dtype=FC.DTYPE)
            trace_p[:] = points[:, :, :2] - rxyz[:, None, :2]
            trace_l = np.zeros((n_states, n_points), dtype=FC.DTYPE)
            trace_d = np.full((n_states, n_points), np.inf, dtype=FC.DTYPE)
            h_trace_si = trace_si.copy()
        
            # step backwards in time, until wake source turbine is hit:
            dxy = data_dxy[hi][:i1]
            precond = (theights[:, None] == h) & (wcoosx > 0.99e20)
            while True:
                sel = (
                    precond & 
                    (h_trace_si > 0) & 
                    (trace_l < self.max_wake_length)
                )
                if np.any(sel):
                    h_trace_si[sel] -= 1

                    delta = dxy[h_trace_si[sel]]
                    dmag = np.linalg.norm(delta, axis=-1)

                    trace_p[sel] -= delta
                    trace_l[sel] += dmag

                    trp = trace_p[sel]
                    d0 = trace_d[sel]
                    d = np.linalg.norm(trp, axis=-1)
                    trace_d[sel] = d

                    # check for turbine hit, then set coordinates:
                    seln = d <= np.minimum(d0, 1.5 * dmag)
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
                        
                        trs = trace_si[sel]
                        trs[seln] = h_trace_si[sel][seln]
                        trace_si[sel] = trs
                        del trs

                else:
                    break
            
            del trace_p, trace_l, trace_d, h_trace_si, dxy, precond

        # store turbines that cause wake:
        tdata[FC.STATE_SOURCE_ORDERI] = downwind_index

        # store states that cause wake for each target point,
        # will be used by model.get_data() during wake calculation:
        tdata.add(
            FC.STATES_SEL,
            trace_si.reshape(n_states, n_targets, n_tpoints),
            (FC.STATE, FC.TARGET, FC.TPOINT),
        )

        return wcoos.reshape(n_states, n_targets, n_tpoints, 3)

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
