import numpy as np
from foxes import config
import foxes.variables as FV
from foxes.core import TurbineModel
from foxes.utils import wd2uv, uv2wd, delta_wd


class YawController(TurbineModel):
    """
    A dummy yaw controller that sets yaw based on wind direction.
    For demonstration: yaws turbine 20 degrees away from wind direction
    when wind is from north (WD around 0 degrees).
    """

    def __init__(self, max_yaw_rate=0.3, max_yawm=7.5, avg_time=60):
        """
        Constructor.

        Parameters
        ----------
        max_yaw_rate : float, optional
            Maximum yaw rate change in degrees per second (default: 0.3).
        max_yawm : float, optional
            Maximum yaw misalignment of turbine relative the running mean wind direction in degrees (default: 7.5).
        avg_time : float, optional
            Averaging time window in seconds for running mean wind direction (default: 60).
        """
        super().__init__()
        self._max_yaw_rate = max_yaw_rate
        self._max_yawm = max_yawm
        self._avg_time = avg_time

    def output_farm_vars(self, algo):
        """The variables modified by this model."""
        return [FV.YAW, FV.YAWM]

    def initialize(self, algo, verbosity=0, force=False):
        """
        Initialize the controller before iterations start.

        Parameters
        ----------
        algo : foxes.algorithms.sequential.Sequential
            The sequential algorithm instance
        """
        super().initialize(algo, verbosity=verbosity, force=force)

        n_turbines = algo.n_turbines

        delta_t = algo.states.index()[1] - algo.states.index()[0]
        self._dt = delta_t.astype("timedelta64[s]").astype(
            float
        )  # number of time steps to consider
        self._n = int(self._avg_time / self._dt)  # number of time steps to consider
        self._targetyaw = np.full((n_turbines), np.nan, dtype=config.dtype_double)
        self._windowstart = np.zeros((n_turbines), dtype=config.dtype_int)
        # self.__once_done = set()

    def calculate(self, algo, mdata, fdata, st_sel):
        """
        The main model calculation.

        Parameters
        ----------
        algo: foxes.core.Algorithm
            The calculation algorithm
        mdata: foxes.core.MData
            The model data
        fdata: foxes.core.FData
            The farm data
        st_sel: slice or numpy.ndarray of bool
            The state-turbine selection,
            for shape: (n_turbines)

        Returns
        -------
        results: dict
            The resulting data, keys: output variable str.
            Values: numpy.ndarray with shape (n_turbines)
        """
        assert fdata.n_states == 1, (
            "This controller only runs with the Sequential algorithm."
        )

        """
        if algo.counter in self.__once_done:
            return {FV.YAW: fdata[FV.YAW], FV.YAWM: fdata[FV.YAWM]}
        else:
            self.__once_done.add(algo.counter)
        """

        # prepare:
        # self.ensure_output_vars(algo, fdata)
        t_sel = np.zeros((fdata.n_states, fdata.n_turbines), dtype=bool)
        t_sel[st_sel] = True
        t_sel = t_sel[0, :]

        # get current data:
        wd = fdata[FV.AMB_WD][0, :]
        ws = fdata[FV.AMB_REWS][0, :]
        yaw = fdata[FV.YAW][0, :]
        yawm = fdata[FV.YAWM][0, :]

        # special case of first time step:
        if algo.counter == 0:
            yawm[:] = 0.0
            return {FV.YAW: fdata[FV.YAW], FV.YAWM: fdata[FV.YAWM]}

        # Respect waiting time for window average:
        lastyaw = algo.farm_results_downwind[FV.YAW].to_numpy()[algo.counter - 1]
        sel = t_sel & (algo.counter < self._windowstart + self._n - 1)
        if np.any(sel):
            yaw[sel] = lastyaw[sel]

        # compute setpoint from last n time steps:
        sel = t_sel & (algo.counter == self._windowstart + self._n - 1)
        if np.any(sel):
            s = np.s_[algo.counter - self._n + 1 : algo.counter + 1]
            wd_hist = algo.farm_results_downwind[FV.AMB_WD].to_numpy()
            wd_hist = wd_hist[s, sel]
            wd_hist[-1] = wd[sel]
            ws_hist = algo.farm_results_downwind[FV.AMB_REWS].to_numpy()
            ws_hist = ws_hist[s, sel]
            ws_hist[-1] = ws[sel]
            uv_hist = wd2uv(wd_hist, ws_hist)
            targets = uv2wd(np.mean(uv_hist, axis=0))
            del wd_hist, ws_hist, uv_hist, s

            # set new setpoint only if exceeding max yaw misalignment:
            sel2 = np.abs(delta_wd(lastyaw[sel], targets)) >= self._max_yawm
            if np.any(sel2):
                self._targetyaw[sel] = np.where(sel2, targets, self._targetyaw[sel])
            if np.any(~sel2):
                yaw[sel] = np.where(~sel2, lastyaw[sel], yaw[sel])
                wstart = self._windowstart[sel]
                self._windowstart[sel] = np.where(~sel2, wstart + 1, wstart)

        # run controller logic:
        sel = (
            t_sel
            & (algo.counter >= self._windowstart + self._n - 1)
            & ~np.isnan(self._targetyaw)
        )
        if np.any(sel):
            # prepare:
            yaw0 = lastyaw[sel]
            wd_target = self._targetyaw[sel]
            delyaw = delta_wd(yaw0, wd_target)  # misalignment towards target yaw
            maxyaw = self._max_yaw_rate * self._dt  # max yaw maneuver during time step

            # set new yaw:
            reached = maxyaw >= np.abs(delyaw)
            yaw[sel] = np.where(reached, wd_target, yaw0 + maxyaw * np.sign(delyaw))

            # reset window if target yaw is reached:
            if np.any(reached):
                self._windowstart[sel] = np.where(
                    reached, algo.counter + 1, self._windowstart[sel]
                )

        yawm[t_sel] = delta_wd(wd[t_sel], yaw[t_sel])

        return {FV.YAW: fdata[FV.YAW], FV.YAWM: fdata[FV.YAWM]}
