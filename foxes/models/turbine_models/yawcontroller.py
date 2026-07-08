from __future__ import annotations
# mypy: disable-error-code=override

import numpy as np
from foxes.config import config
import foxes.variables as FV
from foxes.core import TurbineModel
from foxes.utils import wd2uv, uv2wd, delta_wd
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from foxes.core.algorithm import Algorithm
    from foxes.core.data import FData, MData
    from foxes.core.model import LoadedData


class YawController(TurbineModel):
    """
    A dummy yaw controller that sets yaw based on wind direction.
    For demonstration: yaws turbine 20 degrees away from wind direction
    when wind is from north (WD around 0 degrees).
    """

    def __init__(
        self,
        max_yaw_rate: float = 0.3,
        max_yawm: float = 7.5,
        avg_time: float = 60,
    ) -> None:
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
        self._targetyaw: np.ndarray | None = None
        self._windowstart: np.ndarray | None = None

    def output_farm_vars(self, algo: Algorithm) -> list[str]:
        """The variables modified by this model."""
        return [FV.YAW, FV.YAWM]

    def initialize(
        self,
        algo: Algorithm,
        loaded_data: LoadedData | None = None,
        force: bool = False,
        verbosity: int = 0,
    ) -> LoadedData:
        """
        Initialize the controller before iterations start.

        Parameters
        ----------
        algo : foxes.algorithms.sequential.Sequential
            The sequential algorithm instance
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
        loaded_data = super().initialize(
            algo, loaded_data=loaded_data, force=force, verbosity=verbosity
        )

        n_turbines = algo.n_turbines
        assert n_turbines is not None, "Missing n_turbines in algorithm"

        delta_t = algo.states.index()[1] - algo.states.index()[0]
        self._dt = delta_t.astype("timedelta64[s]").astype(
            float
        )  # number of time steps to consider
        self._n = int(self._avg_time / self._dt)  # number of time steps to consider
        self._targetyaw = np.full((n_turbines), np.nan, dtype=config.dtype_double)
        self._windowstart = np.zeros((n_turbines), dtype=config.dtype_int)
        # self.__once_done = set()
        return loaded_data

    def calculate(
        self,
        algo: Algorithm,
        mdata: MData,
        fdata: FData,
        st_sel: slice | np.ndarray = slice(None),
    ) -> dict[str, np.ndarray]:
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
        n_turbines = fdata.n_turbines
        assert n_turbines is not None
        t_sel = np.zeros((fdata.n_states, n_turbines), dtype=np.bool_)
        t_sel[st_sel] = True
        t_sel = t_sel[0, :]

        # get current data:
        counter = algo.states.counter
        fresults = algo.farm_results_downwind
        assert fresults is not None, "Missing farm_results_downwind"
        wd = fdata[FV.AMB_WD][0, :]
        ws = fdata[FV.AMB_REWS][0, :]
        yaw = fdata[FV.YAW][0, :]
        yawm = fdata[FV.YAWM][0, :]

        # special case of first time step:
        if counter == 0:
            yawm[:] = 0.0
            return {FV.YAW: fdata[FV.YAW], FV.YAWM: fdata[FV.YAWM]}

        # Respect waiting time for window average:
        lastyaw = fresults[FV.YAW].to_numpy()[counter - 1]
        wstart = self._windowstart
        targetyaw = self._targetyaw
        assert wstart is not None and targetyaw is not None
        sel = t_sel & (counter < wstart + self._n - 1)
        if np.any(sel):
            yaw[sel] = lastyaw[sel]

        # compute setpoint from last n time steps:
        sel = t_sel & (counter == wstart + self._n - 1)
        if np.any(sel):
            s = np.s_[counter - self._n + 1 : counter + 1]
            wd_hist = fresults[FV.AMB_WD].to_numpy()
            wd_hist = wd_hist[s, sel]
            wd_hist[-1] = wd[sel]
            ws_hist = fresults[FV.AMB_REWS].to_numpy()
            ws_hist = ws_hist[s, sel]
            ws_hist[-1] = ws[sel]
            uv_hist = wd2uv(wd_hist, ws_hist)
            targets = uv2wd(np.mean(uv_hist, axis=0))
            del wd_hist, ws_hist, uv_hist, s

            # set new setpoint only if exceeding max yaw misalignment:
            sel2 = np.abs(delta_wd(lastyaw[sel], targets)) >= self._max_yawm
            if np.any(sel2):
                targetyaw[sel] = np.where(sel2, targets, targetyaw[sel])
            if np.any(~sel2):
                yaw[sel] = np.where(~sel2, lastyaw[sel], yaw[sel])
                wsel = wstart[sel]
                wstart[sel] = np.where(~sel2, wsel + 1, wsel)

        # run controller logic:
        sel = (
            t_sel
            & (counter >= wstart + self._n - 1)
            & ~np.isnan(targetyaw)
        )
        if np.any(sel):
            # prepare:
            yaw0 = lastyaw[sel]
            wd_target = targetyaw[sel]
            delyaw = delta_wd(yaw0, wd_target)  # misalignment towards target yaw
            maxyaw = self._max_yaw_rate * self._dt  # max yaw maneuver during time step

            # set new yaw:
            reached = maxyaw >= np.abs(delyaw)
            yaw[sel] = np.where(reached, wd_target, yaw0 + maxyaw * np.sign(delyaw))

            # reset window if target yaw is reached:
            if np.any(reached):
                wstart[sel] = np.where(reached, counter + 1, wstart[sel])

        yawm[t_sel] = delta_wd(wd[t_sel], yaw[t_sel])

        return {FV.YAW: fdata[FV.YAW], FV.YAWM: fdata[FV.YAWM]}
