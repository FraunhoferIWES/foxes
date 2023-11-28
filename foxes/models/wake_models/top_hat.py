import numpy as np
from abc import abstractmethod

from foxes.models.wake_models.axisymmetric import AxisymmetricWakeModel
import foxes.variables as FV
import foxes.constants as FC


class TopHatWakeModel(AxisymmetricWakeModel):
    """
    Abstract base class for top-hat wake models.

    :group: models.wake_models

    """

    def __init__(self, superpositions, ct_max=0.9999):
        """
        Constructor.

        Parameters
        ----------
        superpositions: dict
            The superpositions. Key: variable name str,
            value: The wake superposition model name,
            will be looked up in model book
        ct_max: float
            The maximal value for ct, values beyond will be limited
            to this number

        """
        super().__init__(superpositions)
        self.ct_max = ct_max

    @abstractmethod
    def calc_wake_radius(
        self,
        algo,
        mdata,
        fdata,
        pdata,
        states_source_turbine,
        x,
        ct,
    ):
        """
        Calculate the wake radius, depending on x only (not r).

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
        x: numpy.ndarray
            The x values, shape: (n_states, n_points)
        r: numpy.ndarray
            The radial values for each x value, shape:
            (n_states, n_points, n_r_per_x, 2)
        ct: numpy.ndarray
            The ct values of the wake-causing turbines,
            shape: (n_states, n_points)

        Returns
        -------
        wake_r: numpy.ndarray
            The wake radii, shape: (n_states, n_points)

        """
        pass

    @abstractmethod
    def calc_centreline_wake_deltas(
        self,
        algo,
        mdata,
        fdata,
        pdata,
        states_source_turbine,
        sp_sel,
        x,
        wake_r,
        ct,
    ):
        """
        Calculate centre line results of wake deltas.

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
        sp_sel: numpy.ndarray of bool
            The state-point selection, for which the wake
            is non-zero, shape: (n_states, n_points)
        x: numpy.ndarray
            The x values, shape: (n_sp_sel,)
        wake_r: numpy.ndarray
            The wake radii, shape: (n_sp_sel,)
        ct: numpy.ndarray
            The ct values of the wake-causing turbines,
            shape: (n_sp_sel,)

        Returns
        -------
        cl_del: dict
            The centre line wake deltas. Key: variable name str,
            varlue: numpy.ndarray, shape: (n_sp_sel,)

        """
        pass

    def calc_wakes_spsel_x_r(
        self,
        algo,
        mdata,
        fdata,
        pdata,
        states_source_turbine,
        x,
        r,
    ):
        """
        Calculate wake deltas.

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
        x: numpy.ndarray
            The x values, shape: (n_states, n_points)
        r: numpy.ndarray
            The radial values for each x value, shape:
            (n_states, n_points, n_r_per_x, 2)

        Returns
        -------
        wdeltas: dict
            The wake deltas. Key: variable name str,
            value: numpy.ndarray, shape: (n_sp_sel, n_r_per_x)
        sp_sel: numpy.ndarray of bool
            The state-point selection, for which the wake
            is non-zero, shape: (n_states, n_points)

        """
        ct = self.get_data(
            FV.CT,
            FC.STATE_POINT,
            lookup="w",
            fdata=fdata,
            pdata=pdata,
            states_source_turbine=states_source_turbine,
            algo=algo,
        )
        ct[ct > self.ct_max] = self.ct_max

        wake_r = self.calc_wake_radius(
            algo, mdata, fdata, pdata, states_source_turbine, x, ct
        )

        wdeltas = {}
        sp_sel = (ct > 0.0) & (x > 1e-5) & np.any(r < wake_r[:, :, None], axis=2)
        if np.any(sp_sel):
            x = x[sp_sel]
            r = r[sp_sel]
            ct = ct[sp_sel]
            wake_r = wake_r[sp_sel]

            cl_del = self.calc_centreline_wake_deltas(
                algo, mdata, fdata, pdata, states_source_turbine, sp_sel, x, wake_r, ct
            )

            nsel = r >= wake_r[:, None]
            for v, wdel in cl_del.items():
                wdeltas[v] = np.zeros_like(r)
                wdeltas[v][:] = wdel[:, None]
                wdeltas[v][nsel] = 0.0

        return wdeltas, sp_sel
