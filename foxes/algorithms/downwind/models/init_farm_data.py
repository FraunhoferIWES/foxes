import numpy as np

from foxes.core import FarmDataModel, TData
import foxes.variables as FV
import foxes.constants as FC


class InitFarmData(FarmDataModel):
    """
    Sets basic turbine data and applies downwind order

    :group: algorithms.downwind.models

    """

    def __init__(self):
        """
        Constructor.
        """
        super().__init__(pre_rotor=True)

    def output_farm_vars(self, algo):
        """
        The variables which are being modified by the model.

        Parameters
        ----------
        algo: foxes.core.Algorithm
            The calculation algorithm

        Returns
        -------
        output_vars: list of str
            The output variable names

        """
        return [
            FV.X,
            FV.Y,
            FV.H,
            FV.D,
            FV.WD,
            FV.YAW,
            FV.ORDER,
            FV.WEIGHT,
            FV.ORDER_SSEL,
            FV.ORDER_INV,
        ]

    def calculate(self, algo, mdata, fdata):
        """ "
        The main model calculation.

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
        results: dict
            The resulting data, keys: output variable str.
            Values: numpy.ndarray with shape (n_states, n_turbines)

        """
        # prepare:
        n_states = fdata.n_states
        n_turbines = algo.n_turbines

        # define FV.TXYH as vector [X, Y, H]:
        fdata[FV.TXYH] = np.full((n_states, n_turbines, 3), np.nan, dtype=FC.DTYPE)
        fdata.dims[FV.TXYH] = (FC.STATE, FC.TURBINE, FC.XYH)
        for i, v in enumerate([FV.X, FV.Y, FV.H]):
            fdata[v] = fdata[FV.TXYH][..., i]
            fdata.dims[v] = (FC.STATE, FC.TURBINE)

        # set X, Y, H, D:
        fdata[FV.D] = np.zeros((n_states, n_turbines), dtype=FC.DTYPE)
        for ti, t in enumerate(algo.farm.turbines):

            if len(t.xy.shape) == 1:
                fdata[FV.TXYH][:, ti, :2] = t.xy[None, :]
            else:
                i0 = fdata.states_i0()
                s = np.s_[i0 : i0 + fdata.n_states]
                fdata[FV.TXYH][:, ti, :2] = t.xy[s]

            H = t.H
            if H is None:
                H = algo.farm_controller.turbine_types[ti].H
            fdata[FV.TXYH][:, ti, 2] = H

            D = t.D
            if D is None:
                D = algo.farm_controller.turbine_types[ti].D
            fdata[FV.D][:, ti] = D

        # calc WD and YAW at rotor centres:
        tdata = TData.from_points(points=fdata[FV.TXYH])
        sres = algo.states.calculate(algo, mdata, fdata, tdata)
        fdata[FV.WD] = sres[FV.WD][:, :, 0]
        del tdata, sres

        # calculate and inverse:
        order = algo.wake_frame.calc_order(algo, mdata, fdata)
        ssel = np.zeros_like(order)
        ssel[:] = np.arange(n_states)[:, None]
        fdata[FV.ORDER] = order
        fdata[FV.ORDER_SSEL] = ssel
        fdata[FV.ORDER_INV] = np.zeros_like(order)
        fdata[FV.ORDER_INV][ssel, order] = np.arange(n_turbines)[None, :]

        # apply downwind order to all data:
        fdata[FV.TXYH] = fdata[FV.TXYH][ssel, order]
        for i, v in enumerate([FV.X, FV.Y, FV.H]):
            fdata[v] = fdata[FV.TXYH][..., i]
        for v in [FV.D, FV.WD, FV.WEIGHT]:
            if np.any(fdata[v] != fdata[v][0, 0, None, None]):
                fdata[v] = fdata[v][ssel, order]
        fdata[FV.YAW] = fdata[FV.WD].copy()
        for k in mdata.keys():
            if tuple(mdata.dims[k][:2]) == (FC.STATE, FC.TURBINE) and np.any(
                mdata[k] != mdata[k][0, 0, None, None]
            ):
                mdata[k] = mdata[k][ssel, order]

        return {v: fdata[v] for v in self.output_farm_vars(algo)}
