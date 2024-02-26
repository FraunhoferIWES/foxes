import numpy as np

from foxes.core import FarmDataModel, Data
import foxes.variables as FV
import foxes.constants as FC


class InitFarmData(FarmDataModel):
    """
    Sets basic turbine data and applies downwind order

    :group: algorithms.downwind.models

    """

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
        return [FV.X, FV.Y, FV.H, FV.D, FV.WD, FV.YAW, FV.ORDER]

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
        fdata[FV.X] = fdata[FV.TXYH][..., 0]
        fdata[FV.Y] = fdata[FV.TXYH][..., 1]
        fdata[FV.H] = fdata[FV.TXYH][..., 2]

        # set X, Y, H, D:
        fdata[FV.TXYH] = np.full((n_states, n_turbines, 3), np.nan, dtype=FC.DTYPE)
        for ti, t in enumerate(algo.farm.turbines):
            fdata[FV.TXYH][:, ti, :2] = t.xy[None, :]

            H = t.H 
            if H is None:
                H = algo.farm_controller.turbine_types[ti].H
            fdata[FV.TXYH][:, ti, 2] = H 

            D = t.D 
            if D is None:
                D = algo.farm_controller.turbine_types[ti].D
            fdata[FV.D][:, ti] = D 

        # calc WD and YAW at rotor centres:
        pdata = Data.from_points(points=fdata[FV.TXYH])
        sres = algo.states.calculate(algo, mdata, fdata, pdata)
        fdata[FV.WD] = sres[FV.WD]
        del pdata, sres
        
        # calculate and apply downwind order:
        order = algo.wake_frame.calc_order(algo, mdata, fdata)
        fdata[FV.ORDER] = order
        ssel = np.zeros_like(fdata[FV.ORDER])
        ssel[:] = np.arange(n_states)[:, None]
        fdata[FV.WD] = fdata[FV.WD][ssel, order]
        fdata[FV.YAW] = fdata[FV.WD].copy()
        fdata[FV.D] = fdata[FV.D][ssel, order]
        fdata[FV.TXYH] = fdata[FV.TXYH][ssel, order]
        fdata[FV.X] = fdata[FV.TXYH][..., 0]
        fdata[FV.Y] = fdata[FV.TXYH][..., 1]
        fdata[FV.H] = fdata[FV.TXYH][..., 2]
        mdata[FC.TMODEL_SELS] = mdata[FC.TMODEL_SELS][ssel, order]

        return {v: fdata[v] for v in self.output_farm_vars(algo)}
