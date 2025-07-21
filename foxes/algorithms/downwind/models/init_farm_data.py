import numpy as np

from foxes.core import FarmDataModel, TData
import foxes.variables as FV
import foxes.constants as FC
from foxes.config import config


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
            FV.ORDER_SSEL,
            FV.ORDER_INV,
        ]

    def calculate(self, algo, mdata, fdata):
        """
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

        # add and set X, Y, H, D:
        fdata.add(
            FV.TXYH,
            np.zeros((n_states, n_turbines, 3), dtype=config.dtype_double),
            (FC.STATE, FC.TURBINE, FC.XYH),
        )
        fdata.add(
            FV.D,
            np.zeros((n_states, n_turbines), dtype=config.dtype_double),
            (FC.STATE, FC.TURBINE),
        )
        for ti, t in enumerate(algo.farm.turbines):
            if len(t.xy.shape) == 1:
                fdata[FV.TXYH][:, ti, :2] = t.xy[None, :]
            else:
                i0 = fdata.states_i0(counter=True)
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

        # calc WD at rotor centres:
        svrs = algo.states.output_point_vars(algo)
        tdata = TData.from_points(points=fdata[FV.TXYH], variables=svrs)
        sres = algo.states.calculate(algo, mdata, fdata, tdata)
        fdata.add(
            FV.WD,
            sres[FV.WD][:, :, 0],
            (FC.STATE, FC.TURBINE),
        )
        fdata.add(
            FV.AMB_WD,
            fdata[FV.WD].copy(),
            (FC.STATE, FC.TURBINE),
        )
        del tdata, sres, svrs

        # calculate downwind order:
        order = algo.wake_frame.calc_order(algo, mdata, fdata)
        ssel = np.zeros_like(order)
        ssel[:] = np.arange(n_states)[:, None]

        # apply downwind order to all data:
        for data in [fdata, mdata]:
            for k in data.keys():
                if (
                    k not in [FV.X, FV.Y, FV.H]
                    and tuple(data.dims[k][:2]) == (FC.STATE, FC.TURBINE)
                    and np.any(data[k] != data[k][0, 0, None, None])
                ):
                    data[k][:] = data[k][ssel, order]

        # add derived data:
        for i, v in enumerate([FV.X, FV.Y, FV.H]):
            fdata.add(
                v,
                fdata[FV.TXYH][:, :, i],
                (FC.STATE, FC.TURBINE),
            )
        fdata.add(
            FV.YAW,
            fdata[FV.WD].copy(),
            (FC.STATE, FC.TURBINE),
        )
        fdata.add(
            FV.ORDER,
            order,
            (FC.STATE, FC.TURBINE),
        )
        fdata.add(
            FV.ORDER_SSEL,
            ssel,
            (FC.STATE, FC.TURBINE),
        )
        fdata.add(
            FV.ORDER_INV,
            np.zeros_like(order),
            (FC.STATE, FC.TURBINE),
        )
        fdata[FV.ORDER_INV][ssel, order] = np.arange(n_turbines)[None, :]

        return {v: fdata[v] for v in self.output_farm_vars(algo)}
