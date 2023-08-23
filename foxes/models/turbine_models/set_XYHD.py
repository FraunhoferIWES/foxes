import numpy as np

from foxes.core import TurbineModel
import foxes.variables as FV
import foxes.constants as FC


class SetXYHD(TurbineModel):
    """
    Sets basic turbine data, from
    turbine object to farm data.

    Attributes
    ----------
    set_XY: bool
        Flag for (x,y) data
    set_H: bool
        Flag for height data
    set_D: bool
        Flag for rotor diameter data

    :group: models.turbine_models

    """

    def __init__(self, set_XY=True, set_H=True, set_D=True):
        """
        Constructor.

        Parameters
        ----------
        set_XY: bool
            Flag for (x,y) data
        set_H: bool
            Flag for height data
        set_D: bool
            Flag for rotor diameter data

        """
        super().__init__()

        self.set_XY = set_XY
        self.set_H = set_H
        self.set_D = set_D

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
        ovars = []
        if self.set_XY:
            ovars.append(FV.X)
            ovars.append(FV.Y)
        if self.set_H:
            ovars.append(FV.H)
        if self.set_D:
            ovars.append(FV.D)
        return ovars

    def calculate(self, algo, mdata, fdata, st_sel):
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
        st_sel: numpy.ndarray of bool
            The state-turbine selection,
            shape: (n_states, n_turbines)

        Returns
        -------
        results: dict
            The resulting data, keys: output variable str.
            Values: numpy.ndarray with shape (n_states, n_turbines)

        """
        n_states = mdata.n_states
        n_turbines = algo.n_turbines

        if self.set_XY or self.set_H:
            fdata[FV.TXYH] = np.full((n_states, n_turbines, 3), np.nan, dtype=FC.DTYPE)
            if self.set_XY:
                fdata[FV.X] = fdata[FV.TXYH][..., 0]
                fdata[FV.Y] = fdata[FV.TXYH][..., 1]
            if self.set_H:
                fdata[FV.H] = fdata[FV.TXYH][..., 2]

        for ti in range(n_turbines):
            ssel = st_sel[:, ti]
            if np.any(ssel):
                if np.all(ssel):
                    ssel = np.s_[:]

                if self.set_XY:
                    fdata[FV.X][ssel, ti] = algo.farm.turbines[ti].xy[0]
                    fdata[FV.Y][ssel, ti] = algo.farm.turbines[ti].xy[1]

                if self.set_H:
                    H = algo.farm.turbines[ti].H
                    if H is None:
                        H = algo.farm_controller.turbine_types[ti].H
                    fdata[FV.H][ssel, ti] = H

                if self.set_D:
                    D = algo.farm.turbines[ti].D
                    if D is None:
                        D = algo.farm_controller.turbine_types[ti].D
                    fdata[FV.D][ssel, ti] = D

        return {v: fdata[v] for v in self.output_farm_vars(algo)}
