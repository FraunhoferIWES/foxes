import numpy as np
import numbers

from foxes.core import WakeSuperposition
import foxes.variables as FV
import foxes.constants as FC


class MaxSuperposition(WakeSuperposition):
    """
    Maximum supersposition of wake model results,
    optionally rescaled.

    Parameters
    ----------
    scalings : dict or number or str
        Scaling rules. If `dict`, key: variable name str,
        value: number or str. If `str`:
        - `source_turbine`: Scale by source turbine value of variable
        - `source_turbine_amb`: Scale by source turbine ambient value of variable
        - `source_turbine_<var>`: Scale by source turbine value of variable <var>

    Attributes
    ----------
    scalings : dict or number or str
        The scaling rules

    """

    def __init__(self, scalings):
        super().__init__()
        self.scalings = scalings

    def initialize(self, algo, verbosity=0):
        """
        Initializes the model.

        Parameters
        ----------
        algo : foxes.core.Algorithm
            The calculation algorithm
        verbosity : int
            The verbosity level

        """
        self.SIGNS = self.var("SIGNS")
        super().initialize(algo, verbosity)

    def calc_wakes_plus_wake(
        self,
        algo,
        mdata,
        fdata,
        states_source_turbine,
        sel_sp,
        variable,
        wake_delta,
        wake_model_result,
    ):
        """
        Add a wake delta to previous wake deltas.

        Parameters
        ----------
        algo : foxes.core.Algorithm
            The calculation algorithm
        mdata : foxes.core.Data
            The model data
        fdata : foxes.core.Data
            The farm data
        states_source_turbine : numpy.ndarray
            For each state, one turbine index for the
            wake causing turbine. Shape: (n_states,)
        sel_sp : numpy.ndarray of bool
            The selection of points, shape: (n_states, n_points)
        variable : str
            The variable name for which the wake deltas applies
        wake_delta : numpy.ndarray
            The original wake deltas, shape: (n_states, n_points)
        wake_model_result : numpy.ndarray
            The new wake deltas of the selected points,
            shape: (n_sel_sp,)

        Returns
        -------
        wdelta : numpy.ndarray
            The updated wake deltas, shape: (n_states, n_points)

        """

        if np.all(np.max(np.abs(wake_model_result)) < 1e-10):
            return wake_delta

        if self.SIGNS not in mdata:
            mdata[self.SIGNS] = {}
        if variable not in mdata[self.SIGNS]:
            mdata[self.SIGNS][variable] = (
                -1 if np.all(wake_model_result <= 0.0) else 1.0
            )

        if isinstance(self.scalings, dict):
            try:
                scaling = self.scalings[variable]
            except KeyError:
                raise KeyError(
                    f"Model '{self.name}': No scaling found for wake variable '{variable}'"
                )
        else:
            scaling = self.scalings

        wake_model_result = np.abs(wake_model_result)
        odelta = wake_delta[sel_sp]

        if scaling is None:

            wake_delta[sel_sp] = np.maximum(odelta, wake_model_result)
            return wake_delta

        elif isinstance(scaling, numbers.Number):
            wake_delta[sel_sp] = np.maximum(odelta, scaling * wake_model_result)
            return wake_delta

        elif (
            isinstance(scaling, str)
            and len(scaling) >= 14
            and (
                scaling == f"source_turbine"
                or scaling == "source_turbine_amb"
                or (len(scaling) > 15 and scaling[14] == "_")
            )
        ):

            if scaling == f"source_turbine":
                var = variable
            elif scaling == "source_turbine_amb":
                var = FV.var2amb[variable]
            else:
                var = scaling[15:]

            try:
                vdata = fdata[var]

            except KeyError:
                raise KeyError(
                    f"Model '{self.name}': Scaling variable '{var}' for wake variable '{variable}' not found in fdata {sorted(list(fdata.keys()))}"
                )

            n_states = mdata.n_states
            n_points = wake_delta.shape[1]
            stsel = (np.arange(n_states), states_source_turbine)
            scale = np.zeros((n_states, n_points), dtype=FC.DTYPE)
            scale[:] = vdata[stsel][:, None]
            scale = scale[sel_sp]

            wake_delta[sel_sp] = np.maximum(odelta, scale * wake_model_result)

            return wake_delta

        else:
            raise ValueError(
                f"Model '{self.name}': Invalid scaling choice '{scaling}' for wake variable '{variable}', valid choices: None, <scalar>, 'source_turbine', 'source_turbine_amb', 'source_turbine_<var>'"
            )

    def calc_final_wake_delta(
        self, algo, mdata, fdata, variable, amb_results, wake_delta
    ):
        """
        Calculate the final wake delta after adding all
        contributions.

        Parameters
        ----------
        algo : foxes.core.Algorithm
            The calculation algorithm
        mdata : foxes.core.Data
            The model data
        fdata : foxes.core.Data
            The farm data
        variable : str
            The variable name for which the wake deltas applies
        amb_results : numpy.ndarray
            The ambient results, shape: (n_states, n_points)
        wake_delta : numpy.ndarray
            The wake deltas, shape: (n_states, n_points)

        Returns
        -------
        final_wake_delta : numpy.ndarray
            The final wake delta, which will be added to the ambient
            results by simple plus operation. Shape: (n_states, n_points)

        """
        try:
            return mdata[self.SIGNS][variable] * wake_delta
        except KeyError as e:
            if np.max(np.abs(wake_delta)) < 1e-10:
                return wake_delta
            raise e
