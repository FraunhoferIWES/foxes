import numpy as np
import numbers

from foxes.core import WakeSuperposition
import foxes.variables as FV
import foxes.constants as FC


class QuadraticSuperposition(WakeSuperposition):
    """
    Quadratic supersposition of wake model results,
    optionally rescaled.

    Attributes
    ----------
    scalings: dict or number or str
        The scaling rules
    svars: list of str
        The scaling variables

    :group: models.wake_superpositions

    """

    def __init__(self, scalings, svars=None):
        """
        Constructor.

        Parameters
        ----------
        scalings: dict or number or str
            Scaling rules. If `dict`, key: variable name str,
            value: number or str. If `str`:
            - `source_turbine`: Scale by source turbine value of variable
            - `source_turbine_amb`: Scale by source turbine ambient value of variable
            - `source_turbine_<var>`: Scale by source turbine value of variable <var>
        svars: list of str, optional
            The scaling variables

        """
        super().__init__()
        self.scalings = scalings
        self.svars = svars

    def input_farm_vars(self, algo):
        """
        The variables which are needed for running
        the model.

        Parameters
        ----------
        algo: foxes.core.Algorithm
            The calculation algorithm

        Returns
        -------
        input_vars: list of str
            The input variable names

        """
        if self.svars is not None:
            return self.svars
        elif isinstance(self.scalings, dict):
            return list(self.scalings.keys())
        elif (
            isinstance(self.scalings, str)
            and len(self.scalings) > 15
            and self.scalings[:15] == "source_turbine_"
        ):
            return [self.scalings[15:]]
        else:
            raise ValueError(
                f"{self.name}: Unable to determine scaling variable for scaling = '{self.scalings}'"
            )

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
        self.SIGNS = self.var("SIGNS")
        return super().initialize(algo, verbosity)

    def calc_wakes_plus_wake(
        self,
        algo,
        mdata,
        fdata,
        pdata,
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
        sel_sp: numpy.ndarray of bool
            The selection of points, shape: (n_states, n_points)
        variable: str
            The variable name for which the wake deltas applies
        wake_delta: numpy.ndarray
            The original wake deltas, shape: (n_states, n_points)
        wake_model_result: numpy.ndarray
            The new wake deltas of the selected points,
            shape: (n_sel_sp,)

        Returns
        -------
        wdelta: numpy.ndarray
            The updated wake deltas, shape: (n_states, n_points)

        """

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

        if scaling is None:
            wake_delta[sel_sp] += wake_model_result**2
            return wake_delta

        elif isinstance(scaling, numbers.Number):
            wake_delta[sel_sp] += (scaling * wake_model_result) ** 2
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

            scale = self.get_data(
                var,
                FC.STATE_POINT,
                lookup="w",
                fdata=fdata,
                pdata=pdata,
                algo=algo,
                states_source_turbine=states_source_turbine,
            )[sel_sp]

            wake_delta[sel_sp] += (scale * wake_model_result) ** 2

            return wake_delta

        else:
            raise ValueError(
                f"Model '{self.name}': Invalid scaling choice '{scaling}' for wake variable '{variable}', valid choices: None, <scalar>, 'source_turbine', 'source_turbine_amb', 'source_turbine_<var>'"
            )

    def calc_final_wake_delta(
        self,
        algo,
        mdata,
        fdata,
        pdata,
        variable,
        amb_results,
        wake_delta,
    ):
        """
        Calculate the final wake delta after adding all
        contributions.

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
        variable: str
            The variable name for which the wake deltas applies
        amb_results: numpy.ndarray
            The ambient results, shape: (n_states, n_points)
        wake_delta: numpy.ndarray
            The wake deltas, shape: (n_states, n_points)

        Returns
        -------
        final_wake_delta: numpy.ndarray
            The final wake delta, which will be added to the ambient
            results by simple plus operation. Shape: (n_states, n_points)

        """
        try:
            return mdata[self.SIGNS][variable] * np.sqrt(wake_delta)
        except KeyError as e:
            if np.max(np.abs(wake_delta)) < 1e-10:
                return wake_delta
            raise e
