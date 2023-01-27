import numpy as np

from foxes.core import FarmModel


class Turbine2FarmModel(FarmModel):
    """
    Wrapper that promotes turbine models
    into farm models, simply by selecting
    all turbines.

    Parameters
    ----------
    turbine_model : foxes.core.TurbineModel
        The turbine model

    Attributes
    ----------
    turbine_model : foxes.core.TurbineModel
        The turbine model

    """

    def __init__(self, turbine_model):
        super().__init__()
        self.turbine_model = turbine_model

    def __repr__(self):
        return f"{type(self).__name__}({self.turbine_model})"

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
        algo : foxes.core.Algorithm
            The calculation algorithm
        verbosity : int
            The verbosity level, 0 = silent

        Returns
        -------
        idata : dict
            The dict has exactly two entries: `data_vars`,
            a dict with entries `name_str -> (dim_tuple, data_ndarray)`;
            and `coords`, a dict with entries `dim_name_str -> dim_array`

        """
        idata = super().initialize(algo, verbosity)
        algo.update_idata(self.turbine_model, idata=idata, verbosity=verbosity)

        return idata

    def output_farm_vars(self, algo):
        """
        The variables which are being modified by the model.

        Parameters
        ----------
        algo : foxes.core.Algorithm
            The calculation algorithm

        Returns
        -------
        output_vars : list of str
            The output variable names

        """
        return self.turbine_model.output_farm_vars(algo)

    def calculate(self, algo, mdata, fdata, **parameters):
        """ "
        The main model calculation.

        This function is executed on a single chunk of data,
        all computations should be based on numpy arrays.

        Parameters
        ----------
        algo : foxes.core.Algorithm
            The calculation algorithm
        mdata : foxes.core.Data
            The model data
        fdata : foxes.core.Data
            The farm data
        **parameters : dict, optional
            Init parameters forwarded to the turbine model

        Returns
        -------
        results : dict
            The resulting data, keys: output variable str.
            Values: numpy.ndarray with shape (n_states, n_turbines)

        """
        s = np.ones((algo.n_states, algo.n_turbines), dtype=bool)
        return self.turbine_model.calculate(algo, mdata, fdata, st_sel=s, **parameters)

    def finalize(self, algo, verbosity=0):
        """
        Finalizes the model.

        Parameters
        ----------
        algo : foxes.core.Algorithm
            The calculation algorithm
        verbosity : int
            The verbosity level

        """
        if self.turbine_model.initialized:
            self.turbine_model.finalize(algo, verbosity)
        super().finalize(algo, verbosity)
