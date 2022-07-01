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

    def initialize(self, algo, verbosity=0, **parameters):
        """
        Initializes the model.

        Parameters
        ----------
        algo : foxes.core.Algorithm
            The calculation algorithm
        verbosity : int
            The verbosity level
        **parameters : dict, optional
            Init parameters forwarded to the turbine model

        """
        if not self.turbine_model.initialized:
            s = np.ones((algo.n_states, algo.n_turbines), dtype=bool)
            self.turbine_model.initialize(
                algo, st_sel=s, verbosity=verbosity, **parameters
            )
        super().initialize(algo, verbosity=verbosity)

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

    def finalize(self, algo, results, verbosity=0, **parameters):
        """
        Finalizes the model.

        Parameters
        ----------
        algo : foxes.core.Algorithm
            The calculation algorithm
        results : xarray.Dataset
            The calculation results
        clear_mem : bool
            Flag for deleting model data and
            resetting initialization flag
        verbosity : int
            The verbosity level
        **parameters : dict, optional
            Init parameters forwarded to the turbine model

        """
        s = np.ones((algo.n_states, algo.n_turbines), dtype=bool)
        self.turbine_model.finalize(
            algo, results, st_sel=s, verbosity=verbosity, **parameters
        )
