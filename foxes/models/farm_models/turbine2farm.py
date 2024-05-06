import numpy as np

from foxes.core import FarmModel


class Turbine2FarmModel(FarmModel):
    """
    Wrapper that promotes turbine models
    into farm models, simply by selecting
    all turbines.

    Attributes
    ----------
    turbine_model: foxes.core.TurbineModel
        The turbine model

    :group: models.farm_models

    """

    def __init__(self, turbine_model):
        """
        Constructor.

        Parameters
        ----------
        turbine_model: foxes.core.TurbineModel
            The turbine model

        """
        super().__init__()
        self.turbine_model = turbine_model
        self.name = turbine_model.name + "_t2f"

    def __repr__(self):
        return f"{type(self).__name__}({self.turbine_model})"

    def sub_models(self):
        """
        List of all sub-models

        Returns
        -------
        smdls: list of foxes.core.Model
            Names of all sub models

        """
        return [self.turbine_model]

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
        return self.turbine_model.output_farm_vars(algo)

    def calculate(self, algo, mdata, fdata, **parameters):
        """ "
        The main model calculation.

        This function is executed on a single chunk of data,
        all computations should be based on numpy arrays.

        Parameters
        ----------
        algo: foxes.core.Algorithm
            The calculation algorithm
        mdata: foxes.core.MData
            The model data
        fdata: foxes.core.FData
            The farm data
        **parameters: dict, optional
            Init parameters forwarded to the turbine model

        Returns
        -------
        results: dict
            The resulting data, keys: output variable str.
            Values: numpy.ndarray with shape (n_states, n_turbines)

        """
        s = np.ones((algo.n_states, algo.n_turbines), dtype=bool)
        return self.turbine_model.calculate(algo, mdata, fdata, st_sel=s, **parameters)
