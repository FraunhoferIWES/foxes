from abc import abstractmethod

from .farm_data_model import FarmDataModel


class TurbineModel(FarmDataModel):
    """
    Abstract base class for turbine models.

    Turbine models are FarmDataModels that run
    on a selection of turbines.

    """

    def initialize(self, algo, st_sel, verbosity=0):
        """
        Initializes the model.

        Parameters
        ----------
        algo : foxes.core.Algorithm
            The calculation algorithm
        st_sel : numpy.ndarray of bool
            The state-turbine selection,
            shape: (n_states, n_turbines)
        verbosity : int
            The verbosity level

        """
        super().initialize(algo, verbosity=verbosity)

    @abstractmethod
    def calculate(self, algo, mdata, fdata, st_sel):
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
        st_sel : numpy.ndarray of bool
            The state-turbine selection,
            shape: (n_states, n_turbines)

        Returns
        -------
        results : dict
            The resulting data, keys: output variable str.
            Values: numpy.ndarray with shape (n_states, n_turbines)

        """
        pass

    def finalize(self, algo, results, st_sel, clear_mem=False, verbosity=0):
        """
        Finalizes the model.

        Parameters
        ----------
        algo : foxes.core.Algorithm
            The calculation algorithm
        results : xarray.Dataset
            The calculation results
        st_sel : numpy.ndarray of bool
            The state-turbine selection,
            shape: (n_states, n_turbines)
        clear_mem : bool
            Flag for deleting model data and
            resetting initialization flag
        verbosity : int
            The verbosity level

        """
        super().finalize(algo, results, clear_mem, verbosity)
