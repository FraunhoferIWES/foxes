from abc import abstractmethod

from foxes.core.data_calc_model import DataCalcModel

class FarmDataModel(DataCalcModel):
    """
    Abstract base class for models that modify
    farm data.

    Parameters
    ----------
    pre_rotor : bool
        Flag for running this model before
        running the rotor model.
    
    Attributes
    ----------
    pre_rotor : bool
        Flag for running this model before
        running the rotor model.

    """

    def __init__(self, pre_rotor=False):
        super().__init__()
        self.pre_rotor = pre_rotor

    @abstractmethod
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
        return []

    @abstractmethod
    def calculate(self, algo, mdata, fdata):
        """"
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
        
        Returns
        -------
        results : dict
            The resulting data, keys: output variable str.
            Values: numpy.ndarray with shape (n_states, n_turbines)

        """
        pass
