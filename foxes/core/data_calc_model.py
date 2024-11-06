from abc import abstractmethod

from .model import Model


class DataCalcModel(Model):
    """
    Abstract base class for models
    that run calculation on xarray Dataset
    data.

    The calculations are run via xarray's
    `apply_ufunc` function, i.e., they run in
    parallel depending on the dask settings.

    For each individual data chunk the `calculate`
    function is called.

    :group: core

    """

    @abstractmethod
    def output_coords(self):
        """
        Gets the coordinates of all output arrays

        Returns
        -------
        dims: tuple of str
            The coordinates of all output arrays

        """
        pass

    @abstractmethod
    def calculate(self, algo, *data, **parameters):
        """
        The main model calculation.

        This function is executed on a single chunk of data,
        all computations should be based on numpy arrays.

        Parameters
        ----------
        algo: foxes.core.Algorithm
            The calculation algorithm
        data: tuple of foxes.core.Data
            The input data
        parameters: dict, optional
            The calculation parameters

        Returns
        -------
        results: dict
            The resulting data, keys: output variable str.
            Values: numpy.ndarray

        """
        pass
