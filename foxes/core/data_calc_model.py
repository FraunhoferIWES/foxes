from abc import abstractmethod

from .model import Model


class DataCalcModel(Model):
    """
    Abstract base class for models
    that run calculations based on model data.

    Attributes
    ----------
    load_mode: str
        The data loading mode

    :group: core

    """

    def __init__(self, *args, load_mode="preload", **kwargs):
        """
        Constructor.

        Parameters
        ----------
        args: tuple, optional
            Additional parameters for constructor
        load_mode: str
            The data loading mode, e.g. 'preload'
        kwargs: dict, optional
            Additional parameters for constructor

        """
        super().__init__(*args, **kwargs)
        self.load_mode = load_mode

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

    def load_chunk_data(self, algo, *data):
        """
        Load chunk data according to load mode.

        This function adds data to mdata.

        Parameters
        ----------
        algo: foxes.core.Algorithm
            The calculation algorithm
        data: tuple of foxes.core.Data, optional
            The input data, typically either (mdata, fdata) in
            the case of farm calculations, or (mdata, fdata, tdata)
            for point data calculations

        """
        for m in self.sub_models():
            m.load_chunk_data(algo, *data)

        if self.load_mode != "preload":
            raise NotImplementedError(
                f"States '{self.name}': load mode '{self.load_mode}' not implemented."
            )

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
        data: tuple of foxes.core.Data, optional
            The input data, typically either (mdata, fdata) in
            the case of farm calculations, or (mdata, fdata, tdata)
            for point data calculations
        parameters: dict, optional
            The calculation parameters

        Returns
        -------
        results: dict
            The resulting data, keys: output variable str.
            Values: numpy.ndarray

        """
        self.load_chunk_data(algo, *data)
