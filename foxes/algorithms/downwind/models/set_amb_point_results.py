import foxes.variables as FV
from foxes.core import PointDataModel


class SetAmbPointResults(PointDataModel):
    """
    This model copies point results to ambient results.

    Attributes
    ----------
    pvars: list of str
        The point variables to be treated
    vars: list of str
        The variables to be copied to output

    :group: algorithms.downwind.models

    """

    def __init__(self):
        """
        Constructor.
        """
        super().__init__()
        self.pvars = None
        self.vars = None

    def initialize(self, algo, loaded_data=None, force=False, verbosity=0):
        """
        Initializes the model.

        Parameters
        ----------
        algo: foxes.core.Algorithm
            The calculation algorithm
        loaded_data: dict, optional
            Data that has already been loaded, to be extended by this function.
            Keys are "coords", a dict with entries `dim_name_str -> dim_array`;
            "data_vars", a dict with entries `name_str -> (dim_tuple, data_ndarray)`;
            and "extra_data", a dict with non-array additional data.
        force: bool
            Overwrite existing data
        verbosity: int
            The verbosity level, 0 = silent

        Returns
        -------
        loaded_data: dict
            The loaded data, containing keys "coords", "data_vars", and "extra_data".
            Keys are "coords", a dict with entries `dim_name_str -> dim_array`;
            "data_vars", a dict with entries `name_str -> (dim_tuple, data_ndarray)`;
            and "extra_data", a dict with non-array additional data.

        """
        self.pvars = algo.states.output_point_vars(algo)
        self.vars = [v for v in self.pvars if v in FV.var2amb]
        return super().initialize(algo, loaded_data, force, verbosity)

    def output_point_vars(self, algo):
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
        return [FV.var2amb[v] for v in self.vars] + [FV.WEIGHT]

    def calculate(self, algo, mdata, fdata, tdata):
        """
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
        tdata: foxes.core.TData
            The point data

        Returns
        -------
        results: dict
            The resulting data, keys: output variable str.
            Values: numpy.ndarray with shape (n_states, n_points)

        """
        ovars = self.output_point_vars(algo)
        for v in self.vars:
            tdata.add(FV.var2amb[v], tdata[v].copy(), tdata.dims[v])
        return {v: tdata[v] for v in ovars}
