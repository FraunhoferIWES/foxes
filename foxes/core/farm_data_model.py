from abc import abstractmethod
import numpy as np

from .data_calc_model import DataCalcModel
import foxes.constants as FC


class FarmDataModel(DataCalcModel):
    """
    Abstract base class for models that modify
    farm data.

    Attributes
    ----------
    pre_rotor: bool
        Flag for running this model before
        running the rotor model.

    :group: core

    """

    def __init__(self, pre_rotor=False):
        """
        Constructor.

        Parameters
        ----------
        pre_rotor: bool
            Flag for running this model before
            running the rotor model.

        """
        super().__init__()
        self.pre_rotor = pre_rotor

    @abstractmethod
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
        return []

    def ensure_variables(self, algo, mdata, fdata):
        """
        Add variables to fdata, initialized with NaN

        Parameters
        ----------
        algo: foxes.core.Algorithm
            The calculation algorithm
        mdata: foxes.core.Data
            The model data
        fdata: foxes.core.Data
            The farm data

        """
        n_states = fdata.n_states
        n_turbines = fdata.n_turbines
        for v in self.output_farm_vars(algo):
            if v not in fdata:
                fdata[v] = np.full((n_states, n_turbines), np.nan, dtype=FC.DTYPE)
                fdata.dims[v] = (FC.STATE, FC.TURBINE)

    @abstractmethod
    def calculate(self, algo, mdata, fdata):
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

        Returns
        -------
        results: dict
            The resulting data, keys: output variable str.
            Values: numpy.ndarray with shape (n_states, n_turbines)

        """
        pass

    def run_calculation(self, algo, *data, out_vars, **calc_pars):
        """
        Starts the model calculation in parallel, via
        xarray's `apply_ufunc`.

        Typically this function is called by algorithms.

        Parameters
        ----------
        algo: foxes.core.Algorithm
            The calculation algorithm
        *data: tuple of xarray.Dataset
            The input data
        out_vars: list of str
            The calculation output variables
        **calc_pars: dict, optional
            Additional arguments for the `calculate` function

        Returns
        -------
        results: xarray.Dataset
            The calculation results

        """
        return super().run_calculation(
            algo,
            *data,
            out_vars=out_vars,
            loop_dims=[FC.STATE],
            out_core_vars=[FC.TURBINE, FC.VARS],
            **calc_pars,
        )

    def __add__(self, m):
        if isinstance(m, list):
            return FarmDataModelList([self] + m)
        elif isinstance(m, FarmDataModelList):
            return FarmDataModelList([self] + m.models)
        else:
            return FarmDataModelList([self, m])


class FarmDataModelList(FarmDataModel):
    """
    A list of farm data models.

    By using the FarmDataModelList the models'
    `calculate` functions are called together
    under one common call of xarray's `apply_ufunc`.

    Attributes
    ----------
    models: list of foxes.core.FarmDataModel
        The model list

    :group: core

    """

    def __init__(self, models=[]):
        """
        Constructor.

        Parameters
        ----------
        models: list of foxes.core.FarmDataModel
            The model list

        """
        super().__init__()
        self.models = models

    def append(self, model):
        """
        Add a model to the list

        Parameters
        ----------
        model: foxes.core.FarmDataModel
            The model to add

        """
        self.models.append(model)

    def insert(self, index, model):
        """
        Insert a model into the list

        Parameters
        ----------
        index: int
            The index in the model list
        model: foxes.core.FarmDataModel
            The model to insert

        """
        self.models.insert(index, model)

    def sub_models(self):
        """
        List of all sub-models

        Returns
        -------
        smdls: list of foxes.core.Model
            Names of all sub models

        """
        return self.models

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
        ovars = []
        for m in self.models:
            ovars += m.output_farm_vars(algo)
        return list(dict.fromkeys(ovars))

    def calculate(self, algo, mdata, fdata, parameters=[]):
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
        parameters: list of dict, optional
            A list of parameter dicts, one for each model

        Returns
        -------
        results: dict
            The resulting data, keys: output variable str.
            Values: numpy.ndarray with shape (n_states, n_turbines)

        """
        if parameters is None:
            parameters = [{}] * len(self.models)
        elif not isinstance(parameters, list):
            raise ValueError(
                f"{self.name}: Wrong parameters type, expecting list, got {type(parameters).__name__}"
            )
        elif len(parameters) != len(self.models):
            raise ValueError(
                f"{self.name}: Wrong parameters length, expecting list with {len(self.models)} entries, got {len(parameters)}"
            )

        for mi, m in enumerate(self.models):
            # print("MLIST VARS BEFORE",m.name,list(fdata.keys()),parameters[mi])
            res = m.calculate(algo, mdata, fdata, **parameters[mi])
            fdata.update(res)

        return {v: fdata[v] for v in self.output_farm_vars(algo)}
