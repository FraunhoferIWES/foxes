from abc import abstractmethod

from .data_calc_model import DataCalcModel
import foxes.variables as FV


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

        Returns
        -------
        results : dict
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
        algo : foxes.core.Algorithm
            The calculation algorithm
        *data : tuple of xarray.Dataset
            The input data
        out_vars: list of str
            The calculation output variables
        **calc_pars : dict, optional
            Additional arguments for the `calculate` function

        Returns
        -------
        results : xarray.Dataset
            The calculation results

        """
        return super().run_calculation(
            algo,
            *data,
            out_vars=out_vars,
            loop_dims=[FV.STATE],
            out_core_vars=[FV.TURBINE, FV.VARS],
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

    Parameters
    ----------
    models : list of foxes.core.FarmDataModel
        The model list

    Attributes
    ----------
    models : list of foxes.core.FarmDataModel
        The model list

    """

    def __init__(self, models=[]):
        super().__init__()
        self.models = models

    def append(self, model):
        """
        Add a model to the list

        Parameters
        ----------
        model : foxes.core.FarmDataModel
            The model to add

        """
        self.models.append(model)

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
        ovars = []
        for m in self.models:
            ovars += m.output_farm_vars(algo)
        return list(dict.fromkeys(ovars))

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
        if verbosity > 1:
            print(f"-- {self.name}: Starting initialization -- ")

        idata = super().initialize(algo, verbosity)
        algo.update_idata(self.models, idata=idata, verbosity=verbosity)

        if verbosity > 1:
            print(f"-- {self.name}: Finished initialization -- ")

        return idata

    def calculate(self, algo, mdata, fdata, parameters=[]):
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
        parameters : list of dict, optional
            A list of parameter dicts, one for each model

        Returns
        -------
        results : dict
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

    def finalize(self, algo, verbosity=0):
        """
        Finalizes the model.

        Parameters
        ----------
        algo : foxes.core.Algorithm
            The calculation algorithm
        verbosity : int
            The verbosity level, 0 means silent

        """
        for m in self.models:
            if m.initialized:
                algo.finalize_model(m, verbosity)

        super().finalize(algo, verbosity)
