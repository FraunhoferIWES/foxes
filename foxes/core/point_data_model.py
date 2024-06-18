import numpy as np
from abc import abstractmethod

from .data_calc_model import DataCalcModel
import foxes.constants as FC


class PointDataModel(DataCalcModel):
    """
    Abstract base class for models that modify
    point based data.

    :group: core

    """

    @abstractmethod
    def output_point_vars(self, algo):
        """
        The variables which are being modified by the model.

        Returns
        -------
        output_vars: list of str
            The output variable names

        """
        return []

    def ensure_variables(self, algo, mdata, fdata, tdata):
        """
        Add variables to tdata, initialized with NaN

        Parameters
        ----------
        algo: foxes.core.Algorithm
            The calculation algorithm
        mdata: foxes.core.Data
            The model data
        fdata: foxes.core.Data
            The farm data
        tdata: foxes.core.Data
            The target point data

        """
        for v in self.output_point_vars(algo):
            if v not in tdata:
                tdata[v] = np.full(
                    (tdata.n_states, tdata.n_targets, tdata.n_tpoints),
                    np.nan,
                    dtype=FC.DTYPE,
                )
                tdata.dims[v] = (FC.STATE, FC.TARGET, FC.TPOINT)

    @abstractmethod
    def calculate(self, algo, mdata, fdata, tdata):
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
        tdata: foxes.core.TData
            The target point data

        Returns
        -------
        results: dict
            The resulting data, keys: output variable str.
            Values: numpy.ndarray with shape
            (n_states, n_targets, n_tpoints)

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
        results = super().run_calculation(
            algo,
            *data,
            out_vars=out_vars,
            loop_dims=[FC.STATE, FC.TARGET],
            out_core_vars=[FC.TPOINT, FC.VARS],
            **calc_pars,
        )
        if results.sizes[FC.TPOINT] != 1:
            raise ValueError(
                f"PointDataModel '{self.name}': Expecting dimension '{FC.TPOINT}' of size 1, found {results.sizes[FC.TPOINT]}"
            )
        return results.sel({FC.TPOINT: 0}).rename({FC.TARGET: FC.POINT})

    def __add__(self, m):
        if isinstance(m, list):
            return PointDataModelList([self] + m)
        elif isinstance(m, PointDataModelList):
            return PointDataModelList([self] + m.models)
        else:
            return PointDataModelList([self, m])


class PointDataModelList(PointDataModel):
    """
    A list of point data models.

    By using the PointDataModelList the models'
    `calculate` functions are called together
    under one common call of xarray's `apply_ufunc`.

    Attributes
    ----------
    models: list of foxes.core.PointDataModel
        The model list

    :group: core

    """

    def __init__(self, models=[]):
        """
        Constructor.

        Parameters
        ----------
        models: list of foxes.core.PointDataModel
            The model list

        """
        super().__init__()
        self.models = models

    def append(self, model):
        """
        Add a model to the list

        Parameters
        ----------
        model: foxes.core.PointDataModel
            The model to add

        """
        self.models.append(model)

    def sub_models(self):
        """
        List of all sub-models

        Returns
        -------
        smdls: list of foxes.core.Model
            Names of all sub models

        """
        return self.models

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
        ovars = []
        for m in self.models:
            ovars += m.output_point_vars(algo)
        return list(dict.fromkeys(ovars))

    def calculate(self, algo, mdata, fdata, tdata, parameters=None):
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
        tdata: foxes.core.TData
            The target point data
        parameters: list of dict, optional
            A list of parameter dicts, one for each model

        Returns
        -------
        results: dict
            The resulting data, keys: output variable str.
            Values: numpy.ndarray with shape
            (n_states, n_targets, n_tpoints)

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
            res = m.calculate(algo, mdata, fdata, tdata, **parameters[mi])
            tdata.update(res)

        return {v: tdata[v] for v in self.output_point_vars(algo)}
