from __future__ import annotations
# mypy: disable-error-code=override

from abc import abstractmethod
import numpy as np
from typing import TYPE_CHECKING, Any, cast

from foxes.config import config
import foxes.constants as FC
import foxes.variables as FV

from .data_calc_model import DataCalcModel
from .model import Model

if TYPE_CHECKING:
    from foxes.core.algorithm import Algorithm
    from foxes.core.data import FData, MData


class FarmDataModel(DataCalcModel):
    """
    Abstract base class for models that modify
    farm data.

    :group: core

    """

    @abstractmethod
    def output_farm_vars(self, algo: Algorithm) -> list[str]:
        """
        Return the variables modified by the model.

        Parameters
        ----------
        algo
            The calculation algorithm.

        Returns
        -------
        output_vars
            The output variable names.

        """
        return []

    def output_coords(self) -> tuple[str, ...]:
        """
        Gets the coordinates of all output arrays

        Returns
        -------
        dims
            The coordinates of all output arrays

        """
        return (FC.STATE, FC.TURBINE)

    def ensure_output_vars(
        self,
        algo: Algorithm,
        fdata: FData,
        defaults: dict[str, Any] | None = None,
    ) -> None:
        """
        Ensure the output variables are present in the farm data.

        Parameters
        ----------
        algo
            The calculation algorithm.
        fdata
            The farm data.
        defaults
            Default values for the output variables. Keys are variable names,
            values are scalars or array-like data with shape
            ``(n_states, n_turbines)``.

        """
        defs = {
            FV.YAWM: 0.0,
        }
        if defaults is not None:
            defs.update(defaults)

        for var in self.output_farm_vars(algo):
            if var not in fdata:
                fdata.add(
                    var,
                    np.full(
                        (fdata.n_states, fdata.n_turbines),
                        defs.get(var, np.nan),
                        dtype=config.dtype_double,
                    ),
                    (FC.STATE, FC.TURBINE),
                )

    @abstractmethod
    def calculate(
        self,
        algo: Algorithm,
        mdata: MData,
        fdata: FData,
    ) -> dict[str, np.ndarray]:
        """
        Run the main model calculation.

        This function is executed on a single chunk of data; all computations
        should be based on NumPy arrays.

        Parameters
        ----------
        algo
            The calculation algorithm.
        mdata
            The model data.
        fdata
            The farm data.

        Returns
        -------
        results
            The resulting data keyed by output variable name. Values are
            ``numpy.ndarray`` objects with shape ``(n_states, n_turbines)``.

        """
        pass

    def run_calculation(
        self,
        algo: Algorithm,
        *data: tuple[Any, ...],
        out_vars: list[str],
        **calc_pars: Any,
    ) -> Any:
        """
        Starts the model calculation in parallel, via
        xarray's `apply_ufunc`.

        Typically this function is called by algorithms.

        Parameters
        ----------
        algo
            The calculation algorithm
        *data: tuple of xarray.Dataset
            The input data
        out_vars
            The calculation output variables
        **calc_pars
            Additional arguments for the `calculate` function

        Returns
        -------
        results
            The calculation results

        """
        return super().run_calculation(  # type: ignore[misc]
            algo,
            *data,
            out_vars=out_vars,
            loop_dims=[FC.STATE],
            out_core_vars=[FC.TURBINE, FC.VARS],
            **calc_pars,
        )

    def __add__(self, m: Any) -> FarmDataModelList:
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
    models
        The model list

    :group: core

    """

    def __init__(self, models: list[FarmDataModel] | None = None) -> None:
        """
        Constructor.

        Parameters
        ----------
        models
            The model list

        """
        super().__init__()
        self.models = [] if models is None else models

    def __repr__(self) -> str:
        return f"{type(self).__name__}({[m.name for m in self.models]})"

    def append(self, model: FarmDataModel) -> None:
        """
        Add a model to the list

        Parameters
        ----------
        model
            The model to add

        """
        self.models.append(model)

    def insert(self, index: int, model: FarmDataModel) -> None:
        """
        Insert a model into the list

        Parameters
        ----------
        index
            The index in the model list
        model
            The model to insert

        """
        self.models.insert(index, model)

    def sub_models(self) -> list[Model]:
        """
        List of all sub-models

        Returns
        -------
        smdls
            Names of all sub models

        """
        return cast(list[Model], self.models)

    def output_farm_vars(self, algo: Algorithm) -> list[str]:
        """
        The variables which are being modified by the model.

        Parameters
        ----------
        algo
            The calculation algorithm

        Returns
        -------
        output_vars
            The output variable names

        """
        ovars = []
        for m in self.models:
            ovars += m.output_farm_vars(algo)

        return list(dict.fromkeys(ovars))

    def calculate(
        self,
        algo: Algorithm,
        mdata: MData,
        fdata: FData,
        parameters: list[dict[str, Any]] | None = None,
    ) -> dict[str, np.ndarray]:
        """
        The main model calculation.

        This function is executed on a single chunk of data,
        all computations should be based on numpy arrays.

        Parameters
        ----------
        algo
            The calculation algorithm
        mdata
            The model data
        fdata
            The farm data
        parameters
            A list of parameter dicts, one for each model

        Returns
        -------
        results
            The resulting data, keys: output variable str.
            Values are arrays with shape (n_states, n_turbines)

        """
        self.ensure_output_vars(algo, fdata)

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
