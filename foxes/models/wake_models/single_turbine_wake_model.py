from __future__ import annotations

import numpy as np
from typing import TYPE_CHECKING, Any, cast

from foxes.core import Model, WakeModel, WindVectorWakeSuperposition
from foxes.utils import new_instance
import foxes.variables as FV

if TYPE_CHECKING:
    from foxes.core.algorithm import Algorithm
    from foxes.core.data import FData, MData, TData
    from foxes.core.model import LoadedData


class SingleTurbineWakeModel(WakeModel):
    """
    Abstract base class for wake models that represent
    a single turbine wake

    Single turbine wake models depend on superposition models.

    Attributes
    ----------
    wind_superposition
        The wind superposition model name (vector or compenent model),
        will be looked up in model book
    other_superpositions
        The superpositions for other than (ws, wd) variables.
        Key: variable name str, value: The wake superposition
        model name, will be looked up in model book
    vec_superp
        The wind vector wake superposition model
    superp
        The superposition dict, key: variable name str,
        value: the corresponding wake superposition model


    """

    def __init__(
        self,
        wind_superposition: str | None = None,
        other_superpositions: dict[str, str] | None = None,
    ) -> None:
        """
        Constructor.

        Parameters
        ----------
        wind_superposition
            The wind superposition model name (vector or compenent model),
            will be looked up in model book
        other_superpositions
            The superpositions for other than (ws, wd) variables.
            Key: variable name str, value: The wake superposition
            model name, will be looked up in model book

        """
        super().__init__()
        self.wind_superposition = wind_superposition
        self.other_superpositions = (
            {} if other_superpositions is None else dict(other_superpositions)
        )
        self.vec_superp: Any | None = None
        self.superp: dict[str, Any] = {}

        for v in [FV.WS, FV.WD]:
            assert v not in self.other_superpositions, (
                f"Wake model '{self.name}': Found variable '{v}' among 'other_superposition' keyword, use 'wind_superposition' instead"
            )

        self.__has_vector_superp = False

    @property
    def has_vector_wind_superp(self) -> bool:
        """
        This model uses a wind vector superposition

        Returns
        -------
        has_vector_wind_superp
            Flag for wind vector superposition

        """
        return self.__has_vector_superp

    def sub_models(self) -> list[Model]:
        """
        List of all sub-models

        Returns
        -------
        smdls
            Names of all sub models

        """
        w: list[Model] = (
            [cast(Model, self.vec_superp)] if self.vec_superp is not None else []
        )
        return w + list(self.superp.values())

    def initialize(
        self,
        algo: Algorithm,
        loaded_data: LoadedData | None = None,
        force: bool = False,
        verbosity: int = 0,
    ) -> LoadedData:
        """
        Initializes the model.

        Parameters
        ----------
        algo
            The calculation algorithm
        loaded_data
            Data that has already been loaded, to be extended by this function.
            Keys are "coords", a dict with entries `dim_name_str -> dim_array`;
            "data_vars", a dict with entries `name_str -> (dim_tuple, data_ndarray)`;
            and "extra_data", a dict with non-array additional data.
        force
            Overwrite existing data
        verbosity
            The verbosity level, 0 = silent

        Returns
        -------
        loaded_data
            The loaded data, containing keys "coords", "data_vars", and "extra_data".
            Keys are "coords", a dict with entries `dim_name_str -> dim_array`;
            "data_vars", a dict with entries `name_str -> (dim_tuple, data_ndarray)`;
            and "extra_data", a dict with non-array additional data.

        """
        self.superp = {
            v: algo.mbook.wake_superpositions[s]
            for v, s in self.other_superpositions.items()
        }

        if self.wind_superposition is not None:
            self.vec_superp = algo.mbook.wake_superpositions[self.wind_superposition]
            self.__has_vector_superp = isinstance(
                self.vec_superp, WindVectorWakeSuperposition
            )
            if self.__has_vector_superp:
                self._has_uv = True
            else:
                self.superp[FV.WS] = self.vec_superp
                self.vec_superp = None

        return super().initialize(
            algo=algo,
            loaded_data=loaded_data,
            force=force,
            verbosity=verbosity,
        )

    def finalize_wake_deltas(
        self,
        algo: Algorithm,
        mdata: MData,
        fdata: FData,
        tdata: TData,
        wake_deltas: dict[str, np.ndarray],
    ) -> None:
        """
        Finalize the wake calculation.

        Modifies wake_deltas on the fly.

        Parameters
        ----------
        algo
            The calculation algorithm
        mdata
            The model data
        fdata
            The farm data
        tdata
            The target point data
        wake_deltas
            The wake deltas object at the selected target
            turbines. Keys are variable names and values are arrays
            with shape (n_states, n_targets, n_tpoints)

        """
        for v in wake_deltas.keys():
            if v != FV.UV:
                try:
                    wake_deltas[v] = self.superp[v].calc_final_wake_delta(
                        algo, mdata, fdata, tdata, v, wake_deltas[v]
                    )
                except KeyError:
                    raise KeyError(
                        f"Wake model '{self.name}': Variable '{v}' appears to be modified, missing superposition model"
                    )

        if FV.UV in wake_deltas:
            assert self.has_vector_wind_superp, (
                f"{self.name}: Expecting wind vector superposition, got '{self.wind_superposition}'"
            )
            vec_superp = self.vec_superp
            assert vec_superp is not None
            dws, dwd = vec_superp.calc_final_wake_delta_uv(
                algo, mdata, fdata, tdata, wake_deltas.pop(FV.UV)
            )

            wake_deltas[FV.WS] = dws
            wake_deltas[FV.WD] = dwd


class TurbineInductionModel(SingleTurbineWakeModel):
    """
    Abstract base class for turbine induction models.


    """

    @property
    def affects_downwind(self) -> bool:
        """
        Flag for downwind or upwind effects
        on other turbines

        Returns
        -------
        affects_downwind
            Flag for downwind effects by this model

        """
        return False

    @classmethod
    def new(
        cls,
        wmodel_type: str,
        *args: Any,
        **kwargs: Any,
    ) -> TurbineInductionModel:
        """
        Run-time turbine induction model factory.

        Parameters
        ----------
        wmodel_type
            The selected derived class name
        args
            Additional parameters for constructor
        kwargs
            Additional parameters for constructor

        """
        return cast(
            TurbineInductionModel, new_instance(cls, wmodel_type, *args, **kwargs)
        )
