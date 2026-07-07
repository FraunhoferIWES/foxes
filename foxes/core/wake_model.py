from __future__ import annotations
# mypy: disable-error-code=override

from abc import abstractmethod
from typing import TYPE_CHECKING, Any, cast

import numpy as np

from foxes.utils import new_instance
import foxes.variables as FV

from .model import Model
from .wake_superposition import WindVectorWakeSuperposition

if TYPE_CHECKING:
    from foxes.core.algorithm import Algorithm
    from foxes.core.data import FData, MData, TData
    from foxes.core.model import LoadedData


class WakeModel(Model):
    """
    Abstract base class for wake models.

    :group: core

    """

    def __init__(self) -> None:
        """
        Constructor.
        """
        super().__init__()
        self._has_uv = False

    @property
    def affects_ws(self) -> bool:
        """
        Flag for wind speed wake models

        Returns
        -------
        dws: bool
            If True, this model affects wind speed

        """
        return False

    @property
    def affects_downwind(self) -> bool:
        """
        Flag for downwind or upwind effects
        on other turbines

        Returns
        -------
        dwnd: bool
            Flag for downwind effects by this model

        """
        return True

    @property
    def has_uv(self) -> bool:
        """
        This model uses wind vector data

        Returns
        -------
        hasuv: bool
            Flag for wind vector data

        """
        return self._has_uv

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
        wake_deflection = algo.wake_deflection
        if wake_deflection is not None and self.affects_ws and wake_deflection.has_uv:
            self._has_uv = True
        return super().initialize(
            algo=algo,
            loaded_data=loaded_data,
            force=force,
            verbosity=verbosity,
        )

    @abstractmethod
    def new_wake_deltas(
        self,
        algo: Algorithm,
        mdata: MData,
        fdata: FData,
        tdata: TData,
    ) -> dict[str, np.ndarray]:
        """
        Creates new empty wake delta arrays.

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
        wake_deltas: dict
            Key: variable name, value: The zero filled
            wake deltas, shape: (n_states, n_targets, n_tpoints, ...)

        """
        pass

    @abstractmethod
    def contribute(
        self,
        algo: Algorithm,
        mdata: MData,
        fdata: FData,
        tdata: TData,
        downwind_index: int,
        wake_coos: np.ndarray,
        wake_deltas: dict[str, np.ndarray],
    ) -> None:
        """
        Modifies wake deltas at target points by
        contributions from the specified wake source turbines.

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
        downwind_index: int
            The index of the wake causing turbine
            in the downwind order
        wake_coos: numpy.ndarray
            The wake frame coordinates of the evaluation
            points, shape: (n_states, n_targets, n_tpoints, 3)
        wake_deltas: dict
            The wake deltas. Key: variable name,
            value: numpy.ndarray with shape
            (n_states, n_targets, n_tpoints, ...)

        """
        pass

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
        algo: foxes.core.Algorithm
            The calculation algorithm
        mdata: foxes.core.MData
            The model data
        fdata: foxes.core.FData
            The farm data
        tdata: foxes.core.TData
            The target point data
        wake_deltas: dict
            The wake deltas object at the selected target
            turbines. Key: variable str, value: numpy.ndarray
            with shape (n_states, n_targets, n_tpoints)

        """
        pass

    @classmethod
    def new(cls, wmodel_type: str, *args: Any, **kwargs: Any) -> WakeModel:
        """
        Run-time wake model factory.

        Parameters
        ----------
        wmodel_type: str
            The selected derived class name
        args: tuple, optional
            Additional parameters for constructor
        kwargs: dict, optional
            Additional parameters for constructor

        """
        return new_instance(cls, wmodel_type, *args, **kwargs)


class SingleTurbineWakeModel(WakeModel):
    """
    Abstract base class for wake models that represent
    a single turbine wake

    Single turbine wake models depend on superposition models.

    Attributes
    ----------
    wind_superposition: str
        The wind superposition model name (vector or compenent model),
        will be looked up in model book
    other_superpositions: dict
        The superpositions for other than (ws, wd) variables.
        Key: variable name str, value: The wake superposition
        model name, will be looked up in model book
    vec_superp: foxes.core.WindVectorWakeSuperposition or None
        The wind vector wake superposition model
    superp: dict
        The superposition dict, key: variable name str,
        value: `foxes.core.WakeSuperposition`

    :group: models.wake_models

    """

    def __init__(
        self,
        wind_superposition: str | None = None,
        other_superpositions: dict[str, str] = {},
    ) -> None:
        """
        Constructor.

        Parameters
        ----------
        wind_superposition: str, optional
            The wind superposition model name (vector or compenent model),
            will be looked up in model book
        other_superpositions: dict
            The superpositions for other than (ws, wd) variables.
            Key: variable name str, value: The wake superposition
            model name, will be looked up in model book

        """
        super().__init__()
        self.wind_superposition = wind_superposition
        self.other_superpositions = other_superpositions
        self.vec_superp: Any | None = None
        self.superp: dict[str, Any] = {}

        for v in [FV.WS, FV.WD]:
            assert v not in other_superpositions, (
                f"Wake model '{self.name}': Found variable '{v}' among 'other_superposition' keyword, use 'wind_superposition' instead"
            )

        self.__has_vector_superp = False

    @property
    def has_vector_wind_superp(self) -> bool:
        """
        This model uses a wind vector superposition

        Returns
        -------
        hasv: bool
            Flag for wind vector superposition

        """
        return self.__has_vector_superp

    def sub_models(self) -> list[Model]:
        """
        List of all sub-models

        Returns
        -------
        smdls: list of foxes.core.Model
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
        algo: foxes.core.Algorithm
            The calculation algorithm
        mdata: foxes.core.MData
            The model data
        fdata: foxes.core.FData
            The farm data
        tdata: foxes.core.TData
            The target point data
        wake_deltas: dict
            The wake deltas object at the selected target
            turbines. Key: variable str, value: numpy.ndarray
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

    :group: core

    """

    @property
    def affects_downwind(self) -> bool:
        """
        Flag for downwind or upwind effects
        on other turbines

        Returns
        -------
        dwnd: bool
            Flag for downwind effects by this model

        """
        return False

    @classmethod
    def new(
        cls,
        induction_type: str,
        *args: Any,
        **kwargs: Any,
    ) -> TurbineInductionModel:
        """
        Run-time turbine induction model factory.

        Parameters
        ----------
        induction_type: str
            The selected derived class name
        args: tuple, optional
            Additional parameters for constructor
        kwargs: dict, optional
            Additional parameters for constructor

        """
        return new_instance(cls, induction_type, *args, **kwargs)


class WakeK(Model):
    """
    Handler for the wake growth parameter k

    Attributes
    ----------
    k_var: str
        The name of the k variable
    ti_var: str
        The name of the TI variable

    :group: core

    """

    def __init__(
        self,
        k: float | None = None,
        ka: float | None = None,
        kb: float | None = None,
        k_var: str = FV.K,
        ti_var: str = FV.TI,
    ) -> None:
        """
        Constructor.

        Parameters
        ----------
        k: float, optional
            The k value
        ka: float, optional
            The ka value in k = ka * TI + kb
        kb: float, optional
            The kb value in k = ka * TI + kb
        k_var: str
            The name of the k variable
        ti_var: str
            The name of the TI variable

        """
        super().__init__()
        self._k = k
        self._ka = ka
        self._kb = kb
        self.k_var = k_var
        self.ti_var = ti_var

        if k is not None and (ka is not None or kb is not None):
            raise ValueError("Got 'k' and also ('ka' or 'kb') as non-None parameters")
        elif k is None and kb is not None and (ka is None or ka == 0):
            raise ValueError(f"Got k={k}, ka={ka}, kb={kb}, use k={kb} instead")

        setattr(self, self.k_var, None)

    def repr(self) -> str:
        """
        Provides the representative string

        Returns
        -------
        s: str
            The representative string

        """
        if self._k is not None:
            s = f"{self.k_var}={self._k}"
        elif self._ka is not None or self._kb is not None:
            s = f"{self.k_var}={self._ka}*{self.ti_var}"
            if self._kb is not None and self._kb > 0:
                s += f"+{self._kb}"
        else:
            s = f"k_var={self.k_var}"
        return s

    @property
    def is_kTI(self) -> bool:
        """Flag for ka != 0"""
        return self._ka is not None and self._ka != 0

    @property
    def all_none(self) -> bool:
        """Flag for k=ka=kb=None"""
        return self._k is None and self._ka is None and self._kb is None

    @property
    def use_amb_ti(self) -> bool:
        """Flag for using ambient ti"""
        return self.ti_var in FV.amb2var

    def __call__(
        self,
        *args: Any,
        lookup_ti: str = "w",
        lookup_k: str = "sw",
        ti: np.ndarray | None = None,
        amb_ti: np.ndarray | None = None,
        selection: np.ndarray | tuple[slice] | None = None,
        **kwargs: Any,
    ) -> np.ndarray:
        """
        Gets the k value

        Parameters
        ----------
        args: tuple, optional
            Arguments for get_data
        lookup_ti: str
            The ti lookup order for get_data
        lookup_k: str
            The k lookup order for get_data
        ti: numpy.ndarray, optional
            ti data in the requested target shape,
            if known
        amb_ti: numpy.ndarray, optional
            Ambient ti data in the requested target shape,
            if known
        selection: array_like, optional
            Optional data selection for get_data
        kwargs: dict, optional
            Arguments for get_data

        Returns
        -------
        k: numpy.ndarray
            The k array as returned by get_data

        """
        setattr(self, self.k_var, self._k)
        assert len(args) > 0, f"{self.name}: Missing target argument for K call"
        target = cast(str, args[0])
        data_args = args[1:]
        if self._ka is not None or self._kb is not None:
            if self.ti_var == FV.TI and ti is not None:
                pass
            elif self.ti_var == FV.AMB_TI and amb_ti is not None:
                ti = amb_ti
            else:
                ti = cast(
                    np.ndarray,
                    self.get_data(  # type: ignore[call-overload]
                        self.ti_var, target, *data_args, lookup=lookup_ti, **kwargs
                    ),
                )
            kb = 0 if self._kb is None else self._kb
            setattr(self, self.k_var, self._ka * ti + kb)

        k = cast(
            np.ndarray,
            self.get_data(  # type: ignore[call-overload]
                self.k_var,
                target,
                *data_args,
                lookup=lookup_k,
                selection=selection,
                **kwargs,
            ),
        )
        setattr(self, self.k_var, None)
        return k
