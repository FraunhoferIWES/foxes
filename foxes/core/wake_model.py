from __future__ import annotations

import numpy as np
from abc import abstractmethod
from typing import TYPE_CHECKING, Any, cast

from foxes.utils import new_instance
import foxes.variables as FV

from .model import Model

if TYPE_CHECKING:
    from foxes.core.algorithm import Algorithm
    from foxes.core.data import FData, MData, TData
    from foxes.core.model import LoadedData


class WakeModel(Model):
    """
    Abstract base class for wake models.


    """

    def __init__(self) -> None:
        """
        Constructor.
        """
        super().__init__()
        self._has_uv = False

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
        return True

    @property
    def has_uv(self) -> bool:
        """
        This model uses wind vector data

        Returns
        -------
        has_uv
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
    def waked_variables(self) -> list[str]:
        """
        Returns a list of variable names that are affected by this wake model.

        Returns
        -------
        waked_variables
            A list of variable names affected by this wake model.

        """
        pass

    @property
    def affects_ws(self) -> bool:
        """
        Flag for wind speed wake models

        Returns
        -------
        dws
            If True, this model affects wind speed

        """
        return FV.WS in self.waked_variables()

    @abstractmethod
    def new_wake_deltas(
        self,
        algo: Algorithm,
        mdata: MData,
        fdata: FData,
        tdata: TData,
    ) -> dict[str, np.ndarray]:
        """
        Create new empty wake delta arrays.

        Parameters
        ----------
        algo
            The calculation algorithm.
        mdata
            The model data.
        fdata
            The farm data.
        tdata
            The target point data.

        Returns
        -------
        wake_deltas
            A dictionary keyed by variable name. Values are zero-filled wake
            deltas with shape ``(n_states, n_targets, n_tpoints, ...)``.

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
        Modify wake deltas at target points using contributions from wake source
        turbines.

        Parameters
        ----------
        algo
            The calculation algorithm.
        mdata
            The model data.
        fdata
            The farm data.
        tdata
            The target point data.
        downwind_index
            The index of the wake-causing turbine in the downwind order.
        wake_coos
            Wake-frame coordinates of the evaluation points with shape
            ``(n_states, n_targets, n_tpoints, 3)``.
        wake_deltas
            The wake deltas. Keys are variable names and values are arrays with
            shape ``(n_states, n_targets, n_tpoints, ...)``.

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
        pass

    @classmethod
    def new(cls, wmodel_type: str, *args: Any, **kwargs: Any) -> WakeModel:
        """
        Run-time wake model factory.

        Parameters
        ----------
        wmodel_type
            The selected derived class name
        args
            Additional parameters for constructor
        kwargs
            Additional parameters for constructor

        """
        return cast(WakeModel, new_instance(cls, wmodel_type, *args, **kwargs))


class WakeK(Model):
    """
    Handler for the wake growth parameter k

    Attributes
    ----------
    k_var
        The name of the k variable
    ti_var
        The name of the TI variable


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
        k
            The k value
        ka
            The ka value in k = ka * TI + kb
        kb
            The kb value in k = ka * TI + kb
        k_var
            The name of the k variable
        ti_var
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
        s
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
        args
            Arguments for get_data
        lookup_ti
            The ti lookup order for get_data
        lookup_k
            The k lookup order for get_data
        ti
            ti data in the requested target shape,
            if known
        amb_ti
            Ambient ti data in the requested target shape,
            if known
        selection
            Optional data selection for get_data
        kwargs
            Arguments for get_data

        Returns
        -------
        k
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
