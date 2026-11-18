from __future__ import annotations

import numpy as np
from abc import abstractmethod
from typing import TYPE_CHECKING, Any, cast

from foxes.utils import new_instance
from foxes.utils.factory import _reject_legacy_wake_coefficients
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
    """Wake growth parameter ``k = k_0 + k_1 * TI``.

    Supplying either coefficient selects the linear relation, with the other
    coefficient defaulting to zero. A zero multiplier does not require TI.
    Without coefficients, use uniform ``k`` or retrieve ``k_var`` from data.

    Raises
    ------
    ValueError
        If ``k`` and either linear coefficient are supplied together.
    DeprecationWarning
        If removed coefficient keywords are supplied. This is raised as an
        exception, not emitted as a warning.
    """

    def __init__(
        self,
        k: float | None = None,
        k_0: float | None = None,
        k_1: float | None = None,
        k_var: str = FV.K,
        ti_var: str = FV.TI,
        **kwargs: Any,
    ) -> None:
        """
        Parameters
        ----------
        k
            Uniform dimensionless wake growth parameter.
        k_0
            Dimensionless intercept in ``k = k_0 + k_1 * TI``.
        k_1
            Dimensionless TI multiplier in ``k = k_0 + k_1 * TI``.
        k_var
            Wake growth field to retrieve when no uniform values are supplied.
        ti_var
            Turbulence field, such as ``FV.TI`` or ``FV.AMB_TI``.
        kwargs
            Removed keywords are rejected with migration guidance; no extra
            configuration is accepted.
        """
        _reject_legacy_wake_coefficients(kwargs)
        if kwargs:
            raise TypeError(f"WakeK: Unexpected parameters {sorted(kwargs)}")
        super().__init__()
        self._k = k
        self._k_0 = k_0
        self._k_1 = k_1
        self.k_var = k_var
        self.ti_var = ti_var

        if k is not None and (k_0 is not None or k_1 is not None):
            raise ValueError("Got 'k' and also ('k_0' or 'k_1') as non-None parameters")

        setattr(self, self.k_var, k)
        self._intercept = 0.0 if k_0 is None else k_0

    def repr(self) -> str:
        """Describe the configured wake growth relation.

        Returns
        -------
        relation
            Uniform growth, the intercept/multiplier relation, or the field
            requested from data.
        """
        if self._k is not None:
            s = f"{self.k_var}={self._k}"
        elif self._k_0 is not None or self._k_1 is not None:
            k_0 = 0.0 if self._k_0 is None else self._k_0
            k_1 = 0.0 if self._k_1 is None else self._k_1
            s = f"{self.k_var}={k_0}+{k_1}*{self.ti_var}"
        else:
            s = f"k_var={self.k_var}"
        return s

    @property
    def is_kTI(self) -> bool:
        """Whether the TI multiplier is nonzero.

        Returns
        -------
        uses_ti
            Whether the linear relation requires turbulence intensity.
        """
        return self._k_1 is not None and self._k_1 != 0

    @property
    def all_none(self) -> bool:
        """Whether uniform growth and both coefficients are unspecified.

        Returns
        -------
        data_driven
            Whether to retrieve the growth field from data.
        """
        return self._k is None and self._k_0 is None and self._k_1 is None

    @property
    def use_amb_ti(self) -> bool:
        """Whether the turbulence field is ambient.

        Returns
        -------
        ambient
            Whether ``ti_var`` belongs to ``FV.amb2var``.
        """
        return self.ti_var in FV.amb2var

    def __call__(
        self,
        *args: Any,
        lookup_ti: str = "w",
        lookup_k: str = "sw",
        ti: np.ndarray | None = None,
        amb_ti: np.ndarray | None = None,
        selection: np.ndarray | tuple[slice, ...] | slice | None = None,
        **kwargs: Any,
    ) -> np.ndarray:
        """Evaluate dimensionless wake growth without changing model state.

        Parameters
        ----------
        args
            Target dimensions identifier for ``Model.get_data()``, followed
            by any positional data-lookup arguments. Supported targets are
            ``FC.STATE_TURBINE``, ``FC.STATE_TARGET`` and
            ``FC.STATE_TARGET_TPOINT``.
        lookup_ti
            Lookup order for ``ti_var`` when turbulence is not supplied.
        lookup_k
            Growth lookup order. ``s`` selects the configured uniform value or
            linear relation; other sources retrieve ``k_var`` from data.
        ti
            Local TI array in the full requested target shape. Used only when
            ``ti_var`` is ``FV.TI``.
        amb_ti
            Ambient TI array in the full requested target shape. Used only when
            ``ti_var`` is ``FV.AMB_TI``.
        selection
            Selection applied to the full target-shaped growth array.
        kwargs
            Data containers and validation options for ``Model.get_data()``.

        Returns
        -------
        k
            Dimensionless growth array aligned to the requested target and
            selection. Intercept-only growth is a read-only broadcast view
            when expansion is required.

        """
        assert len(args) > 0, f"{self.name}: Missing target argument for K call"
        target = cast(str, args[0])
        data_args = args[1:]
        if (self._k_0 is not None or self._k_1 is not None) and "s" in lookup_k:
            preceding = lookup_k.split("s", 1)[0]
            if preceding:
                configured = self.get_data(  # type: ignore[call-overload]
                    self.k_var,
                    target,
                    *data_args,
                    lookup=preceding,
                    selection=selection,
                    **{**kwargs, "accept_none": True},
                )
                if configured is not None:
                    return cast(np.ndarray, configured)
            k = cast(
                np.ndarray,
                self.get_data(  # type: ignore[call-overload]
                    "_intercept",
                    target,
                    *data_args,
                    lookup="s",
                    selection=selection,
                    **kwargs,
                ),
            )
            if self.is_kTI:
                assert self._k_1 is not None
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
                k = np.asarray(
                    k + self._k_1 * (ti if selection is None else ti[selection])
                )
            if not kwargs.get("accept_nan", True) and np.isnan(k).any():
                raise ValueError(f"Model '{self.name}': NaN values in '{self.k_var}'")
            return k

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
        return k
