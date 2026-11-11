"""Chunk-local turbine performance metrics without state aggregation."""

from __future__ import annotations
# mypy: disable-error-code=override

from typing import TYPE_CHECKING, Any, Literal, TypeAlias

import numpy as np

from foxes.config import config
from foxes.core import TurbineModel
import foxes.constants as FC
import foxes.variables as FV

if TYPE_CHECKING:
    from foxes.core import Algorithm
    from foxes.core.data import FData, MData


_Selection: TypeAlias = slice | np.ndarray | tuple[slice | np.ndarray | int, ...]
_YIELD_FACTORS = {FV.YLD: 0.0, FV.P75: 0.675, FV.P90: 1.282}
_BASE_VARIABLES = {FV.CAP, FV.EFF, FV.CAPF, *_YIELD_FACTORS}
_SUPPORTED_VARIABLES = _BASE_VARIABLES | {
    FV.var2amb[v] for v in _BASE_VARIABLES if v in FV.var2amb
}


class TurbineEvaluation(TurbineModel):
    """Performance evaluation at each state and turbine.

    Apply after the turbine type and any power-changing models. Yield is the
    annualized value of each state's power, not a weighted state contribution.
    No reduction over states or turbines is performed.

    Attributes
    ----------
    variables
        Configured output variable names, copied at construction.
    power_uncert
        Relative power uncertainty used for P75/P90 yields.
    """

    BASE_VARIABLES = _BASE_VARIABLES
    SUPPORTED_VARIABLES = _SUPPORTED_VARIABLES

    def __init__(
        self,
        variables: list[str] | None = None,
        name: str = "TurbineEvaluation",
        *,
        power_uncert: float | None = None,
        **kwargs: Any,
    ) -> None:
        """Configure the turbine performance metrics.

        Parameters
        ----------
        variables
            Requested ``FV.CAP``, ``FV.EFF``, ``FV.CAPF``, ``FV.YLD``,
            ``FV.P75``, ``FV.P90``, or ambient counterparts of capacity factor
            and yields. Merged with
            variables supplied to ``calculate()``; ``None`` requests none.
        name
            The model name.
        power_uncert
            Finite, nonnegative relative uncertainty; ``0.1`` means 10 percent.
            Required when requesting P75/P90, including ambient variants.
        kwargs
            Additional keyword arguments passed to the base class.

        Raises
        ------
        ValueError
            If a variable is unsupported, uncertainty is invalid, or a
            requested P75/P90 metric has no uncertainty.
        """
        # Own the configuration instead of retaining the caller's mutable list.
        super().__init__(**kwargs)
        self.name = name
        self.variables = [] if variables is None else list(dict.fromkeys(variables))
        self.power_uncert = power_uncert

        # Probability yields must not infer an uncertainty from missing input.
        if power_uncert is not None and (
            not np.isfinite(power_uncert) or power_uncert < 0.0
        ):
            raise ValueError(
                f"{self.name}: power_uncert must be finite and nonnegative"
            )
        self._validate_variables(self.variables)

    def _validate_variables(self, variables: list[str]) -> None:
        unknown = set(variables) - _SUPPORTED_VARIABLES
        if unknown:
            raise ValueError(
                f"{self.name}: Unsupported evaluation variables {sorted(unknown)}"
            )

        quantiles = {FV.P75, FV.P90, FV.AMB_P75, FV.AMB_P90}
        if self.power_uncert is None and quantiles.intersection(variables):
            raise ValueError(f"{self.name}: power_uncert is required for P75/P90")

    def output_farm_vars(self, algo: Algorithm) -> list[str]:
        """Return the configured farm output variables.

        Parameters
        ----------
        algo
            The calculation algorithm.

        Returns
        -------
        output_vars
            Configured output names. Variables supplied only to
            ``calculate()`` are not declared here.
        """
        return list(self.variables)

    def _turbine_values(
        self,
        algo: Algorithm,
        order: np.ndarray,
        attribute: Literal["P_nominal", "P_unit"],
    ) -> np.ndarray:
        """Select turbine-type metadata in the current calculation order."""
        # Turbine types are stored in original farm order, not downwind order.
        turbine_types = algo.farm_controller.turbine_types
        if turbine_types is None:
            raise ValueError(
                f"{self.name}: Farm controller turbine types are unavailable"
            )
        if attribute == "P_nominal":
            values = np.asarray(
                [t.P_nominal for t in turbine_types], dtype=config.dtype_double
            )
        else:
            try:
                values = np.asarray(
                    [FC.P_UNITS[t.P_unit] for t in turbine_types],
                    dtype=config.dtype_double,
                )
            except KeyError as exc:
                raise ValueError(
                    f"{self.name}: Unsupported turbine power unit {exc.args[0]!r}"
                ) from exc

        # Validate selected turbines only; other turbines may lack nominal power.
        selected = values[order]
        if np.any(~np.isfinite(selected) | (selected <= 0.0)):
            raise ValueError(f"{self.name}: {attribute} must be finite and positive")
        return selected

    def _evaluate(
        self,
        variable: str,
        fdata: FData,
        st_sel: _Selection,
        capacity: np.ndarray | None,
        power_units: np.ndarray | None,
    ) -> np.ndarray:
        """Calculate a validated metric for selected state-turbine entries."""
        # Ambient and waked metrics share formulas but use different power fields.
        base = FV.amb2var.get(variable, variable)
        if base == FV.CAP:
            assert capacity is not None
            return capacity

        power_var = FV.AMB_P if variable in FV.amb2var else FV.P
        power = self.get_data(
            power_var, FC.STATE_TURBINE, lookup="f", fdata=fdata, selection=st_sel
        )

        # Efficiency intentionally retains raw division, including NaNs and infs.
        if base == FV.EFF:
            ambient = self.get_data(
                FV.AMB_P, FC.STATE_TURBINE, lookup="f", fdata=fdata, selection=st_sel
            )
            return power / ambient

        if base == FV.CAPF:
            assert capacity is not None
            return power / capacity

        # Convert the turbine's power unit before annualizing, without weights.
        assert power_units is not None
        factor = _YIELD_FACTORS[base]
        if factor:
            assert self.power_uncert is not None
            factor = 1.0 - factor * self.power_uncert
        else:
            factor = 1.0
        return power * power_units * (8760.0 / 1.0e9) * factor

    def calculate(
        self,
        algo: Algorithm,
        mdata: MData,
        fdata: FData,
        st_sel: _Selection = slice(None),
        variables: list[str] | None = None,
    ) -> dict[str, np.ndarray]:
        """Calculate performance metrics for one farm-data chunk.

        Parameters
        ----------
        algo
            Algorithm supplying turbine types, nominal powers and power units.
        mdata
            The model data.
        fdata
            Farm data with dimensions ``(FC.STATE, FC.TURBINE)`` and power
            fields ``FV.P`` and/or ``FV.AMB_P`` as required by the metrics.
            ``FV.ORDER``, when present, maps calculation columns to original
            farm indices. Output arrays are added or updated in place.
        st_sel
            NumPy selection in calculation order: a state slice, boolean
            state-turbine mask, or tuple such as ``(:, turbine_index)``.
        variables
            Additional output names, merged with configured variables without
            changing the model's declared outputs.

        Returns
        -------
        results
            Requested variable names mapped to full ``fdata`` arrays with
            dimensions ``(FC.STATE, FC.TURBINE)``. Unselected entries retain
            their previous values, or NaN for newly added outputs.

        Raises
        ------
        ValueError
            If a metric or uncertainty is invalid, required turbine types are
            unavailable, selected nominal powers are nonpositive or nonfinite,
            a turbine power unit is unsupported, or an input power field is
            missing.

        Notes
        -----
        ``CAP`` is nominal turbine power in the turbine type's power unit.
        ``CAPF = P / CAP``.
        ``EFF = P / AMB_P`` is not clipped or guarded against zero division.
        Yield is ``P * P_unit_W * 8760 / 1e9`` in GWh/a. Apply state weights
        only during later aggregation. ``P75 = YLD * (1 - 0.675 * power_uncert)``
        and ``P90 = YLD * (1 - 1.282 * power_uncert)``. Ambient metrics use
        ``AMB_P`` in the same formulas. Neither inputs nor model state are
        modified; only requested ``fdata`` outputs are updated.
        """
        # Resolve call-local outputs without changing the configured model.
        variables = list(dict.fromkeys(self.variables + (variables or [])))
        self._validate_variables(variables)
        bases = {FV.amb2var.get(v, v) for v in variables}
        n_states, n_turbines = fdata.n_states, fdata.n_turbines
        assert n_states is not None and n_turbines is not None
        shape = (n_states, n_turbines)

        # Align metadata with the same selection used for the power arrays.
        capacity = power_units = None
        if bases.intersection({FV.CAP, FV.CAPF, *_YIELD_FACTORS}):
            if FV.ORDER in fdata:
                order = fdata[FV.ORDER][st_sel]
            else:
                order = np.broadcast_to(np.arange(n_turbines), shape)[st_sel]
            if bases.intersection({FV.CAP, FV.CAPF}):
                capacity = self._turbine_values(algo, order, "P_nominal")
            if bases.intersection(_YIELD_FACTORS):
                power_units = self._turbine_values(algo, order, "P_unit")

        # Write through full farm arrays so unselected entries survive.
        results: dict[str, np.ndarray] = {}
        for variable in variables:
            values = self._evaluate(variable, fdata, st_sel, capacity, power_units)
            if variable not in fdata:
                fdata.add(
                    variable,
                    np.full(shape, np.nan, dtype=config.dtype_double),
                    (FC.STATE, FC.TURBINE),
                )
            fdata[variable][st_sel] = values
            results[variable] = fdata[variable]
        return results
