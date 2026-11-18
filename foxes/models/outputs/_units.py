"""Scientific dataset units and output-only power conversion."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

import numpy as np

import foxes.constants as FC
import foxes.variables as FV

if TYPE_CHECKING:
    from xarray import Dataset

    from foxes.core import Algorithm
    from ._result_data import _Results


_POWER_VARIABLES = {FV.P, FV.CAP, FV.MAX_P}
_UNITS = {
    **dict.fromkeys(
        (FV.X, FV.Y, FV.H, FV.D, FV.TXYH, FC.TARGETS, FV.Z0, FV.MOL, "x", "y", "z"), "m"
    ),
    **dict.fromkeys(
        (
            FV.WS,
            FV.REWS,
            FV.REWS2,
            FV.REWS3,
            FV.UV,
            FV.U,
            FV.V,
            FV.WEIBULL_A,
            FV.MEAN_WS,
            FV.USTAR,
        ),
        "m/s",
    ),
    **dict.fromkeys((FV.WD, FV.YAW, FV.YAWM, FV.MAIN_WD, FV.LAT, FV.LON), "deg"),
    **dict.fromkeys(
        (
            FV.TI,
            FV.CT,
            FV.EFF,
            FV.CAPF,
            FV.WEIGHT,
            FV.OPERATING,
            FV.WEIBULL_k,
            FV.ORDER,
            FV.K,
            FV.K_0,
            FV.K_1,
            FV.SHEAR,
            FV.PA_ALPHA,
            FV.PA_BETA,
        ),
        "1",
    ),
    FV.TKE: "m2/s2",
    FV.RHO: "kg/m3",
    FV.T: "K",
    FV.p: "Pa",
    FV.YLD: "GWh/a",
    FV.P75: "GWh/a",
    FV.P90: "GWh/a",
    "frequency": "%",
}


def _power_unit(algo: Algorithm) -> str:
    """Resolve the smallest power unit used by the initialized turbine types."""
    turbine_types = algo.farm_controller.turbine_types
    if not turbine_types:
        raise ValueError("Output power units require initialized turbine types")

    units = {turbine_type.P_unit for turbine_type in turbine_types}
    unknown = units.difference(FC.P_UNITS)
    if unknown:
        raise ValueError(f"Unknown turbine-type power units: {sorted(unknown)}")

    return min(units, key=FC.P_UNITS.__getitem__)


def _normalize_farm_power(
    payload: _Results, algo: Algorithm, previous: Dataset | None = None
) -> None:
    """Convert farm power fields before reduction, without mutating input arrays.

    Previously collected fields use their declared unit rather than the native
    turbine units. This makes repeated output processing idempotent.
    """
    fields = payload["data_vars"]
    power_variables = [
        variable
        for variable in fields
        if FV.amb2var.get(variable, variable) in _POWER_VARIABLES
    ]
    if not power_variables:
        return

    power_unit = _power_unit(algo)
    turbine_types = algo.farm_controller.turbine_types
    assert turbine_types is not None
    turbines = payload["coords"][FC.TURBINE]
    native_factors = np.asarray(
        [FC.P_UNITS[turbine_types[int(index)].P_unit] for index in turbines]
    )

    # Declared units prevent repeated native-to-common conversion on reused data.
    for variable in power_variables:
        dims, values = fields[variable]
        source_unit = (
            previous[variable].attrs.get("units")
            if previous is not None and variable in previous
            else None
        )
        factor = _power_conversion_factor(
            variable, dims, native_factors, source_unit, power_unit
        )
        if np.any(factor != 1):
            fields[variable] = (dims, values * factor)

    payload["attrs"]["power_unit"] = power_unit


def _power_conversion_factor(
    variable: str,
    dims: tuple[str, ...],
    native_factors: np.ndarray,
    source_unit: str | None,
    power_unit: str,
) -> float | np.ndarray:
    """Resolve a declared-unit scalar or a broadcastable native-turbine factor.

    A field without a turbine axis must declare its source unit unless all
    selected turbines share the same native unit.
    """
    if source_unit is not None:
        if source_unit not in FC.P_UNITS:
            raise ValueError(
                f"Output field '{variable}': invalid power unit {source_unit!r}"
            )
        return FC.P_UNITS[source_unit] / FC.P_UNITS[power_unit]

    if FC.TURBINE in dims:
        shape = tuple(len(native_factors) if dim == FC.TURBINE else 1 for dim in dims)
        return (native_factors / FC.P_UNITS[power_unit]).reshape(shape)
    if len(set(native_factors)) == 1:
        return native_factors[0] / FC.P_UNITS[power_unit]

    raise ValueError(
        f"Output field '{variable}' needs a turbine axis or declared power unit"
    )


def _set_units(
    dataset: Dataset,
    power_unit: str | None = None,
    label_map: Mapping[str, str] | None = None,
    dimensionless: Sequence[str] = (),
) -> Dataset:
    """Attach known units in place, preserving custom attributes and names.

    Unknown fields and categorical/index coordinates receive no invented unit.
    Normalized quantities explicitly listed in ``dimensionless`` use ``1``.
    """
    original_names = {label: name for name, label in (label_map or {}).items()}
    power_unit = power_unit or dataset.attrs.get("power_unit")

    for name in dataset.variables:
        original = original_names.get(str(name), str(name))
        base = FV.amb2var.get(original, original)
        for prefix in ("bin_min_", "bin_max_"):
            if base.startswith(prefix):
                base = FV.amb2var.get(base[len(prefix) :], base[len(prefix) :])

        unit = power_unit if base in _POWER_VARIABLES else _UNITS.get(base)
        if original in dimensionless:
            unit = "1"
        if unit is not None:
            dataset[name].attrs["units"] = unit

    return dataset
