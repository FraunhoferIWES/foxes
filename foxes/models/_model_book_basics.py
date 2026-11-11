"""Default point, rotor, turbine and farm registry declarations."""

from __future__ import annotations

from math import sqrt
from typing import TYPE_CHECKING

import foxes.models as fm
import foxes.variables as FV
from foxes.utils import FDict
from foxes.utils.factory import _parse_coefficient

if TYPE_CHECKING:
    from .model_book import ModelBook


def _grid_side(n2: str | float) -> int:
    """Convert a square grid-point count to its integer side length.

    Parameters
    ----------
    n2
        Total number of grid points, possibly encoded in a factory-name token.

    Returns
    -------
    side
        Number of points along each side of the square grid.

    Raises
    ------
    Exception
        If the point count is not the square of an integer.
    """
    count = float(n2)
    side = int(sqrt(count))
    if side**2 != count:
        raise Exception(
            f"GridRotor factory: Value {count} is not the square of an integer"
        )
    return side


def _populate_point_models(book: ModelBook) -> None:
    """Populate the point-field conversion presets.

    Parameters
    ----------
    book
        Model book whose registry is populated in place. Each call creates
        fresh model instances and factory configuration for this book.
    """
    book.point_models = FDict(_name="point_models")
    book.point_models["tke2ti"] = fm.point_models.TKE2TI()
    book.point_models["ustar2ti"] = fm.point_models.Ustar2TI()


def _populate_rotor_models(book: ModelBook) -> None:
    """Populate rotor presets and point-count factories.

    Parameters
    ----------
    book
        Model book whose registry is populated in place. Each call creates
        fresh model instances and factory configuration for this book.
    """
    book.rotor_models = FDict(_name="rotor_models")
    book.rotor_models["centre"] = fm.rotor_models.CentreRotor()
    book.rotor_models["direct_mdata"] = fm.rotor_models.DirectMDataInfusion()

    book.rotor_models.add_factory(
        fm.rotor_models.GridRotor,
        "grid<n2>_raw",
        kwargs=dict(reduce=False),
        var2arg={"n2": "n"},
        n2=_grid_side,
        hints={"n2": "(Number of points in square grid)"},
    )
    book.rotor_models.add_factory(
        fm.rotor_models.GridRotor,
        "grid<n2>",
        kwargs=dict(reduce=True),
        var2arg={"n2": "n"},
        n2=_grid_side,
        hints={"n2": "(Number of points in square grid)"},
    )
    book.rotor_models.add_factory(
        fm.rotor_models.LevelRotor,
        "level<n>_raw",
        kwargs=dict(reduce=False),
        n=lambda x: int(x),
        hints={"n": "(Number of vertical levels)"},
    )
    book.rotor_models.add_factory(
        fm.rotor_models.LevelRotor,
        "level<n>",
        kwargs=dict(reduce=True),
        n=lambda x: int(x),
        hints={"n": "(Number of vertical levels)"},
    )


def _populate_turbine_types(book: ModelBook, Pct_file: str | None) -> None:
    """Populate built-in and optional file-backed turbine types.

    Parameters
    ----------
    book
        Model book whose registry is populated in place. Each call creates
        fresh model instances and factory configuration for this book.
    Pct_file
        Power/ct curve path, or ``None`` to omit the custom ``Pct`` preset.
    """
    book.turbine_types = FDict(_name="turbine_types")
    book.turbine_types["null_type"] = fm.turbine_types.NullType()
    book.turbine_types["NREL5MW"] = fm.turbine_types.PCtFile(
        "NREL-5MW-D126-H90.csv", rho=1.225
    )
    book.turbine_types["DTU10MW"] = fm.turbine_types.PCtFile(
        "DTU-10MW-D178d3-H119.csv", rho=1.225
    )
    book.turbine_types["IEA15MW"] = fm.turbine_types.PCtFile(
        "IEA-15MW-D240-H150.csv", rho=1.225
    )
    book.turbine_types["IWT7.5MW"] = fm.turbine_types.PCtFile(
        "IWT-7d5MW-D164-H100.csv", rho=1.225
    )
    if Pct_file is not None:
        book.turbine_types["Pct"] = fm.turbine_types.PCtFile(Pct_file)


def _populate_turbine_models(book: ModelBook) -> None:
    """Populate turbine operations and parameterized turbulence models.

    Parameters
    ----------
    book
        Model book whose registry is populated in place. Each call creates
        fresh model instances and factory configuration for this book.
    """
    book.turbine_models = FDict(
        _name="turbine_models",
        kTI=fm.turbine_models.kTI(),
        kTI_amb=fm.turbine_models.kTI(ti_var=FV.AMB_TI),
        thrust2ct=fm.turbine_models.Thrust2Ct(),
        PMask=fm.turbine_models.PowerMask(),
        yaw2yawm=fm.turbine_models.YAW2YAWM(),
        yawm2yaw=fm.turbine_models.YAWM2YAW(),
    )
    book.turbine_models.add_factory(
        fm.turbine_models.kTI,
        "kTI_amb_<kTI>_<kb>",
        kwargs=dict(ti_var=FV.AMB_TI),
        kTI=_parse_coefficient,
        kb=_parse_coefficient,
        hints={
            "kTI": "(Value, e.g. 04 for 0.4)",
            "kb": "(Value, e.g. 004 for 0.04)",
        },
    )
    book.turbine_models.add_factory(
        fm.turbine_models.kTI,
        "kTI_amb_<kTI>",
        kwargs=dict(ti_var=FV.AMB_TI),
        kTI=_parse_coefficient,
        hints={"kTI": "(Value, e.g. 04 for 0.4)"},
    )
    book.turbine_models.add_factory(
        fm.turbine_models.kTI,
        "kTI_<kTI>_<kb>",
        kTI=_parse_coefficient,
        kb=_parse_coefficient,
        hints={
            "kTI": "(Value, e.g. 04 for 0.4)",
            "kb": "(Value, e.g. 004 for 0.04)",
        },
    )
    book.turbine_models.add_factory(
        fm.turbine_models.kTI,
        "kTI_<kTI>",
        kTI=_parse_coefficient,
        hints={"kTI": "(Value, e.g. 02 for 0.2)"},
    )

    book.turbine_models["hubh_data"] = fm.turbine_models.RotorCentreCalc(
        {
            f"{FV.WD}_HH": FV.WD,
            f"{FV.WS}_HH": FV.WS,
            f"{FV.TI}_HH": FV.TI,
            f"{FV.RHO}_HH": FV.RHO,
        }
    )


def _populate_farm_models(book: ModelBook) -> None:
    """Populate farm-wide turbine operations and controllers.

    Parameters
    ----------
    book
        Model book whose registry is populated in place. Each call creates
        fresh model instances and factory configuration for this book.
    """
    book.farm_models = FDict(
        {
            f"farm_{mname}": fm.farm_models.Turbine2FarmModel(m)
            for mname, m in book.turbine_models.items()
        },
        _name="farm_models",
    )

    book.farm_controllers = FDict(
        _name="farm_controllers",
        basic_ctrl=fm.farm_controllers.BasicFarmController(),
    )
