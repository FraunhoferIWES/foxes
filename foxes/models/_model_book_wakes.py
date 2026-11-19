"""Default wake, superposition and ground registry declarations."""

from __future__ import annotations

from math import sqrt
from typing import TYPE_CHECKING, Any

import foxes.models as fm
import foxes.variables as FV
from foxes.utils import FDict
from foxes.utils.factory import _parse_coefficient

from ._model_book_basics import _grid_side

if TYPE_CHECKING:
    from foxes.core import WakeModel, WakeSuperposition

    from .model_book import ModelBook


def _minutes(token: str) -> float:
    """Convert a seconds/minutes factory token to minutes.

    Parameters
    ----------
    token
        Numeric duration followed by ``"s"`` or ``"min"``.

    Returns
    -------
    minutes
        Duration in minutes.

    Raises
    ------
    NotImplementedError
        If the token uses an unsupported duration suffix.
    """
    if token[-1] == "s":
        return float(token[:-1]) / 60
    if token[-3:] == "min":
        return float(token[:-3])
    raise NotImplementedError(f"Cannot translate '{token}' into minutes")


def _add_ws_variants(
    book: ModelBook,
    name: str,
    superposition: type[WakeSuperposition | fm.wake_superpositions.WindVectorLinear],
    kwargs: dict[str, Any] | None = None,
    prefix: str = "ws",
    limits: bool = True,
) -> None:
    """Register source/target scaling and lower-limit factory variants.

    Parameters
    ----------
    book
        Model book whose superposition factory registry is extended in place.
    name
        Reduction token in the model-book factory name.
    superposition
        Superposition class instantiated by the factory.
    kwargs
        Shared constructor parameters; ``None`` supplies no extra parameters.
    prefix
        Physical-field prefix, such as ``"ws"`` or ``"vector"``.
    limits
        Register lower-limit variants for classes accepting ``lim_low``.
    """
    base_kwargs = {} if kwargs is None else kwargs
    variants: dict[str, dict[str, bool | float]] = {
        "": {},
        "_amb": {"scale_amb": True},
        "_target": {"scale_target": True},
        "_amb_target": {"scale_amb": True, "scale_target": True},
        "_loc": {"scale_amb": True, "scale_target": True},
    }
    if limits:
        variants.update(
            {
                f"{suffix}_lim": {**options, "lim_low": 1e-4}
                for suffix, options in variants.items()
            }
        )
    book.wake_superpositions.add_factory(
        superposition,
        f"{prefix}{'_' if name else ''}{name}<suffix>",
        kwargs=base_kwargs,
        suffix=variants,
        var2arg={"suffix": None},
    )


def _populate_partial_wakes(book: ModelBook) -> None:
    """Populate partial-wake integrations and quadrature factories.

    Parameters
    ----------
    book
        Model book whose registry is populated in place. Each call creates
        fresh model instances and factory configuration for this book.
    """
    book.partial_wakes = FDict(
        _name="partial_wakes",
        rotor_points=fm.partial_wakes.RotorPoints(),
        top_hat=fm.partial_wakes.PartialTopHat(),
        centre=fm.partial_wakes.PartialCentre(),
        gaussian=fm.partial_wakes.PartialGaussian(),
        gaussian_lookup=fm.partial_wakes.PartialGaussianLookup(),
    )
    book.partial_wakes.add_factory(
        fm.partial_wakes.PartialAxiwake,
        "axiwake<n>",
        n=lambda x: int(x),
        hints={"n": "(Number of evaluation points)"},
    )
    book.partial_wakes.add_factory(
        fm.partial_wakes.PartialGrid,
        "grid<n2>",
        var2arg={"n2": "n"},
        n2=_grid_side,
        hints={"n2": "(Number of points in square grid)"},
    )


def _populate_wake_deflections(book: ModelBook) -> None:
    """Populate wake-deflection presets and path-integration factories.

    Parameters
    ----------
    book
        Model book whose registry is populated in place. Each call creates
        fresh model instances and factory configuration for this book.
    """
    book.wake_deflections = FDict(
        _name="wake_deflections",
        no_deflection=fm.wake_deflections.NoDeflection(),
        Bastankhah2016=fm.wake_deflections.Bastankhah2016Deflection(),
        Jimenez=fm.wake_deflections.JimenezDeflection(rotate=True),
        JimenezProj=fm.wake_deflections.JimenezDeflection(rotate=False),
        JimenezPath=fm.wake_deflections.JimenezDeflection(rotate=None),
    )

    book.wake_deflections.add_factory(
        fm.wake_deflections.JimenezDeflection,
        "Jimenez_b<beta>",
        beta=_parse_coefficient,
        hints={"beta": "(The Jimenez beta coefficient, e.g. 01 for 0.1)"},
        kwargs=dict(rotate=True),
    )
    book.wake_deflections.add_factory(
        fm.wake_deflections.JimenezDeflection,
        "Jimenez_b<beta>_dx<dx>",
        beta=_parse_coefficient,
        dx=lambda dx: float(dx),
        var2arg={"dx": "step_x"},
        hints={
            "beta": "(The Jimenez beta coefficient, e.g. 01 for 0.1)",
            "dx": "(The step size in m for integration along wake path, e.g. 10)",
        },
        kwargs=dict(rotate=True),
    )
    book.wake_deflections.add_factory(
        fm.wake_deflections.JimenezDeflection,
        "JimenezProj_b<beta>",
        beta=_parse_coefficient,
        hints={"beta": "(The Jimenez beta coefficient, e.g. 01 for 0.1)"},
        kwargs=dict(rotate=False),
    )
    book.wake_deflections.add_factory(
        fm.wake_deflections.JimenezDeflection,
        "JimenezProj_b<beta>_dx<dx>",
        beta=_parse_coefficient,
        dx=lambda dx: float(dx),
        var2arg={"dx": "step_x"},
        hints={
            "beta": "(The Jimenez beta coefficient, e.g. 01 for 0.1)",
            "dx": "(The step size in m for integration along wake path, e.g. 10)",
        },
        kwargs=dict(rotate=False),
    )
    book.wake_deflections.add_factory(
        fm.wake_deflections.JimenezDeflection,
        "JimenezPath_b<beta>",
        beta=_parse_coefficient,
        hints={"beta": "(The Jimenez beta coefficient, e.g. 01 for 0.1)"},
        kwargs=dict(rotate=None),
    )
    book.wake_deflections.add_factory(
        fm.wake_deflections.JimenezDeflection,
        "JimenezPath_b<beta>_dx<dx>",
        beta=_parse_coefficient,
        dx=lambda dx: float(dx),
        var2arg={"dx": "step_x"},
        hints={
            "beta": "(The Jimenez beta coefficient, e.g. 01 for 0.1)",
            "dx": "(The step size in m for integration along wake path, e.g. 10)",
        },
        kwargs=dict(rotate=None),
    )


def _populate_wake_frames(book: ModelBook) -> None:
    """Populate spatial and time-dependent wake-frame factories.

    Parameters
    ----------
    book
        Model book whose registry is populated in place. Each call creates
        fresh model instances and factory configuration for this book.
    """
    book.wake_frames = FDict(
        _name="wake_frames",
        rotor_wd=fm.wake_frames.RotorWD(),
        rotor_wd_farmo=fm.wake_frames.FarmOrder(),
    )
    book.wake_frames.add_factory(
        fm.wake_frames.Streamlines2D,
        "streamlines_<step>",
        step=lambda x: float(x),
        hints={"step": "(Step size in m)"},
    )
    book.wake_frames.add_factory(
        fm.wake_frames.Streamlines2D,
        "streamlines_<step>_c<chunksize>",
        step=lambda x: float(x),
        chunksize=lambda x: int(x),
        var2arg={"chunksize": "chunksize_steps"},
        hints={
            "step": "(Step size in m)",
            "chunksize": "(Chunk size in steps)",
        },
    )

    book.wake_frames["timelines"] = fm.wake_frames.Timelines()
    book.wake_frames["dyn_wakes"] = fm.wake_frames.DynamicWakes()
    book.wake_frames["seq_dyn_wakes"] = fm.wake_frames.SeqDynamicWakes()

    book.wake_frames.add_factory(
        fm.wake_frames.Timelines,
        "timelines_<dt>",
        dt=_minutes,
        var2arg={"dt": "dt_min"},
        hints={"dt": "(Time step, e.g '10s', '1min' etc.)"},
    )
    book.wake_frames.add_factory(
        fm.wake_frames.SeqDynamicWakes,
        "seq_dyn_wakes_<dt>",
        dt=_minutes,
        var2arg={"dt": "dt_min"},
        hints={"dt": "(Time step, e.g '10s', '1min' etc.)"},
    )


def _populate_superpositions(book: ModelBook) -> None:
    """Populate wind-speed and turbulence superposition variants.

    Parameters
    ----------
    book
        Model book whose registry is populated in place. Each call creates
        fresh model instances and factory configuration for this book.
    """
    book.wake_superpositions = FDict(
        _name="wake_superpositions",
        ws_product=fm.wake_superpositions.WSProduct(),
        ws_product_lim=fm.wake_superpositions.WSProduct(lim_low=1e-4),
        ti_linear=fm.wake_superpositions.TILinear(superp_to_amb="quadratic"),
        ti_quadratic=fm.wake_superpositions.TIQuadratic(superp_to_amb="quadratic"),
        ti_cubic=fm.wake_superpositions.TIPow(pow=3, superp_to_amb="quadratic"),
        ti_quartic=fm.wake_superpositions.TIPow(pow=4, superp_to_amb="quadratic"),
        ti_max=fm.wake_superpositions.TIMax(superp_to_amb="quadratic"),
    )

    _add_ws_variants(book, "linear", fm.wake_superpositions.WSLinear)
    _add_ws_variants(book, "quadratic", fm.wake_superpositions.WSQuadratic)
    _add_ws_variants(book, "cubic", fm.wake_superpositions.WSPow, {"pow": 3})
    _add_ws_variants(book, "quartic", fm.wake_superpositions.WSPow, {"pow": 4})
    _add_ws_variants(book, "max", fm.wake_superpositions.WSMax)
    _add_ws_variants(
        book,
        "",
        fm.wake_superpositions.WindVectorLinear,
        prefix="vector",
        limits=False,
    )


def _populate_axial_induction(book: ModelBook) -> None:
    """Populate axial-induction presets.

    Parameters
    ----------
    book
        Model book whose registry is populated in place. Each call creates
        fresh model instances and factory configuration for this book.
    """
    book.axial_induction = FDict(_name="induction_models")
    book.axial_induction["Betz"] = fm.axial_induction.BetzAxialInduction()
    book.axial_induction["Madsen"] = fm.axial_induction.MadsenAxialInduction()


def _add_wind_factory(
    book: ModelBook,
    model_type: type[WakeModel],
    template: str,
    kwargs: dict[str, Any] | None = None,
    hints: dict[str, str] | None = None,
    **variables: Any,
) -> None:
    """Register a wind-wake factory with consistent superposition tokens.

    Parameters
    ----------
    book
        Model book whose wake-model factories are extended in place.
    model_type
        Wake class instantiated by the factory.
    template
        Parameterized model-book name.
    kwargs
        Shared constructor parameters, or ``None`` for no extra parameters.
    hints
        Descriptions of additional factory tokens.
    variables
        Converters for tokens other than the shared superposition token.
    """
    register = (
        book.wake_models.add_k_factory
        if "_[wake_k]" in template
        else book.wake_models.add_factory
    )
    register(
        model_type,
        template,
        kwargs=kwargs,
        superposition=lambda token: (
            f"ws_{token}" if f"ws_{token}" in book.wake_superpositions else token
        ),
        hints={
            "superposition": "(Superposition, e.g. linear for ws_linear, or vector)",
            **({} if hints is None else hints),
        },
        **variables,
    )


def _populate_reference_wakes(book: ModelBook) -> None:
    """Populate reference Gaussian and top-hat wake factories.

    Parameters
    ----------
    book
        Model book whose registry is populated in place. Each call creates
        fresh model instances and factory configuration for this book.
    """
    book.wake_models = FDict(_name="wake_models")

    # Reference for the IEA37Gaussian wake deficit model:
    # https://github.com/byuflowlab/iea37-wflo-casestudies/blob/master/cs1-2/iea37-wakemodel.pdf
    book.wake_models["IEA37Gaussian"] = fm.wake_models.wind.Bastankhah2014(
        superposition="ws_quadratic",
        k=0.0324555,
        sbeta=1 / sqrt(8),
    )
    _add_wind_factory(
        book,
        fm.wake_models.wind.Bastankhah2014,
        "IEA37Gaussian_<superposition>_[wake_k]",
        kwargs=dict(sbeta=1 / sqrt(8)),
    )

    _add_wind_factory(
        book,
        fm.wake_models.wind.JensenWake,
        "Jensen_<superposition>_[wake_k]",
        kwargs=dict(induction="Betz"),
    )

    _add_wind_factory(
        book,
        fm.wake_models.wind.JensenTurbOParkWake,
        "JensenTurbOPark_<superposition>_[wake_k]",
        kwargs=dict(induction="Betz"),
    )

    book.wake_models["Bastankhah2014"] = fm.wake_models.wind.Bastankhah2014(
        superposition="ws_linear",
        k=0.04,
        induction="Madsen",
    )


def _populate_gaussian_wakes(book: ModelBook) -> None:
    """Populate Gaussian wake presets and induction variants.

    Parameters
    ----------
    book
        Model book whose registry is populated in place. Each call creates
        fresh model instances and factory configuration for this book.
    """
    book.wake_models["Niayifar"] = fm.wake_models.wind.Bastankhah2014(
        superposition="ws_linear",
        k_1=0.3837,
        k_0=0.003678,
        ti_var=FV.TI,
        induction="Madsen",
    )
    _add_wind_factory(
        book,
        fm.wake_models.wind.Bastankhah2014,
        "Bastankhah2014_<superposition>_[wake_k]",
        kwargs=dict(sbeta_factor=0.2, induction="Madsen"),
    )
    _add_wind_factory(
        book,
        fm.wake_models.wind.Bastankhah2014,
        "Bastankhah2014B_<superposition>_[wake_k]",
        kwargs=dict(sbeta_factor=0.2, induction="Betz"),
    )
    _add_wind_factory(
        book,
        fm.wake_models.wind.Bastankhah2014,
        "Bastankhah025_<superposition>_[wake_k]",
        kwargs=dict(sbeta_factor=0.25, induction="Madsen"),
    )
    _add_wind_factory(
        book,
        fm.wake_models.wind.Bastankhah2014,
        "Bastankhah025B_<superposition>_[wake_k]",
        kwargs=dict(sbeta_factor=0.25, induction="Betz"),
    )

    _add_wind_factory(
        book,
        fm.wake_models.wind.Bastankhah2016,
        "Bastankhah2016_<superposition>_[wake_k]",
        kwargs=dict(induction="Madsen"),
    )
    _add_wind_factory(
        book,
        fm.wake_models.wind.Bastankhah2016,
        "Bastankhah2016B_<superposition>_[wake_k]",
        kwargs=dict(induction="Betz"),
    )


def _populate_park_wakes(book: ModelBook) -> None:
    """Populate park-wake presets and integration factories.

    Parameters
    ----------
    book
        Model book whose registry is populated in place. Each call creates
        fresh model instances and factory configuration for this book.
    """
    book.wake_models["TurbOPark"] = fm.wake_models.wind.TurbOParkWake(
        superposition="ws_quadratic_amb_target",
        k_1=0.04,
        ti_var=FV.AMB_TI,
        induction="Betz",
    )

    book.wake_models["TurbOPark0045"] = fm.wake_models.wind.TurbOParkWake(
        superposition="ws_quadratic_amb_target",
        k_1=0.045,
        ti_var=FV.AMB_TI,
        induction="Betz",
    )

    book.wake_models["JensenTurbOPark"] = fm.wake_models.wind.JensenTurbOParkWake(
        superposition="ws_quadratic_amb_target",
        k_1=0.6,
        ti_var=FV.AMB_TI,
        induction="Betz",
    )

    _add_wind_factory(
        book,
        fm.wake_models.wind.TurbOParkWake,
        "TurbOPark_<superposition>_[wake_k]",
        kwargs=dict(induction="Betz"),
    )
    _add_wind_factory(
        book,
        fm.wake_models.wind.TurbOParkWake,
        "TurbOParkB_<superposition>_[wake_k]",
        kwargs=dict(induction="Betz"),
    )

    _add_wind_factory(
        book,
        fm.wake_models.wind.TurbOParkWakeIX,
        "TurbOParkIX_<superposition>_k<k>_dx<dx>",
        k=_parse_coefficient,
        dx=float,
        hints={
            "k": "(Value, e.g. 004 for 0.04)",
            "dx": "(Integration step in m)",
        },
    )
    _add_wind_factory(
        book,
        fm.wake_models.wind.TurbOParkWakeIX,
        "TurbOParkIX_<superposition>_dx<dx>",
        dx=lambda x: float(x),
        hints={"dx": "(Integration step in m)"},
    )

    book.wake_models.add_k_factory(
        fm.wake_models.ti.CrespoHernandezTIWake,
        "CrespoHernandez_<superposition>_[wake_k]",
        kwargs=dict(use_ambti=False),
        superposition=lambda s: f"ti_{s}",
        hints={"superposition": "(Superposition, e.g. linear for ti_linear)"},
    )


def _populate_turbulence_wakes(book: ModelBook) -> None:
    """Populate wake-added turbulence factories.

    Parameters
    ----------
    book
        Model book whose registry is populated in place. Each call creates
        fresh model instances and factory configuration for this book.
    """
    book.wake_models.add_factory(
        fm.wake_models.ti.IECTIWake,
        "IECTI2019_<superposition>",
        kwargs=dict(iec_type="2019"),
        superposition=lambda s: f"ti_{s}",
        hints={"superposition": "(Superposition, e.g. linear for ti_linear)"},
    )
    book.wake_models.add_k_factory(
        fm.wake_models.ti.IECTIWake,
        "IECTI2019k_<superposition>_[wake_k]",
        kwargs=dict(iec_type="2019", opening_angle=None),
        superposition=lambda s: f"ti_{s}",
        hints={"superposition": "(Superposition, e.g. linear for ti_linear)"},
    )

    book.wake_models.add_factory(
        fm.wake_models.ti.IECTIWake,
        "IECTI2005_<superposition>",
        kwargs=dict(iec_type="2005"),
        superposition=lambda s: f"ti_{s}",
        hints={"superposition": "(Superposition, e.g. linear for ti_linear)"},
    )
    book.wake_models.add_k_factory(
        fm.wake_models.ti.IECTIWake,
        "IECTI2005k_<superposition>_[wake_k]",
        kwargs=dict(iec_type="2005", opening_angle=None),
        superposition=lambda s: f"ti_{s}",
        hints={"superposition": "(Superposition, e.g. linear for ti_linear)"},
    )

    book.wake_models["RHB"] = fm.wake_models.induction.RankineHalfBody()

    book.wake_models["VortexSheet"] = fm.wake_models.induction.VortexSheet()
    book.wake_models.add_factory(
        fm.wake_models.induction.VortexSheet,
        "VortexSheet_<superposition>",
        superposition=lambda s: (
            f"ws_{s}" if f"ws_{s}" in book.wake_superpositions else s
        ),
        hints={
            "superposition": "(Superposition, e.g. linear for ws_linear, or vector)"
        },
    )


def _populate_induction_wakes(book: ModelBook) -> None:
    """Populate turbine-induction model presets and factories.

    Parameters
    ----------
    book
        Model book whose registry is populated in place. Each call creates
        fresh model instances and factory configuration for this book.
    """
    book.wake_models["Rathmann"] = fm.wake_models.induction.Rathmann()
    book.wake_models.add_factory(
        fm.wake_models.induction.Rathmann,
        "Rathmann_<superposition>",
        superposition=lambda s: (
            f"ws_{s}" if f"ws_{s}" in book.wake_superpositions else s
        ),
        hints={
            "superposition": "(Superposition, e.g. linear for ws_linear, or vector)"
        },
    )

    book.wake_models["SelfSimilar"] = fm.wake_models.induction.SelfSimilar()
    book.wake_models["SelfSimilar2020"] = fm.wake_models.induction.SelfSimilar2020()
    book.wake_models.add_factory(
        fm.wake_models.induction.SelfSimilar,
        "SelfSimilar_<superposition>",
        superposition=lambda s: (
            f"ws_{s}" if f"ws_{s}" in book.wake_superpositions else s
        ),
        hints={
            "superposition": "(Superposition, e.g. linear for ws_linear, or vector)"
        },
    )
    book.wake_models.add_factory(
        fm.wake_models.induction.SelfSimilar2020,
        "SelfSimilar2020_<superposition>",
        superposition=lambda s: (
            f"ws_{s}" if f"ws_{s}" in book.wake_superpositions else s
        ),
        hints={
            "superposition": "(Superposition, e.g. linear for ws_linear, or vector)"
        },
    )


def _populate_ground_models(book: ModelBook) -> None:
    """Populate ground-reflection presets and height factories.

    Parameters
    ----------
    book
        Model book whose registry is populated in place. Each call creates
        fresh model instances and factory configuration for this book.
    """
    book.ground_models = FDict(_name="ground_models")
    book.ground_models["no_ground"] = fm.ground_models.NoGround()
    book.ground_models["ground_mirror"] = fm.ground_models.GroundMirror()
    book.ground_models.add_factory(
        fm.ground_models.WakeMirror,
        "blh_mirror_h<height>",
        var2arg={"height": "heights"},
        height=lambda h: [0.0, float(h)],
        hints={"height": "(Boundary layer wake reflection height)"},
        example_vars={"height": 500},
    )
