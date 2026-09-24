"""Provide binned ambient states on a regular spatial grid.

The public :class:`BinnedFieldData` model reduces source states during
initialization, then delegates target interpolation and state chunking to the
native :class:`~foxes.input.states.field_data.FieldData` implementation.
"""

from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

import numpy as np
import xarray as xr

import foxes.constants as FC
import foxes.variables as FV
from foxes.core import Algorithm, LoadedData, States
from foxes.input.states.field_data import FieldData

from ._output import get_support_wind_roses_figure, write_support_wind_roses
from ._reduction import _BinnedStateReduction


class BinnedFieldData(FieldData):
    """
    Ambient states reduced to histogram bins on a regular support grid.

    When initialized from a source :class:`~foxes.core.States` model, the
    source is evaluated at every configured support point and reduced into the
    Cartesian product of ``bin_vars``. Empty bins are discarded. Each retained
    flat bin index becomes an ``FC.STATE`` label. Binned wind direction is
    represented by its bin center; all other outputs contain conditional,
    source-weighted means.

    The reduced variables and weights have native dimensions
    ``(state, x, y, height)``. Runtime calculations are inherited from
    :class:`~foxes.input.states.field_data.FieldData`: FOXES state chunking is
    applied first, then values and weights are interpolated to target points.
    Consequently, calculated weights have dimensions
    ``(state, target, tpoint)``.
    """

    def __init__(
        self,
        states: States | str | Path | xr.Dataset,
        *,
        bin_vars: Mapping[str, Sequence[float] | int] | None = None,
        mean_vars: Sequence[str] | None = None,
        support_grid: Mapping[str, Sequence[float]] | None = None,
        output_file: str | Path | None = None,
        interpolation: str = "linear",
        fill_value: float | None = np.nan,
        bounds_error: bool = True,
        nan_policy: Literal["raise", "interpolate"] = "raise",
        nan_threshold: float = 0.0,
        **kwargs: Any,
    ) -> None:
        """
        Initialize regular-grid binned ambient states.

        Parameters
        ----------
        states
            Source states to evaluate and reduce, or a NetCDF path or
            :class:`xarray.Dataset` artifact written by ``BinnedFieldData``.
        bin_vars
            Mapping from variables to monotonically increasing bin edges. An
            integer is accepted for ``WS`` or ``WD`` and creates that many
            standard bins. Required for source states and optional for an
            artifact, which stores its bin definitions.
        mean_vars
            Additional variables retained as conditional weighted means. If
            ``None`` for source states, use every source output not listed in
            ``bin_vars``. Artifact inputs obtain the default from metadata.
        support_grid
            Mapping containing non-empty, strictly increasing ``x``, ``y``,
            and ``height`` axes. Required for source states and forbidden for
            artifact input, which carries its own grid.
        output_file
            Optional path for writing the reduced native NetCDF artifact
            during :meth:`load_data`.
        interpolation
            Regular-grid interpolation method passed to SciPy through
            :class:`FieldData`.
        fill_value
            Value returned outside the support grid when ``bounds_error`` is
            ``False``.
        bounds_error
            Whether target points outside the support grid raise an error.
        nan_policy
            Handling of non-finite statistics in active bins: ``"raise"`` or
            spatially ``"interpolate"`` them before native loading.
        nan_threshold
            Accepted for API consistency with ``BinnedPointCloudData``. Point
            removal is unavailable for a regular grid.
        kwargs
            Additional keyword arguments for :class:`FieldData`.
        """
        initial_source = None if isinstance(states, States) else states
        interp_pars = dict(kwargs.pop("interp_pars", {}) or {})
        interp_pars.setdefault("method", interpolation)
        interp_pars.setdefault("fill_value", fill_value)
        interp_pars.setdefault("bounds_error", bounds_error)
        self._binned = _BinnedStateReduction(
            states,
            topology="field",
            bin_vars=bin_vars,
            mean_vars=mean_vars,
            support_points=None,
            support_grid=support_grid,
            output_file=output_file,
            interpolation=interpolation,
            nan_policy=nan_policy,
            nan_threshold=nan_threshold,
        )
        super().__init__(
            data_source=initial_source,
            output_vars=[],
            var2ncvar={},
            load_mode="preload",
            states_coord=FC.STATE,
            x_coord=FV.X,
            y_coord=FV.Y,
            h_coord=FV.H,
            time_format=None,
            weight_ncvar=FV.WEIGHT,
            bounds_extra_space=np.inf,
            height_bounds=np.inf,
            check_times=False,
            interp_pars=interp_pars,
            **kwargs,
        )

    @property
    def bin_vars(self) -> dict[str, np.ndarray]:
        """
        Return the configured bin edges by variable.

        Returns
        -------
        dict[str, numpy.ndarray]
            Bin edges in Cartesian histogram-axis order.
        """
        return self._binned.bin_vars

    @property
    def mean_vars(self) -> list[str] | None:
        """
        Return the configured conditional-mean variables.

        Returns
        -------
        list[str] or None
            Mean variables, or ``None`` until defaults are resolved.
        """
        return self._binned.mean_vars

    @property
    def support_grid(self) -> dict[str, np.ndarray] | None:
        """
        Return the configured regular support axes.

        Returns
        -------
        dict[str, numpy.ndarray] or None
            The configured ``x``, ``y``, and ``height`` axes. Artifact-backed
            objects return ``None`` because their support is owned by the
            loaded native dataset.
        """
        return self._binned.support_grid

    def sub_models(self) -> list[Any]:
        """
        Return the wrapped source states model.

        Returns
        -------
        list[foxes.core.Model]
            A one-element list for source reduction, otherwise an empty list.
        """
        return self._binned.sub_models()

    def size(self) -> int:
        """
        Return the current number of histogram states.

        Returns
        -------
        int
            Number of Cartesian bins before reduction and retained non-empty
            bins after initialization.
        """
        return self._binned.size()

    def index(self) -> list[int]:
        """
        Return the flat Cartesian indices of the histogram states.

        Returns
        -------
        list[int]
            Retained bin indices used as ``FC.STATE`` labels.
        """
        return self._binned.index()

    def output_point_vars(self, algo: Any) -> list[str]:
        """
        Return variables produced at calculation points.

        Parameters
        ----------
        algo
            The algorithm using this states model.

        Returns
        -------
        list[str]
            Binned variables followed by conditional-mean variables.
        """
        return self._binned.output_vars(algo)

    def _configure_native_data(self, algo: Any) -> None:
        """
        Map reduced outputs and spatial weights to the field-data contract.

        Parameters
        ----------
        algo
            Algorithm used to resolve default conditional-mean variables.
        """
        calculation_vars = self._binned.calculation_vars(algo)
        self.ovars = calculation_vars
        self.variables = [*calculation_vars, FV.WEIGHT]
        self.var2ncvar.update({var: var for var in calculation_vars})
        self.var2ncvar[FV.WEIGHT] = FV.WEIGHT

    def load_data(  # type: ignore[override]
        self,
        algo: Algorithm,
        loaded_data: LoadedData,
        force: bool = False,
        bounds_extra_space: float | str | None = None,
        height_bounds: tuple[float, float] | None = None,
        verbosity: int = 0,
    ) -> None:
        """
        Reduce source states or load an artifact into native field data.

        The resulting values and spatial weights are stored in ``loaded_data``
        with ``FC.STATE`` as their first dimension. The base ``FieldData``
        loader then prepares them for ordinary FOXES state chunking.

        Parameters
        ----------
        algo
            The calculation algorithm.
        loaded_data
            Shared model data populated by this method.
        force
            Replace already loaded entries.
        bounds_extra_space
            Optional horizontal farm-bound extension forwarded to
            :class:`FieldData`.
        height_bounds
            Optional vertical bounds forwarded to :class:`FieldData`.
        verbosity
            Verbosity level, where zero is silent.
        """
        if not force and self.var("meta") in loaded_data["extra_data"]:
            return
        data = self._binned.prepare_dataset(algo, loaded_data, verbosity)
        self._configure_native_data(algo)
        self._binned.preserve_source_data(self, loaded_data)
        self._set_data_source(data)
        super().load_data(
            algo,
            loaded_data,
            force=force,
            bounds_extra_space=bounds_extra_space,
            height_bounds=height_bounds,
            verbosity=verbosity,
        )
        self._binned.install_metadata(self, loaded_data, data)

    def get_support_wind_rose_data(self, loaded_data: Any) -> xr.Dataset:
        """
        Build wind-rose frequencies at every regular-grid point.

        Parameters
        ----------
        loaded_data
            Data returned by model initialization.

        Returns
        -------
        xarray.Dataset
            Frequencies in percent with dimensions ``(point, WD, WS)`` and
            flattened regular-grid support coordinates.

        Raises
        ------
        KeyError
            If the histogram does not contain exactly ``WS`` and ``WD``.
        """
        return self._binned.support_wind_rose_data(self, loaded_data)

    def get_support_wind_roses_figure(
        self,
        data: xr.Dataset,
        **kwargs: Any,
    ) -> Any:
        """
        Create one wind rose per support point on a single canvas.

        Parameters
        ----------
        data
            Dataset returned by :meth:`get_support_wind_rose_data`.
        kwargs
            Figure options accepted by the binned wind-rose renderer.

        Returns
        -------
        matplotlib.figure.Figure
            Figure containing the support-point wind roses.
        """
        return get_support_wind_roses_figure(data, **kwargs)

    def write_support_wind_roses(
        self,
        file_name: str,
        data: xr.Dataset,
        **kwargs: Any,
    ) -> None:
        """
        Write a support-point wind-rose canvas.

        Parameters
        ----------
        file_name
            Output image path.
        data
            Dataset returned by :meth:`get_support_wind_rose_data`.
        kwargs
            Figure options accepted by the binned wind-rose renderer.
        """
        write_support_wind_roses(file_name, data, **kwargs)
