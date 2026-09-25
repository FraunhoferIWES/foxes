"""Provide binned ambient states on scattered support points.

The public :class:`BinnedPointCloudData` model reduces source states during
initialization, then delegates target interpolation and state chunking to the
native :class:`~foxes.input.states.point_cloud_data.PointCloudData` class.
"""

from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

import numpy as np
import xarray as xr

import foxes.constants as FC
import foxes.variables as FV
from foxes.core import Algorithm, LoadedData, States
from foxes.input.states.point_cloud_data import PointCloudData

from ._output import get_support_wind_roses_figure, write_support_wind_roses
from ._reduction import _BinnedStateReduction


class BinnedPointCloudData(PointCloudData):
    """
    Ambient states reduced to histogram bins on scattered support points.

    When initialized from a source :class:`~foxes.core.States` model, the
    source is evaluated at every support point and reduced into the Cartesian
    product of ``bin_vars``. Empty bins are discarded. Each retained flat bin
    index becomes an ``FC.STATE`` label. All histogram variables are represented
    by bin centers reconstructed from their bounds. Other outputs are weighted
    conditional means over source states and support points.

    Artifacts store non-histogram variables with dimension ``(state,)`` and
    weights with dimensions ``(state, point)``. Optional spatial conditional
    means and population standard deviations use ``(state, point)``. Runtime
    calculations are inherited from
    :class:`~foxes.input.states.point_cloud_data.PointCloudData`: FOXES state
    chunking is applied first, then values and weights are interpolated to
    target points. Calculated weights therefore have dimensions
    ``(state, target, tpoint)``.

    Source support is transferred to model-scoped loaded data during
    initialization. Before worker dispatch, the canonical runtime dataset is
    moved to the standard model stash and duplicate constructor data is
    released from this object. :meth:`unset_running` restores the controller's
    original references after execution.
    """

    def __init__(
        self,
        states: States | str | Path | xr.Dataset,
        *,
        bin_vars: Mapping[str, Sequence[float] | int] | None = None,
        mean_vars: Sequence[str] | None = None,
        support_points: np.ndarray | None = None,
        output_file: str | Path | None = None,
        write_mean_std: bool = False,
        interpolation: str = "linear",
        fill_value: float | None = np.nan,
        bounds_error: bool = True,
        nan_policy: Literal["raise", "interpolate", "remove"] = "raise",
        nan_threshold: float = 0.0,
        **kwargs: Any,
    ) -> None:
        """
        Initialize point-cloud binned ambient states.

        Parameters
        ----------
        states
            Source states to evaluate and reduce, or a NetCDF path or
            :class:`xarray.Dataset` artifact written by
            ``BinnedPointCloudData``.
        bin_vars
            Mapping from variables to monotonically increasing bin edges. An
            integer is accepted for ``WS`` or ``WD`` and creates that many
            standard bins. Required for source states and optional for an
            artifact, which stores its bin definitions.
        mean_vars
            Additional variables retained as weighted means over source states
            and support points. If ``None`` for source states, use every source
            output not listed in ``bin_vars``. Artifact inputs obtain the
            default from metadata.
        support_points
            Scattered support coordinates with shape ``(n_points, 3)`` and
            columns ``x``, ``y``, and ``height``. Required for source states
            and forbidden for artifact input.
        output_file
            Optional path for writing the reduced native NetCDF artifact
            during :meth:`load_data`.
        write_mean_std
            Whether written artifacts include spatial conditional
            ``<variable>_mean`` and ``<variable>_std`` diagnostics for every
            output variable. Defaults to ``False``. Wind-direction deviations
            use circular statistics.
        interpolation
            Scattered interpolation method passed to SciPy through
            :class:`PointCloudData` and used when filling missing statistics.
        fill_value
            Value returned outside the support cloud when ``bounds_error`` is
            ``False``.
        bounds_error
            Whether interpolation outside the support cloud raises an error.
            Internally this is implemented with a NaN fill value and the
            standard ``PointCloudData`` interpolation check.
        nan_policy
            Handling of non-finite statistics in active bins: ``"raise"``,
            spatially ``"interpolate"``, or ``"remove"`` support points before
            filling remaining gaps.
        nan_threshold
            Maximum fraction of active bins with invalid statistics at a
            support point retained by the ``"remove"`` policy.
        kwargs
            Additional keyword arguments for :class:`PointCloudData`.
        """
        initial_source = None if isinstance(states, States) else states
        interp_pars = dict(kwargs.pop("interp_pars", {}) or {})
        interp_pars.setdefault("method", interpolation)
        interp_pars.setdefault(
            "fill_value",
            np.nan if bounds_error else fill_value,
        )
        self._binned = _BinnedStateReduction(
            states,
            topology="point_cloud",
            bin_vars=bin_vars,
            mean_vars=mean_vars,
            support_points=support_points,
            support_grid=None,
            output_file=output_file,
            write_mean_std=write_mean_std,
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
            point_coord=FC.POINT,
            x_ncvar=FV.X,
            y_ncvar=FV.Y,
            h_ncvar=FV.H,
            weight_ncvar=FV.WEIGHT,
            time_format=None,
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
    def support_points(self) -> np.ndarray | None:
        """
        Return the configured scattered support coordinates.

        Returns
        -------
        numpy.ndarray or None
            Configured coordinates with shape ``(n_points, 3)``.
            Artifact-backed objects return ``None`` because their support is
            owned by the loaded native dataset. Source-backed objects return
            ``None`` while running and recover their original support after
            :meth:`unset_running`.
        """
        return self._binned.support_points

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
        Map reduced outputs and spatial weights to the point-cloud contract.

        Parameters
        ----------
        algo
            Algorithm used to resolve default conditional-mean variables.
        """
        calculation_vars = self._binned.calculation_vars(algo)
        self.ovars = calculation_vars
        self.variables = [FV.X, FV.Y, FV.H, *calculation_vars, FV.WEIGHT]
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
        Reduce source states or load an artifact into native point-cloud data.

        The resulting values and spatial weights are stored in ``loaded_data``
        with ``FC.STATE`` as their first dimension. The base
        ``PointCloudData`` loader then prepares them for ordinary FOXES state
        chunking.

        Parameters
        ----------
        algo
            The calculation algorithm.
        loaded_data
            Shared model data populated by this method.
        force
            Replace already loaded entries.
        bounds_extra_space
            Optional horizontal bound extension accepted by the common
            ``DatasetStates`` loader.
        height_bounds
            Optional vertical bounds forwarded to ``PointCloudData``.
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
        self._binned.adopt_runtime_dataset(data)

    def set_running(
        self,
        algo: Algorithm,
        data_stash: dict[str, dict[str, object]] | None,
        sel: dict[str, object] | None = None,
        isel: dict[str, object] | None = None,
        verbosity: int = 0,
    ) -> None:
        """Stash native data and release duplicate initialization inputs."""
        super().set_running(algo, data_stash, sel, isel, verbosity)
        if data_stash is not None:
            data_stash[self.name]["binned"] = self._binned.stash_worker_data()

    def unset_running(
        self,
        algo: Algorithm,
        data_stash: dict[str, dict[str, object]] | None,
        sel: dict[str, object] | None = None,
        isel: dict[str, object] | None = None,
        verbosity: int = 0,
    ) -> None:
        """Restore native and binned initialization data after execution."""
        super().unset_running(algo, data_stash, sel, isel, verbosity)
        if data_stash is not None:
            data = data_stash[self.name].pop("binned")
            if not isinstance(data, dict):
                raise TypeError(f"States '{self.name}': Invalid binned data stash")
            self._binned.restore_worker_data(data)

    def get_support_wind_rose_data(self, loaded_data: Any) -> xr.Dataset:
        """
        Build wind-rose frequencies at every scattered support point.

        Parameters
        ----------
        loaded_data
            Data returned by model initialization.

        Returns
        -------
        xarray.Dataset
            Frequencies in percent with dimensions ``(point, WD, WS)`` and
            explicit support coordinates.

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
