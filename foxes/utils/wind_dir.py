import numpy as np


class WindDirectionHistogram:
    """Incremental overlapping histogram for a main wind direction."""

    def __init__(self, bin_width: float = 10.0) -> None:
        """
        Initialize the histogram.

        Parameters
        ----------
        bin_width
            The minimum sector width in degrees. The actual width is adjusted
            upwards so that equally sized sectors cover the full circle.
            Sectors overlap by 50 percent, with one sector centred on north.
        """
        width = float(bin_width)
        if not np.isfinite(width) or width <= 0.0 or width > 360.0:
            raise ValueError(
                f"Wind-direction histogram bin width must be in (0, 360], got {bin_width}"
            )
        base_bins = int(360.0 / width)
        self.bin_width = 360.0 / base_bins
        self.bin_step = self.bin_width / 2.0
        self.n_bins = 2 * base_bins
        self._counts: np.ndarray | None = None
        self._sin_sums: np.ndarray | None = None
        self._cos_sums: np.ndarray | None = None

    def add(
        self,
        wind_directions: np.ndarray,
        weights: float | np.ndarray = 1.0,
        axis: int = -1,
    ) -> None:
        """Add wind directions and optional statistical weights."""
        directions = np.asarray(wind_directions, dtype=float)
        if directions.ndim == 0:
            directions = directions.reshape(1)
            axis = 0
        try:
            sample_weights = np.broadcast_to(
                np.asarray(weights, dtype=float), directions.shape
            )
        except ValueError as exc:
            raise ValueError("Wind-direction weights are not broadcastable") from exc
        if not np.all(np.isfinite(sample_weights)) or np.any(sample_weights < 0.0):
            raise ValueError("Wind-direction weights must be finite and non-negative")

        directions = np.moveaxis(directions, axis, -1)
        sample_weights = np.moveaxis(sample_weights, axis, -1)
        self._initialize(directions.shape[:-1])
        counts, sin_sums, cos_sums = self._statistics()
        valid = np.isfinite(directions) & (sample_weights > 0.0)
        normalized = np.mod(np.where(valid, directions, 0.0), 360.0)
        lower_indices = (
            np.floor(normalized / self.bin_step).astype(int).clip(max=self.n_bins - 1)
        )
        upper_indices = (lower_indices + 1) % self.n_bins
        radians = np.deg2rad(normalized)
        for bin_index in range(self.n_bins):
            in_sector = (lower_indices == bin_index) | (upper_indices == bin_index)
            bin_weights = np.where(valid & in_sector, sample_weights, 0.0)
            counts[..., bin_index] += np.sum(bin_weights, axis=-1)
            sin_sums[..., bin_index] += np.sum(bin_weights * np.sin(radians), axis=-1)
            cos_sums[..., bin_index] += np.sum(bin_weights * np.cos(radians), axis=-1)

    def combine(self, other: "WindDirectionHistogram") -> None:
        """Add statistics from another compatible histogram."""
        if self.n_bins != other.n_bins:
            raise ValueError(
                "Cannot combine wind-direction histograms with different bins"
            )
        if other._counts is None:
            return
        other_counts, other_sin_sums, other_cos_sums = other._statistics()
        if self._counts is None:
            self._counts = other_counts.copy()
            self._sin_sums = other_sin_sums.copy()
            self._cos_sums = other_cos_sums.copy()
        elif self._counts.shape != other_counts.shape:
            raise ValueError(
                "Cannot combine wind-direction histograms of different shapes"
            )
        else:
            counts, sin_sums, cos_sums = self._statistics()
            counts += other_counts
            sin_sums += other_sin_sums
            cos_sums += other_cos_sums

    def main_direction(self) -> np.ndarray:
        """Return the circular mean of the sector with the greatest weight."""
        counts, sin_sums, cos_sums = self._statistics()
        winner = np.argmax(counts, axis=-1)[..., None]
        max_counts = np.take_along_axis(counts, winner, axis=-1)[..., 0]
        sin_sum = np.take_along_axis(sin_sums, winner, axis=-1)[..., 0]
        cos_sum = np.take_along_axis(cos_sums, winner, axis=-1)[..., 0]
        magnitude = np.hypot(sin_sum, cos_sum)
        valid = (max_counts > 0.0) & (magnitude > 1.0e-12 * max_counts)
        direction = np.mod(np.rad2deg(np.arctan2(sin_sum, cos_sum)), 360.0)
        direction = np.where(
            np.isclose(direction, 360.0, rtol=0.0, atol=1.0e-12), 0.0, direction
        )
        return np.where(valid, direction, np.nan)

    @property
    def counts(self) -> np.ndarray:
        """Accumulated weight in each 50%-overlapping direction sector."""
        return self._statistics()[0]

    def _initialize(self, output_shape: tuple[int, ...]) -> None:
        """Initialize or validate the accumulator arrays."""
        shape = output_shape + (self.n_bins,)
        if self._counts is None:
            self._counts = np.zeros(shape, dtype=float)
            self._sin_sums = np.zeros(shape, dtype=float)
            self._cos_sums = np.zeros(shape, dtype=float)
        elif self._counts.shape != shape:
            raise ValueError(
                f"Wind-direction histogram shape changed from {self._counts.shape} to {shape}"
            )

    def _statistics(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return initialized accumulator arrays."""
        if self._counts is None:
            raise ValueError("No wind directions have been added")
        assert self._sin_sums is not None and self._cos_sums is not None
        return self._counts, self._sin_sums, self._cos_sums


def wd2wdvec(
    wd: np.ndarray, ws: float | np.ndarray = 1.0, axis: int = -1
) -> np.ndarray:
    """
    Calculate wind direction vectors from wind directions
    in degrees.

    Parameters
    ----------
    wd
        Wind direction array (any shape)
    ws
        The wind speed. Has to broadcast against wd.
    axis
        Location where to insert the (x, y) dimension
        into the shape of wd

    Returns
    -------
    wdvec
        The wind direction vectors


    """
    wdr = wd * np.pi / 180.0
    n = np.stack([np.sin(wdr), np.cos(wdr)], axis=axis)

    if np.isscalar(ws):
        return np.asarray(ws * n)

    return np.expand_dims(ws, axis) * n


def wd2uv(wd: np.ndarray, ws: float | np.ndarray = 1.0, axis: int = -1) -> np.ndarray:
    """
    Calculate wind vectors from wind directions
    in degrees.

    Parameters
    ----------
    wd
        Wind direction array (any shape)
    ws
        The wind speed. Has to broadcast against wd.
    axis
        Axis location where to insert the (u, v) components
        into the shape of wd

    Returns
    -------
    uv
        The wind vectors


    """
    return -wd2wdvec(wd, ws, axis)


def uv2wd(uv: np.ndarray, axis: int = -1) -> np.ndarray:
    """
    Calculate wind direction from wind vectors.

    Parameters
    ----------
    uv
        The wind vectors, any shape
    axis
        The axis which corresponds to (u, v) components

    Returns
    -------
    wd
        The wind direction array


    """
    if axis == -1:
        u = uv[..., 0]
        v = uv[..., 1]
    else:
        s = tuple(0 if a == axis else slice(None) for a in range(len(uv.shape)))
        u = uv[s]
        s = tuple(1 if a == axis else slice(None) for a in range(len(uv.shape)))
        v = uv[s]

    return np.mod(180 + np.rad2deg(np.arctan2(u, v)), 360)


def wdvec2wd(wdvec: np.ndarray, axis: int = -1) -> np.ndarray:
    """
    Calculate wind direction from wind direction vectors.

    Parameters
    ----------
    wdvec
        The wind direction vectors, any shape
    axis
        The axis which corresponds to (x, y) components

    Returns
    -------
    wd
        The wind direction array


    """
    return uv2wd(-wdvec, axis)


def delta_wd(wd_a: np.ndarray, wd_b: np.ndarray) -> np.ndarray:
    """
    Calculates wd_b - wd_a.

    Parameters
    ----------
    wd_a
        Array of wind directions.
        Shape
    wd_b
        Array of wind directions.
        Shape: same as wd_a

    Returns
    -------
    Array
        Array of wind direction deltas.
        Shape: same as wd_a, wd_b


    """
    out = wd_b - wd_a

    out[out < -180.0] += 360.0
    out[out > 180.0] -= 360.0

    return out
