import numpy as np
import pytest

from foxes.utils.wind_dir import WindDirectionHistogram


def test_wind_direction_histogram_combines_batches_across_north():
    first = WindDirectionHistogram(30.0)
    first.add(np.array([355.0, 90.0]))
    second = WindDirectionHistogram(30.0)
    second.add(np.array([5.0]), weights=np.array([3.0]))

    first.combine(second)

    assert first.n_bins == 24
    assert first.bin_width == 30.0
    assert first.bin_step == 15.0
    assert first.counts[0] == 4.0
    expected = np.rad2deg(
        np.arctan2(
            np.sin(np.deg2rad(355.0)) + 3.0 * np.sin(np.deg2rad(5.0)),
            np.cos(np.deg2rad(355.0)) + 3.0 * np.cos(np.deg2rad(5.0)),
        )
    )
    np.testing.assert_allclose(first.main_direction(), expected)


def test_wind_direction_histogram_captures_mode_across_bin_boundary():
    histogram = WindDirectionHistogram(10.0)
    histogram.add(np.array([4.9, 5.1, 90.0]))

    assert histogram.counts[1] == 2.0
    np.testing.assert_allclose(histogram.main_direction(), 5.0)


def test_wind_direction_histogram_returns_nan_without_valid_direction():
    histogram = WindDirectionHistogram(30.0)
    histogram.add(np.array([np.nan, np.nan]))

    assert np.isnan(histogram.main_direction())


def test_wind_direction_histogram_canonicalizes_symmetric_north():
    histogram = WindDirectionHistogram(30.0)
    histogram.add(np.array([355.0, 5.0]))

    assert histogram.main_direction() == 0.0

    histogram = WindDirectionHistogram(30.0)
    histogram.add(np.array([359.999]))

    np.testing.assert_allclose(histogram.main_direction(), 359.999)


@pytest.mark.parametrize("bin_width", [0.0, -1.0, 361.0, np.nan, np.inf])
def test_wind_direction_histogram_rejects_invalid_width(bin_width):
    with pytest.raises(ValueError, match="bin width must be in"):
        WindDirectionHistogram(bin_width)


def test_wind_direction_histogram_rejects_invalid_weights():
    histogram = WindDirectionHistogram()

    with pytest.raises(ValueError, match="weights must be finite and non-negative"):
        histogram.add(np.array([0.0, 10.0]), weights=np.array([1.0, -1.0]))
