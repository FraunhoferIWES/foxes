"""Power-mask figures retain the masked and unmasked scientific comparisons."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest

import foxes
import foxes.variables as FV
from foxes.utils import load_module


EXAMPLE = Path(__file__).resolve().parents[2] / "examples" / "power_mask"


@pytest.mark.parametrize("n_turbines", [1, 5])
def test_power_mask_comparison_figures(monkeypatch, n_turbines):
    # Capture real figures and both farm datasets without opening plot windows.
    figures, results = {}, []
    original = foxes.algorithms.Downwind.calc_farm

    def calculate(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        if kwargs.get("ret_dict"):
            results.append(result["farm_results"])
        return result

    def capture_figures(*args, **kwargs):
        for number in plt.get_fignums():
            figures[plt.figure(number)] = None

    monkeypatch.chdir(EXAMPLE)
    monkeypatch.setattr(foxes.algorithms.Downwind, "calc_farm", calculate)
    monkeypatch.setattr(plt, "show", capture_figures)
    before = set(plt.get_fignums())
    try:
        load_module("power_mask_figures", EXAMPLE / "run.py").main(
            ["-e", "numpy", "-nt", str(n_turbines), "-v", "0"]
        )
        assert len(figures) == 3
        assert len(results) == 2
        masked, unmasked = results
        curve_figure, power_figure, thrust_figure = figures

        # Match the dev curve labels, sampling and per-curve styling.
        np.testing.assert_allclose(curve_figure.get_size_inches(), [10, 4])
        for ax, title, ylabel in zip(
            curve_figure.axes,
            ["Power curve", "Thrust curve"],
            ["Power [kW]", "ct [-]"],
        ):
            assert ax.get_title() == f"{title}, NREL-5MW-D126-H90"
            assert ax.get_ylabel() == ylabel
            assert [line.get_label() for line in ax.lines] == ["default", "PMask"]
            assert [line.get_linewidth() for line in ax.lines] == [2.5, 1.8]
            assert [line.get_linestyle() for line in ax.lines] == ["-", "-"]
            for line in ax.lines:
                np.testing.assert_allclose(
                    line.get_xdata(), np.arange(0, 30.1, 0.1)
                )
            assert all(line.get_visible() for line in ax.get_xgridlines())

        # Each panel must use its own case and share the dev colour scale.
        limit = np.nanmax(masked[FV.MAX_P])
        for figure, variable, label, cmap, vmax in (
            (power_figure, FV.P, "Power", "YlOrRd", limit),
            (thrust_figure, FV.CT, "ct", "YlGn", 1.0),
        ):
            np.testing.assert_allclose(figure.get_size_inches(), [11, 5])
            assert len(figure.axes) == 6
            for ax, data, field, title, palette, maximum in (
                (
                    figure.axes[0],
                    unmasked,
                    variable,
                    f"{label}, no power mask",
                    cmap,
                    vmax,
                ),
                (
                    figure.axes[1],
                    masked,
                    FV.MAX_P,
                    "Power mask",
                    "YlOrRd",
                    limit,
                ),
                (
                    figure.axes[2],
                    masked,
                    variable,
                    f"{label}, with power mask",
                    cmap,
                    vmax,
                ),
            ):
                assert ax.get_title() == title
                image = ax.collections[0]
                assert image.get_cmap().name == palette
                assert image.get_clim() == (0, maximum)
                np.testing.assert_allclose(
                    image.get_array().filled(np.nan), data[field].to_numpy().T
                )
        assert set(plt.get_fignums()) == before
    finally:
        for number in set(plt.get_fignums()) - before:
            plt.close(number)
