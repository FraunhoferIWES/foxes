from __future__ import annotations

import argparse
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import foxes
import foxes.variables as FV


def build_case(
    y_values: np.ndarray,
    distance: float,
    rotor_model: str,
    partial_wakes: str,
    outputs: list[str],
    nofig: bool = False,
    verbosity: int = 0,
):
    """Run a two-turbine lateral-offset case with the selected chunk outputs.

    Parameters
    ----------
    y_values
        Lateral positions of the downstream turbine in metres, one per state.
    distance
        Downstream spacing in rotor diameters.
    rotor_model
        Rotor model-book name.
    partial_wakes
        Partial-wake model-book name.
    outputs
        Output model-book names passed to the farm calculation.
    nofig
        Disable display on selected output models.
    verbosity
        Example reporting level; two prints collected farm results, three adds
        detailed library diagnostics, and zero is silent.

    Returns
    -------
    rews
        Downstream rotor-effective wind speeds, one per state, or ``None`` if
        the selected outputs do not collect farm results containing ``REWS``.
    diameter
        Turbine rotor diameter in metres.
    """
    mbook = foxes.models.ModelBook()
    if nofig:
        for name in outputs:
            output = mbook.outputs[name]
            if hasattr(output, "show"):
                output.show = False
    turbine_type = mbook.turbine_types["NREL5MW"]
    mbook.turbine_models["set_y"] = foxes.models.turbine_models.SetFarmVars()
    y_data = np.full((y_values.size, 2), np.nan)
    y_data[:, 1] = y_values
    mbook.turbine_models["set_y"].add_var(FV.Y, y_data)

    farm = foxes.WindFarm()
    farm.add_turbine(
        foxes.Turbine(
            xy=np.array([0.0, 0.0]),
            turbine_models=["NREL5MW"],
        ),
        verbosity=max(verbosity - 1, 0),
    )
    farm.add_turbine(
        foxes.Turbine(
            xy=np.array([distance * turbine_type.D, 0.0]),
            turbine_models=["set_y", "NREL5MW"],
        ),
        verbosity=max(verbosity - 1, 0),
    )

    states_data = pd.DataFrame(
        {
            "ws": np.full(y_values.size, 9.0),
            "wd": np.full(y_values.size, 270.0),
            "ti": np.full(y_values.size, 0.08),
            "rho": np.full(y_values.size, 1.225),
        }
    )
    states = foxes.input.states.StatesTable(
        states_data,
        output_vars=[FV.WS, FV.WD, FV.TI, FV.RHO],
        var2col={FV.WS: "ws", FV.WD: "wd", FV.TI: "ti", FV.RHO: "rho"},
    )
    algo = foxes.algorithms.Downwind(
        farm,
        states,
        wake_models=["Bastankhah2014_linear_k004"],
        rotor_model=rotor_model,
        partial_wakes={"Bastankhah2014_linear_k004": partial_wakes},
        wake_frame="rotor_wd",
        mbook=mbook,
        verbosity=max(verbosity - 1, 0),
    )
    farm_results = algo.calc_farm(outputs=outputs)

    # Writer-only cases still run, but have no numerical comparison payload.
    if isinstance(farm_results, dict):
        farm_results = farm_results.get("farm_results")
    elif not any(
        isinstance(mbook.outputs[name], foxes.models.outputs.FarmResultsOutput)
        for name in outputs
    ):
        farm_results = None
    if farm_results is not None and verbosity > 1:
        print("\nFarm results:\n", farm_results)
    required = {FV.REWS}
    if farm_results is None or not required.issubset(farm_results.data_vars):
        return None, turbine_type.D

    return farm_results[FV.REWS].to_numpy()[:, 1], turbine_type.D


def build_benchmark_farm(n_turbines: int, diameter: float) -> foxes.WindFarm:
    """Build a deterministic square-grid benchmark farm."""
    if n_turbines < 1:
        raise ValueError("n_turbines must be >= 1")

    n_columns = int(np.ceil(np.sqrt(n_turbines)))
    spacing = 6.0 * diameter
    farm = foxes.WindFarm()
    for turbine_index in range(n_turbines):
        row, column = divmod(turbine_index, n_columns)
        farm.add_turbine(
            foxes.Turbine(
                xy=np.array([column * spacing, row * spacing]),
                turbine_models=["NREL5MW"],
            ),
            verbosity=0,
        )
    return farm


def benchmark_large_case(
    engine: foxes.Engine,
    n_turbines: int,
    outputs: list[str],
    nofig: bool = False,
    verbosity: int = 0,
) -> None:
    """Compare output-inclusive runtimes for the generated benchmark farm.

    Parameters
    ----------
    engine
        Active calculation engine.
    n_turbines
        Number of turbines in the generated square-grid farm.
    outputs
        Output model-book names passed to every benchmark calculation.
    nofig
        Disable display on selected output models.
    verbosity
        Example reporting level; two prints collected farm results and runtime
        reports, three adds detailed library diagnostics, and zero is silent.
    """
    runtimes: dict[str, float] = {}
    cases = {
        "gaussian": ("centre", "gaussian"),
        "gaussian_lookup": ("centre", "gaussian_lookup"),
        "grid400": ("grid400", "rotor_points"),
    }
    for name, (rotor_model, partial_wakes) in cases.items():
        mbook = foxes.models.ModelBook()
        if nofig:
            for output_name in outputs:
                output = mbook.outputs[output_name]
                if hasattr(output, "show"):
                    output.show = False
        farm = build_benchmark_farm(n_turbines, mbook.turbine_types["NREL5MW"].D)
        states = foxes.input.states.Timeseries(
            "timeseries_8000.csv.gz",
            output_vars=[FV.WS, FV.WD, FV.TI, FV.RHO],
            var2col={FV.WS: "ws", FV.WD: "wd", FV.TI: "ti"},
            fixed_vars={FV.RHO: 1.225},
        )
        algo = foxes.algorithms.Downwind(
            farm,
            states,
            wake_models=["Bastankhah2014_linear_k004"],
            rotor_model=rotor_model,
            partial_wakes=partial_wakes,
            wake_frame="rotor_wd",
            mbook=mbook,
            verbosity=max(verbosity - 1, 0),
        )
        start = time.perf_counter()
        farm_results = algo.calc_farm(outputs=outputs)
        runtimes[name] = time.perf_counter() - start
        if isinstance(farm_results, dict):
            farm_results = farm_results.get("farm_results")
        elif not any(
            isinstance(mbook.outputs[name], foxes.models.outputs.FarmResultsOutput)
            for name in outputs
        ):
            farm_results = None
        if farm_results is not None and verbosity > 1:
            print("\nFarm results:\n", farm_results)

    grid_time = runtimes["grid400"]
    if verbosity > 1:
        print(f"\nLarge-case runtime comparison ({n_turbines} turbines, 8000 states)")
        print(f"gaussian:        {runtimes['gaussian']:.3f} s")
        print(f"gaussian_lookup: {runtimes['gaussian_lookup']:.3f} s")
        print(f"grid400:         {grid_time:.3f} s")
        print(
            f"gaussian / grid400 runtime ratio: {runtimes['gaussian'] / grid_time:.3f}"
        )
        print(
            "gaussian_lookup / grid400 runtime ratio: "
            f"{runtimes['gaussian_lookup'] / grid_time:.3f}"
        )


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare Gaussian, Gaussian lookup, and grid400 partial wakes"
    )
    parser.add_argument(
        "-d",
        "--distance",
        type=float,
        default=4.0,
        help="Downstream turbine distance in rotor diameters",
    )
    parser.add_argument(
        "-y",
        "--y-span",
        type=float,
        default=2.0,
        help="Lateral scan half-width in rotor diameters",
    )
    parser.add_argument(
        "-s",
        "--step",
        type=float,
        default=0.05,
        help="Lateral scan spacing in rotor diameters",
    )
    parser.add_argument("-e", "--engine", default="process", help="Engine type")
    parser.add_argument("-n", "--n-cpus", type=int, default=None, help="Worker count")
    parser.add_argument(
        "-c",
        "--chunk-size-states",
        type=int,
        default=None,
        help="State chunk size",
    )
    parser.add_argument(
        "-C",
        "--chunk-size-points",
        type=int,
        default=None,
        help="Point chunk size",
    )
    parser.add_argument(
        "--figure-file", type=Path, default=None, help="Optional PNG output path"
    )
    parser.add_argument(
        "-rt",
        "--runtime-benchmark",
        action="store_true",
        help="Also compare runtime on a generated farm with 8000 states",
    )
    parser.add_argument(
        "-nt",
        "--benchmark-turbines",
        type=int,
        default=67,
        help="Number of turbines in the runtime benchmark grid",
    )
    parser.add_argument(
        "-nf", "--nofig", action="store_true", help="Do not show the figure"
    )

    # Select chunk outputs separately from the comparison figure filename.
    parser.add_argument(
        "-o", "--outputs", help="The output models", nargs="+", default=["farm_results"]
    )
    parser.add_argument(
        "-v",
        "--verbosity",
        type=int,
        default=2,
        help="0 quiet, 1 summary, 2 reports, 3 library progress",
    )
    return parser.parse_args(argv)


def main(argv=None) -> None:
    """Run this tutorial with an explicit argument list or command-line options."""
    args = parse_args(argv)
    if args.verbosity > 0:
        print("Running example: compare_gaussian_lookup_pwakes")
    y_values_D = np.arange(-args.y_span, args.y_span + args.step * 0.5, args.step)
    y_values = y_values_D * 126.0

    engine = foxes.Engine.new(
        engine_type=args.engine,
        n_procs=args.n_cpus,
        chunk_size_states=args.chunk_size_states,
        chunk_size_points=args.chunk_size_points,
        verbosity=max(args.verbosity - 1, 0),
    )
    with engine:
        gaussian_rews, diameter = build_case(
            y_values,
            args.distance,
            "centre",
            "gaussian",
            args.outputs,
            args.nofig,
            verbosity=args.verbosity,
        )
        lookup_rews, diameter = build_case(
            y_values,
            args.distance,
            "centre",
            "gaussian_lookup",
            args.outputs,
            args.nofig,
            verbosity=args.verbosity,
        )
        grid_rews, _ = build_case(
            y_values,
            args.distance,
            "grid400",
            "rotor_points",
            args.outputs,
            args.nofig,
            verbosity=args.verbosity,
        )

        # Include requested writers in every optional benchmark case too.
        if args.runtime_benchmark:
            benchmark_large_case(
                engine,
                args.benchmark_turbines,
                args.outputs,
                args.nofig,
                verbosity=args.verbosity,
            )

    # Writer-only selections finish all cases without requiring a comparison figure.
    if gaussian_rews is None or lookup_rews is None or grid_rews is None:
        print("Skipping comparison: select a farm-results output containing REWS.")
        return

    gaussian_norm = gaussian_rews / 9.0
    lookup_norm = lookup_rews / 9.0
    grid_norm = grid_rews / 9.0
    gaussian_difference = np.abs(gaussian_norm - grid_norm)
    lookup_difference = np.abs(lookup_norm - grid_norm)

    if args.verbosity > 1:
        print("Gaussian and Gaussian lookup vs grid400")
        print(
            "Gaussian maximum absolute normalized REWS difference: "
            f"{gaussian_difference.max():.3e}"
        )
        print(
            "Gaussian mean absolute normalized REWS difference: "
            f"{gaussian_difference.mean():.3e}"
        )
        print(
            "Gaussian lookup maximum absolute normalized REWS difference: "
            f"{lookup_difference.max():.3e}"
        )
        print(
            "Gaussian lookup mean absolute normalized REWS difference: "
            f"{lookup_difference.mean():.3e}"
        )
        print(f"Scan range: {y_values_D[0]:.2f}D to {y_values_D[-1]:.2f}D")
        print(f"Downstream distance: {args.distance:.2f}D (NREL5MW D={diameter:.1f} m)")

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(9, 7),
        sharex=True,
        gridspec_kw={"height_ratios": (3, 1)},
        layout="constrained",
    )
    axes[0].plot(y_values_D, gaussian_norm, label="gaussian", linewidth=2)
    axes[0].plot(y_values_D, lookup_norm, label="gaussian_lookup", linewidth=2)
    axes[0].plot(
        y_values_D,
        grid_norm,
        "--",
        label="grid400",
        linewidth=2,
    )
    axes[0].set_ylabel("Downstream REWS / ambient WS")
    axes[0].set_title(
        "Partial-wake comparison at a lateral scan of the downstream rotor"
    )
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    axes[1].plot(
        y_values_D,
        gaussian_difference,
        label="gaussian - grid400",
        linewidth=1.5,
    )
    axes[1].plot(
        y_values_D,
        lookup_difference,
        label="gaussian_lookup - grid400",
        linewidth=1.5,
    )
    axes[1].set_xlabel("Downstream turbine lateral offset [D]")
    axes[1].set_ylabel("Absolute difference")
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    if args.figure_file is not None:
        args.figure_file.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.figure_file, dpi=150)
        if args.verbosity > 1:
            print(f"Wrote comparison figure to {args.figure_file}")
    if not args.nofig:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()
