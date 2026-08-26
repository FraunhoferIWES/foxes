from __future__ import annotations

import argparse
from pathlib import Path
from typing import Literal, Sequence

from foxes.utils.gaussian_lookup import LOOKUP_VERSION
from foxes.utils.gaussian_lookup import generate_lookup_dataset
from foxes.utils.gaussian_lookup import save_lookup_dataset


def create_gaussian_lookup_artifact(
    out_file: str | Path,
    min_weight: float = 1.0e-8,
    r_over_sigma_max: float | None = None,
    sigma_over_d_min: float = 0.02,
    sigma_over_d_max: float = 20.0,
    radial_resolution: float = 0.1,
    sigma_resolution: float = 0.05,
    sigma_spacing: Literal["linear", "log"] = "log",
    n_rho: int = 512,
    version_tag: str = LOOKUP_VERSION,
    radial_expand_factor: float = 1.2,
    complevel: int = 5,
    nc_engine: str | None = None,
    verbosity: int = 1,
) -> Path:
    """
    Generate and persist a Gaussian lookup NetCDF artifact.

    Parameters
    ----------
    out_file
        Target NetCDF file path.
    radial_resolution
        Approximate spacing between interpolation samples on the ``R/sigma`` axis.
    sigma_resolution
        Approximate spacing between interpolation samples on the ``sigma/D`` axis.
    sigma_spacing
        Axis spacing mode for ``sigma/D``, either ``"linear"`` or ``"log"``.
    n_rho
        Radial quadrature resolution used for weight integration.
    version_tag
        Artifact schema version tag.
    min_weight
        Lower weight threshold used to derive the radial lookup extent when
        ``r_over_sigma_max`` is omitted.
    r_over_sigma_max
        Optional upper bound of the ``R/sigma`` axis.
    sigma_over_d_min
        Lower bound of the ``sigma/D`` axis.
    sigma_over_d_max
        Upper bound of the ``sigma/D`` axis.
    radial_expand_factor
        Radial auto-expansion multiplier.
    complevel
        NetCDF compression level.
    nc_engine
        NetCDF engine used for writing.
    verbosity
        Verbosity level, 0 disables progress messages.

    Returns
    -------
    out_path
        The created artifact path.
    """
    out_path = Path(out_file).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    ds = generate_lookup_dataset(
        radial_resolution=radial_resolution,
        sigma_resolution=sigma_resolution,
        sigma_spacing=sigma_spacing,
        n_rho=n_rho,
        version_tag=version_tag,
        min_weight=min_weight,
        r_over_sigma_max=r_over_sigma_max,
        sigma_over_d_min=sigma_over_d_min,
        sigma_over_d_max=sigma_over_d_max,
        radial_expand_factor=radial_expand_factor,
    )
    if verbosity > 0:
        print(f"Writing Gaussian lookup artifact to {out_path}")

    save_lookup_dataset(
        ds,
        out_path,
        complevel=complevel,
        nc_engine=nc_engine,
    )
    return out_path


def _build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for Gaussian lookup artifact creation."""
    parser = argparse.ArgumentParser(
        description="Generate Gaussian partial-wake lookup weights as NetCDF artifact"
    )
    parser.add_argument("out_file", type=str, help="Output NetCDF artifact path")
    parser.add_argument(
        "--radial-resolution",
        dest="radial_resolution",
        type=float,
        default=0.1,
        help="Approximate spacing between R/sigma points",
    )
    parser.add_argument(
        "--sigma-resolution",
        dest="sigma_resolution",
        type=float,
        default=0.05,
        help="Approximate spacing between sigma/D points",
    )
    parser.add_argument(
        "--sigma-spacing",
        dest="sigma_spacing",
        choices=("linear", "log"),
        default="log",
        help="Spacing mode for sigma/D axis",
    )
    parser.add_argument(
        "--n-rho",
        dest="n_rho",
        type=int,
        default=512,
        help="Radial quadrature resolution",
    )
    parser.add_argument(
        "--version-tag",
        dest="version_tag",
        type=str,
        default=LOOKUP_VERSION,
        help="Artifact schema version tag",
    )
    parser.add_argument(
        "--min-weight",
        dest="min_weight",
        type=float,
        default=1.0e-8,
        help="Lower weight threshold for radial sizing and runtime cutoff",
    )
    parser.add_argument(
        "--r-over-sigma-max",
        dest="r_over_sigma_max",
        type=float,
        default=None,
        help="Upper R/sigma axis bound; omitted derives from min-weight",
    )
    parser.add_argument(
        "--sigma-over-d-min",
        dest="sigma_over_d_min",
        type=float,
        default=0.02,
        help="Lower sigma/D axis bound",
    )
    parser.add_argument(
        "--sigma-over-d-max",
        dest="sigma_over_d_max",
        type=float,
        default=20.0,
        help="Upper sigma/D axis bound",
    )
    parser.add_argument(
        "--expand-factor",
        dest="radial_expand_factor",
        type=float,
        default=1.2,
        help="Radial auto-expansion multiplier",
    )
    parser.add_argument(
        "--complevel",
        dest="complevel",
        type=int,
        default=5,
        help="NetCDF compression level",
    )
    parser.add_argument(
        "--nc-engine",
        dest="nc_engine",
        type=str,
        default="netcdf4",
        help="NetCDF backend engine",
    )
    parser.add_argument(
        "-v",
        "--verbosity",
        type=int,
        default=1,
        help="Verbosity level, 0 = silent",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """
    Command-line entry point for Gaussian lookup artifact generation.

    Parameters
    ----------
    argv
        Optional command-line argument list. Uses ``sys.argv`` when omitted.
    """
    parser = _build_parser()
    args = parser.parse_args(argv)
    create_gaussian_lookup_artifact(
        out_file=args.out_file,
        radial_resolution=args.radial_resolution,
        sigma_resolution=args.sigma_resolution,
        sigma_spacing=args.sigma_spacing,
        n_rho=args.n_rho,
        version_tag=args.version_tag,
        min_weight=args.min_weight,
        r_over_sigma_max=args.r_over_sigma_max,
        sigma_over_d_min=args.sigma_over_d_min,
        sigma_over_d_max=args.sigma_over_d_max,
        radial_expand_factor=args.radial_expand_factor,
        complevel=args.complevel,
        nc_engine=args.nc_engine,
        verbosity=args.verbosity,
    )


if __name__ == "__main__":
    main()