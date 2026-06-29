import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from foxes.utils.geom2d import AreaGeometry


DEFAULT_SHP_PATH = Path(__file__).with_name("data") / "area.shp"


def run(args):
    combine_mode = "intersection" if args.intersection else "union"

    if args.to_utm:
        geom, utm_zone = AreaGeometry.from_shp(
            args.shp_path,
            name_col=args.name_col,
            to_utm=True,
            combine_mode=combine_mode,
            ret_utm_zone=True,
        )
    else:
        geom = AreaGeometry.from_shp(
            args.shp_path,
            name_col=args.name_col,
            to_utm=False,
            combine_mode=combine_mode,
        )
        utm_zone = None

    fig, ax = plt.subplots(figsize=(8, 6))
    geom.add_to_figure(
        ax,
        show_boundary=True,
        fill_mode="inside_lightgreen",
        pars_boundary={"edgecolor": "forestgreen", "linewidth": 1.5},
    )
    if args.to_utm:
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
    else:
        ax.set_xlabel("longitude [deg]")
        ax.set_ylabel("latitude [deg]")

    title = args.title
    if utm_zone is not None:
        title += f" (UTM {utm_zone})"
    ax.set_title(title)
    fig.tight_layout()

    if args.output is not None:
        fig.savefig(args.output, dpi=150)
        print(f"Wrote figure to {args.output}")

    if not args.nofig:
        plt.show()

    plt.close(fig)
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-s",
        "--shp_path",
        help="Path to input .shp file or to a directory containing .shp files",
        default=str(DEFAULT_SHP_PATH),
    )
    parser.add_argument(
        "-u",
        "--to_utm",
        help="Convert lon/lat polygons to UTM coordinates",
        action="store_true",
    )
    parser.add_argument(
        "-nc",
        "--name_col",
        help="Column containing polygon names in the shapefile",
        default="TYPE",
    )
    parser.add_argument(
        "-t",
        "--title",
        help="Base figure title",
        default="AreaGeometry",
    )
    parser.add_argument(
        "-i",
        "--intersection",
        help="Use intersection instead of union when combining multiple areas",
        action="store_true",
    )
    parser.add_argument(
        "-o",
        "--output",
        help="Output image path",
        default=None,
    )
    parser.add_argument(
        "-nf",
        "--nofig",
        help="Skip figure display",
        action="store_true",
    )

    raise SystemExit(run(parser.parse_args()))
