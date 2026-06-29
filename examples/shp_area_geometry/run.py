import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from foxes.utils.geom2d import AreaGeometry


DEFAULT_SHP_PATH = Path(__file__).with_name("data") / "area.shp"


def _resolve_shp_path(path_arg):
    path = Path(path_arg).expanduser()
    if path.is_file():
        return path
    if path.is_dir():
        files = sorted(path.glob("*.shp"))
        if len(files):
            return files[0]
    return None


def run(args):
    shp_file = _resolve_shp_path(args.shp_path)
    if shp_file is None:
        print(f"No shapefile found in path: {args.shp_path}")
        print("Skipping example. Provide --shp_path to run it with your data.")
        return 0

    try:
        if args.to_utm:
            geom, utm_zone = AreaGeometry.from_shp(
                str(shp_file),
                name_col=args.name_col,
                to_utm=True,
                ret_utm_zone=True,
            )
        else:
            geom = AreaGeometry.from_shp(
                str(shp_file),
                name_col=args.name_col,
                to_utm=False,
            )
            utm_zone = None
    except ImportError as err:
        print(f"Skipping example because optional dependency is missing: {err}")
        return 0
    except Exception as err:
        print(f"Skipping example because shapefile loading failed: {err}")
        return 0

    fig, ax = plt.subplots(figsize=(8, 6))
    geom.add_to_figure(
        ax,
        show_boundary=True,
        fill_mode="inside_lightgreen",
        pars_boundary={"edgecolor": "forestgreen", "linewidth": 1.5},
    )
    title = f"AreaGeometry loaded from {shp_file.name}"
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
