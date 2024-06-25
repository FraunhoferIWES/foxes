import argparse

import foxes

if __name__ == "__main__":
    # define arguments and options:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-wio",
        "--windio_yaml",
        help="The windio wind energy systems yaml file",
        default="windio_5turbines_timeseries.yaml",
    )
    parser.add_argument(
        "-nf", "--nofig", help="Do not show figures", action="store_true"
    )
    parser.add_argument(
        "-V", "--verbosity", help="The verbosity level, 0=silent", type=int, default=1
    )
    args = parser.parse_args()

    wio_runner = foxes.input.windio.read_windio(
        args.windio_yaml, verbosity=args.verbosity
    )

    with wio_runner as runner:
        runner.run()
