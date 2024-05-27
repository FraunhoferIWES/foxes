import numpy as np
import argparse
import matplotlib.pyplot as plt

import foxes
import foxes.variables as FV

if __name__ == "__main__":
    # define arguments and options:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-wio", "--windio_yaml", 
        help="The windio wind energy systems yaml file", 
        default="windio_5turbines_timeseries.yaml",
    )
    parser.add_argument(
        "-nf", "--nofig", help="Do not show figures", action="store_true"
    )
    args = parser.parse_args()

    mbook, farm, states, algo = foxes.input.windio.read_case(args.windio_yaml)