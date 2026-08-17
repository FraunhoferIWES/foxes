from typing import Any

import pandas as pd

from foxes.core import WindFarm

from .from_csv import add_from_csv


def add_from_df(
    farm: WindFarm,
    data_source: str | pd.DataFrame,
    *args: Any,
    **kwargs: Any,
) -> None:
    """
    Add turbines to wind farm via pandas DataFrame.

    Parameters
    ----------
    farm
        The wind farm
    data_source
        The input csv file or data source
    args
        Additional parameters for add_from_csv()
    kwargs
        Additional parameters for add_from_csv()

    :group: input.farm_layout

    """
    add_from_csv(farm, data_source, *args, **kwargs)
