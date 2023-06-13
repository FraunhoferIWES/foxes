from .from_csv import add_from_csv


def add_from_df(
    farm,
    data_source,
    *args,
    **kwargs,
):
    """
    Add turbines to wind farm via pandas DataFrame.

    Parameters
    ----------
    farm: foxes.WindFarm
        The wind farm
    data_source: str or pandas.DataFrame
        The input csv file or data source
    args: tuple, optional
        Additional parameters for add_from_csv()
    kwargs: dict, optional
        Additional parameters for add_from_csv()

    :group: input.farm_layout

    """
    add_from_csv(farm, data_source, *args, **kwargs)
