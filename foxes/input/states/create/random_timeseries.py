import numpy as np
import pandas as pd

import foxes.variables as FV


def random_timseries_data(
    n_times,
    data_ranges=None,
    start_time="2000-01-01 00:00:00",
    freq="h",
    seed=None,
    iname="Time",
):
    """
    Creates random uniform timeseries data

    Parameters
    ----------
    n_times: int
        The number of time steps
    data_ranges: dict, optional
        The data ranges. Key: variable name,
        value: tuple, [min, max) values
    start_time: str
        The first time stamp in the series
    freq: str
        The time period range frequency
    seed: int, optional
        The random seed
    iname: str
        The index name

    Returns
    -------
    sdata: pandas.DataFrame
        The timeseries data

    :group: input.states.create

    """
    if seed:
        np.random.seed(seed)

    dranges = {FV.WS: (0.0, 30.0), FV.WD: (0.0, 360.0)}
    if data_ranges:
        dranges.update(data_ranges)

    times = pd.period_range(start=start_time, periods=n_times, freq=freq)
    times = times.astype(str).astype("datetime64[ns]")
    sdata = pd.DataFrame(
        index=times,
        data={v: np.random.uniform(d[0], d[1], n_times) for v, d in dranges.items()},
    )
    sdata.index.name = iname
    return sdata
