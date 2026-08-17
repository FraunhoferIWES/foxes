import numpy as np
import pandas as pd

import foxes.variables as FV


def random_timseries_data(
    n_times: int,
    data_ranges: dict[str, tuple[float, float]] | None = None,
    start_time: str = "2000-01-01 00:00:00",
    freq: str = "h",
    seed: int | None = None,
    iname: str = "Time",
) -> pd.DataFrame:
    """
    Creates random uniform timeseries data

    Parameters
    ----------
    n_times
        The number of time steps
    data_ranges
        The data ranges keyed by variable name. Values define
        the half-open interval [min, max) for each variable.
    start_time
        The first time stamp in the series
    freq
        The time period range frequency
    seed
        The random seed
    iname
        The index name

    Returns
    -------
    sdata
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
