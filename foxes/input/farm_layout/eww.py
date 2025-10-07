import pandas as pd

from foxes.core import Turbine
from foxes.config import get_input_path
from foxes.models.turbine_types import PCtFile

def add_from_eww(
    farm,
    data_source,
    filter={},
    mbook=None,
    csv_dir=None,
    rho=1.225,
    verbosity=1,
    pct_pars={},
    **turbine_parameters,
):
    """
    Add turbines to wind farm via eww database csv input file.

    Parameters
    ----------
    farm: foxes.WindFarm
        The wind farm
    data_source: str or pandas.DataFrame
        The input csv file or data source
    filter: dict, optional
        A dictionary of filters to apply to the dataframe, e.g. {"wind_farm": ["Farm1", "Farm2"]}
    mbook: foxes.ModelBook, optional
        The model book, only needed if csv_dir is specified
    csv_dir: str, optional
        The csv file directory, containing turbine type data files
    rho: float
        The air density for the turbine types, if csv_dir is given
    verbosity: int
        The verbosity level, 0 = silent
    pct_pars: dict
        Additional parameters for the PCtFile constructor
    turbine_parameters: dict, optional
        Additional parameters are forwarded to the WindFarm.add_turbine().

    Examples
    --------
    Data source format:

    ,wind_farm,oem_manufacturer,latitude,longitude,country,rated_power,rotor_diameter,hub_height,turbine_type,commissioning_date
    0,Aberdeen Offshore Wind Farm,Vestas,57.230095,-1.9742404,United Kingdom,8.4,164.0,108.5,V164-8.4 MW,2018-09
    1,Aberdeen Offshore Wind Farm,Vestas,57.2235827,-2.0127432,United Kingdom,8.4,164.0,108.5,V164-8.4 MW,2018-09
    2,Aberdeen Offshore Wind Farm,Vestas,57.2169301,-2.0055697,United Kingdom,8.4,164.0,108.5,V164-8.4 MW,2018-09
    ...

    :group: input.farm_layout

    """
    assert farm.data_is_lonlat, "Require input_is_lonlat = True in WindFarm constructor"
    
    if isinstance(data_source, pd.DataFrame):
        data = data_source
    else:
        if verbosity:
            print("Reading file", data_source)
        pth = get_input_path(data_source)
        data = pd.read_csv(pth, index_col=0)

    ttypes = data["turbine_type"].unique()
    if csv_dir is not None:
        if mbook is None:
            raise ValueError("Model book must be given if csv_dir is specified")
        csv_dir = get_input_path(csv_dir)
        csv_files = sorted(list(csv_dir.glob("*.csv")))
        csv_names = [f.stem for f in csv_files]
        if verbosity > 0:
            print(f"Reading {len(csv_files)} CSV files from {csv_dir}")
        csv_map = []
        ntt = []
        for t in ttypes:
            if t not in csv_names:
                t = t.replace(" ", "_")
            if t not in csv_names:
                raise ValueError(f"Turbine type {t} not found in {csv_dir}")
            ntt.append(t)
            csv_map.append(csv_files[csv_names.index(t)])
            if verbosity > 0:
                print(f"  {t} -> {csv_map[-1].name}")
        ttypes = ntt
        for t in ttypes:
            mbook.turbine_types[t] = None

    else:
        if verbosity > 0:
            print("No csv_dir specified, assuming turbine types correspond to model book names")

    if filter is not None:
        for k, v in filter.items():
            assert k in data.columns, f"Column {k} not in data, found {data.columns}"
            for val in v:
                assert val in data[k].values, f"Value '{val}' not found in column '{k}', got {data[k].unique().tolist()}"
            data = data[data[k].isin(v)]
            if verbosity:
                print(f"Filtering {k} for {v}, now {len(data)} turbines")

    farms = []
    tmodels = turbine_parameters.pop("turbine_models", [])
    for i0, i in enumerate(data.index):
        fname = str(data.loc[i, "wind_farm"]).replace(" ", "_")
        if fname not in farms:
            farms.append(fname)
            j = 0

        ttype = data.loc[i, "turbine_type"]
        if csv_dir is not None and mbook.turbine_types[ttype] is None:
            mbook.turbine_types[ttype] = PCtFile(csv_map[i], rho=rho, **pct_pars)
        
        farm.add_turbine(
            Turbine(
                name=f"{fname}_{j}",
                index=i0,
                xy=data.loc[i, ["longitude", "latitude"]].values,
                H=data.loc[i, "hub_height"],
                D=data.loc[i, "rotor_diameter"],
                turbine_models=[ttype] + tmodels,
                **turbine_parameters,
            ),
            verbosity=verbosity,
        )

        j += 1
    
    farm.lock(verbosity=verbosity)
