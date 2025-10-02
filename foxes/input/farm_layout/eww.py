import pandas as pd

from foxes.core import Turbine
from foxes.config import get_input_path
from foxes.models.turbine_types import TBLFile

def add_from_eww(
    farm,
    data_source,
    filter={},
    mbook=None,
    tbl_dir=None,
    rho=1.225,
    verbosity=1,
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
        The model book, only needed if tbl_dir is specified
    tbl_dir: str, optional
        The tbl file directory. Either turbine type names correspond to tbl file names, 
        or the sorted list of tbl files is interpreted in that order.
    rho: float
        The air density for the turbine types, if tbl_dir is given
    verbosity: int
        The verbosity level, 0 = silent
    turbine_parameters: dict, optional
        Additional parameters are forwarded to the WindFarm.add_turbine().

    Examples
    --------
    CSV file format:

    ,wind_farm,oem_manufacturer,latitude,longitude,country,rated_power,rotor_diameter,hub_height,turbine_type,commissioning_date
    0,Aberdeen Offshore Wind Farm,Vestas,57.230095,-1.9742404,United Kingdom,8.4,164.0,108.5,V164-8.4 MW,2018-09
    1,Aberdeen Offshore Wind Farm,Vestas,57.2235827,-2.0127432,United Kingdom,8.4,164.0,108.5,V164-8.4 MW,2018-09
    2,Aberdeen Offshore Wind Farm,Vestas,57.2169301,-2.0055697,United Kingdom,8.4,164.0,108.5,V164-8.4 MW,2018-09
    ...

    :group: input.farm_layout

    """

    if isinstance(data_source, pd.DataFrame):
        data = data_source
    else:
        if verbosity:
            print("Reading file", data_source)
        pth = get_input_path(data_source)
        data = pd.read_csv(pth, index_col=0)

    ttypes = data["turbine_type"].unique()
    if tbl_dir is not None:
        if mbook is None:
            raise ValueError("Model book must be given if tbl_dir is specified")
        tbl_dir = get_input_path(tbl_dir)
        tbl_files = sorted(list(tbl_dir.glob("*.tbl")))
        tbl_names = [f.stem for f in tbl_files]
        if verbosity:
            print(f"Reading {len(tbl_files)} turbine tables from {tbl_dir}")
        tbl_map = []
        if ttypes[0] in tbl_names:
            for t in ttypes:
                if t not in tbl_names:
                    raise ValueError(f"Turbine type {t} not found in {tbl_dir}")
                tbl_map.append(tbl_files[tbl_names.index(t)])
                if verbosity > 0:
                    print(f"  {t} -> {tbl_map[-1].name}")
        else:
            assert len(ttypes) == len(tbl_files), (
                f"Number of turbine types ({len(ttypes)}) does not match "
                f"number of tbl files ({len(tbl_files)}), and turbine types do not correspond "
                "to tbl file names."
            )
            if verbosity:
                print(
                    "Turbine types do not correspond to tbl file names, interpreting sorted list of tbl files in that order"
                )
                for i, t in enumerate(ttypes):
                    print(f"  {t} -> {tbl_files[i].name}")
            tbl_map = tbl_files
        for t in ttypes:
            mbook.turbine_types[t] = None

    else:
        if verbosity:
            print("No tbl_dir specified, assuming turbine types correspond to model book names")

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
        if tbl_dir is not None and mbook.turbine_types[ttype] is None:
            mbook.turbine_types[ttype] = TBLFile(tbl_map[i], rho=rho)
        
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
