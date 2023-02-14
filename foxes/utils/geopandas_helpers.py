import ast
import numpy as np
import pandas as pd
import argparse
from pathlib import Path

try:
    import geopandas as gpd
    IMPORT_OK = True
except ImportError:
    gpd = None
    IMPORT_OK = False

def check_import():
    """
    Checks if library import worked,
    raises error otherwise.
    """
    if not IMPORT_OK:
        print("\n\nFailed to import geopandas. Please install, either via pip:\n")
        print("  pip install geopandas\n")
        print("or via conda:\n")
        print("  conda install -c conda-forge geopandas\n")
        raise ImportError("Failed to import geopandas")

def read_shp(fname, **kwargs):
    """
    Read a shape file

    Parameters
    ----------
    fname : str
        Path to the .shp file
    kwargs : dict, optional
        Additional parameters for geopandas.read_file()
    
    Returns
    -------
    data : geopandas.GeoDataFrame
        The data frame in WSG84

    """
    check_import()
    gpdf=gpd.read_file(fname, **kwargs)
    return gpdf.to_crs("EPSG:4326") # Convert to WGS84

def shp2csv(ifile, ofile, in_kwargs={}, out_kwargs={}, verbosity=1):
    """
    Read shape file, write csv file

    Parameters
    ----------
    iname : str
        Path to the input .shp file
    oname : str
        Path to the output .csv file
    in_kwargs : dict
        Additional parameters for geopandas.read_file()
    out_kwargs : dict
        Additional parameters for geopandas to_csv()
    verbosity : int
        The verbosity level, 0 = silent

    """
    if verbosity > 0:
        print("Reading file", ifile)

    gpdf = read_shp(ifile, **in_kwargs)
    
    if verbosity > 0:
        print("Writing file", ofile)
    
    gpdf.to_csv(ofile, **out_kwargs)

    return gpdf

def read_shp_polygons(fname, names=None, name_col="Name", geom_col=None):
    """
    Reads the polygon points from csv or shp file.
    
    The points are read from geom_col,
    containing string entries like

    POLYGON ((6.415 54.086, 6.568 54.089, ...))

    or

    MULTIPOLYGON (((7.170 55.235, ...)), ((...)), ...)

    Parameters
    ----------
    fn: str
        Path to the .csv or .shp file
    names: list: of str, optinal
        The names of the polygons to be extracted. All by
        default
    name_col: int
        Column that contains the area names
    geom_col : str, optional
        The geometry column, default 'geom' or 'geometry'

    Returns
    -------
    dict:
        Dict with list of array of points. Key: area name,
        Value: list:np.ndarray, shape of latter: (n_points, 2)

    """
    if isinstance(fname, pd.DataFrame):
        pdf = fname.reset_index()
    else:
        fn = Path(fname)
        if fn.suffix == ".shp":
            pdf = pd.DataFrame(read_shp(fn)).reset_index()
        else:
            pdf = pd.read_csv(fn)

    pdf.set_index(name_col, inplace=True)
    if geom_col is None:
        geom_col = "geom" if "geom" in pdf.columns else "geometry"
    pdf[geom_col] = [str(g) for g in pdf[geom_col]]

    out = {}
    names = pdf.index.tolist() if names is None else names
    for name in names:

        if not name in pdf.index:
            raise KeyError(f"Name '{name}' not found in file '{fn}'. Names: {pdf.index.tolist()}")

        a = pdf.loc[name, geom_col]

        multi = "MULTIPOLYGON" in a

        a = a.replace("MULTIPOLYGON", "").replace("POLYGON", "")
        a = a.replace(", ", ",")
        a = a.lstrip().rstrip()

        a = ";".join([ w.replace(" ", ",") for w in a.split(",") ] )
        a = a.replace(");(", "),(").replace(";", "),(")
        data = ast.literal_eval(a)
        if not multi:
            data = [data]

        out[name] = [np.array(d) for d in  data]
    
    return out

if __name__ == "__main__":

    # define arguments and options:
    parser = argparse.ArgumentParser()
    parser.add_argument("shp_file", help="The input .shp file")
    args = parser.parse_args()

    polys = read_shp_polygons(args.shp_file)
    print(polys)
