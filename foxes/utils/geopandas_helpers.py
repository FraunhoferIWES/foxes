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

def read_shp_polygons(fname, names, index_col=1):
    """
    Reads the polygon points from csv or shp file.
    
    The points are read from column 'geom',
    containing string entries like

    POLYGON ((6.415 54.086, 6.568 54.089, ...))

    or

    MULTIPOLYGON (((7.170 55.235, ...)), ((...)), ...)

    Parameters
    ----------
    fn: str
        Path to the .csv or .shp file
    names: list:str
        The names of the polygons to be extracted
    index_col: int
        Index of the column that contains the area names

    Returns
    -------
    dict:
        Dict with list of array of points. Key: area name,
        Value: list:np.ndarray, shape of latter: (n_points, 2)

    """
    if isinstance(fname, pd.DataFrame):
        pdf = fname
    else:
        fn = Path(fname)
        if fn.suffix == ".shp":
            pdf = pd.DataFrame(read_shp(fn)).rename(columns={"geometry": "geom"})
        else:
            pdf = pd.read_csv(fn, index_col=index_col)

    out = {}
    for name in names:

        if not name in pdf.index:
            raise KeyError(f"Name '{name}' not found in file '{fn}'. Names: {list(pdf.index)}")

        a = pdf[pdf.index==name].geom.values[0]

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
