import argparse

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

if __name__ == "__main__":

    # define arguments and options:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="The input .shp file")
    parser.add_argument("outname", help="The output file name")
    args = parser.parse_args()

    gpdf = shp2csv(args.input, args.output)
    print(gpdf)

