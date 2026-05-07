import numpy as np

from .load import import_module

def calc_era5_density(era5,z, var2ncvar={}):
    """
    Returns local air densities at specified height using the 
    surface variables msl, t2m, d2m

    Parameters
    ----------
    era5 : xarray.Dataset
        The era5 data
    z : float
        height at which to calculate density
    var2ncvar : dict, optional
        A dictionary mapping variable names to netCDF variable names.
    
    Returns
    -------
    rho : numpy.array
        The air density in kg/m3
    
    :group: utils
    
    """
    # dynamically import packages:
    mc = import_module("metpy.calc", pip_hint="pip install metpy", conda_hint="conda install -c conda-forge metpy")
    mu = import_module("metpy.units", pip_hint="pip install metpy", conda_hint="conda install -c conda-forge metpy")

    # get column names:
    c_msl = var2ncvar.get("msl", "msl")
    c_t2m = var2ncvar.get("t2m", "t2m")
    c_d2m = var2ncvar.get("d2m", "d2m")

    # calculate pressure at height
    z = z * mu.units.meter
    p0 = np.array(era5[c_msl]) * mu.units.pascal
    pz = p0/mc.height_to_pressure_std(0 * mu.units.meter) * mc.height_to_pressure_std(z)
    t2 = np.array(era5[c_t2m]) * mu.units.K
    d2 = np.array(era5[c_d2m]) * mu.units.K
    
    # calculate temperature at height
    tz = mc.dry_lapse(pressure=pz,temperature=t2,reference_pressure=p0) 
    
    # calculate mixing ratio
    rh = mc.relative_humidity_from_dewpoint(t2 ,d2) # assuming rh constant with height
    m = mc.mixing_ratio_from_relative_humidity(pressure=pz,temperature=tz,relative_humidity=rh) 
    
    # calculate density at height z
    dens = mc.density(pz,tz,m).magnitude

    return dens
