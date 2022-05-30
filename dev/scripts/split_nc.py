import xarray as xr
from netCDF4 import Dataset
import numpy as np

data = xr.load_dataset("field_data.nc")

slices = [slice(0,44), slice(44,100), slice(100,144)]
for si, s in enumerate(slices):

    print(f"\nSLICE {s}\n")

    sdata = data.isel(Time=s)
    print(sdata)

    ncfile = Dataset(f'data_{si}.nc',mode='w',format='NETCDF4')

    ncfile.title = "Example field data"
    ncfile.subtitle = f"Part {si}/{len(slices)}"

    dim_Time   = ncfile.createDimension('Time', sdata.sizes['Time']) 
    dim_height = ncfile.createDimension('height', sdata.sizes['height']) 
    dim_UTMY   = ncfile.createDimension('UTMY', sdata.sizes['UTMY']) 
    dim_UTMX   = ncfile.createDimension('UTMX', sdata.sizes['UTMX'])

    var_Time = ncfile.createVariable('Time', str, ('Time',), zlib=True) 
    #var_Time.units = 'm'
    #var_Time.long_name = 'Time stamp'
    var_Time[:] = sdata["Time"].values

    var_height = ncfile.createVariable('height', np.float32, ('height',), zlib=True, least_significant_digit=1) 
    var_height.units = 'm'
    var_height[:] = sdata["height"].values

    var_UTMY = ncfile.createVariable('UTMY', np.float32, ('UTMY',), zlib=True, least_significant_digit=2) 
    var_UTMY.units = 'm'
    var_UTMY[:] = sdata["UTMY"].values - sdata["UTMY"].mean().values

    var_UTMX = ncfile.createVariable('UTMX', np.float32, ('UTMX',), zlib=True, least_significant_digit=2) 
    var_UTMX.units = 'm'
    var_UTMX[:] = sdata["UTMX"].values - sdata["UTMX"].mean().values

    var_WS = ncfile.createVariable('WS', np.float32, ('Time', 'height', 'UTMY', 'UTMX'), zlib=True, least_significant_digit=2) 
    var_WS.units = 'm/s'
    var_WS[:] = sdata["WS"].values

    var_WD = ncfile.createVariable('WD', np.float32, ('Time', 'height', 'UTMY', 'UTMX'), zlib=True, least_significant_digit=2) 
    var_WD.units = 'deg'
    var_WD[:] = sdata["WD"].values

    var_TI = ncfile.createVariable('TI', np.float32, ('Time', 'height', 'UTMY', 'UTMX'), zlib=True, least_significant_digit=3) 
    #var_TI.units = 'm/s'
    var_TI[:] = sdata["TI"].values

    var_RHO = ncfile.createVariable('RHO', np.float32, ('Time', 'UTMY', 'UTMX'), zlib=True, least_significant_digit=2) 
    var_RHO.units = 'kg/m3'
    var_RHO[:] = sdata["RHO"].values
    
    print(ncfile)
