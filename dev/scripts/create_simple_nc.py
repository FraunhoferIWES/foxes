from netCDF4 import Dataset
import numpy as np

ncfile = Dataset("wind_rotation.nc", mode="w", format="NETCDF4")

ncfile.title = "Wind Rotation example"
ncfile.subtitle = (
    "A single wind state with uniform wind speed and spatial wind direction changes"
)
ncfile.author = "IWES"
ncfile.date = "22.06.2021"

s_dim = ncfile.createDimension("state", 2)
h_dim = ncfile.createDimension("h", 2)
y_dim = ncfile.createDimension("y", 2)
x_dim = ncfile.createDimension("x", 2)

s = ncfile.createVariable("state", np.int32, ("state",))
s[:] = [0, 1]

h = ncfile.createVariable(
    "h", np.float32, ("h",)
)  # , zlib=True, least_significant_digit=2)
h.units = "m"
h.long_name = "Height"
h[:] = [0.0, 300.0]

y = ncfile.createVariable(
    "y", np.float32, ("y",)
)  # , zlib=True, least_significant_digit=2)
y.units = "m"
y[:] = [0.0, 2500.0]

x = ncfile.createVariable(
    "x", np.float32, ("x",)
)  # , zlib=True, least_significant_digit=2)
x.units = "m"
x[:] = [0.0, 2500.0]

ws = ncfile.createVariable(
    "ws", np.float32, ("state", "h", "y", "x")
)  # , zlib=True, least_significant_digit=2)
ws.units = "m/s"
ws.long_name = "Wind speed"
ws[:] = 9.0

wd = ncfile.createVariable(
    "wd", np.float32, ("state", "h", "y", "x")
)  # , zlib=True, least_significant_digit=2)
wd.units = "deg"
wd.long_name = "Wind direction"
wd[0, :, 0, 0] = 180.0
wd[0, :, 1, 0] = 220.0
wd[0, :, 1, 1] = 250.0
wd[0, :, 0, 1] = 270.0
wd[1, :, 0, 0] = 0.0
wd[1, :, 1, 0] = 30.0
wd[1, :, 1, 1] = 90.0
wd[1, :, 0, 1] = 120.0

print(ncfile)

# close the Dataset.
ncfile.close()
