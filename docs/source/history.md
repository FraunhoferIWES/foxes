# History

## v0.1.0-alpha

This is the initial release of **foxes** - ready for testing.

So far not many models have been transferred from the Fraunhofer IWES in-house predecessor *flappy*, they will be added in the following versions. Also optimization is not yet included. We are just getting started here!

Enjoy - we are awaiting comments and issues, thanks for testing.

**Full Changelog**: [](https://github.com/FraunhoferIWES/foxes/commits/v0.1.0)

## v0.1.1-alpha

- New code style, created by *black*
- Small fixes, discovered by *flake8*
- Tests now via *pytest* instead of *unittest*
- Introducing github workflow for automatic testing

**Full Changelog**: [](https://github.com/FraunhoferIWES/foxes/commits/v0.1.1)

## v0.1.2-alpha

- Adding support for Python 3.9, 3.10

**Full Changelog**: [](https://github.com/FraunhoferIWES/foxes/commits/v0.1.2)

## v0.1.3-alpha

- Adding docu: [](https://fraunhoferiwes.github.io/foxes.docs/index.html)

**Full Changelog**: [](https://github.com/FraunhoferIWES/foxes/commits/v0.1.3)

## v0.1.4-alpha

- Fixes
  - Static data: Adding missing data `wind_rotation.nc` to manifest
- Models
  - New wake model added: `TurbOParkWake` from Orsted
  - New turbine type added: `PCtSingleFiles`, reads power and thrust curves from two separate files
  - New turbulence intensity wake model added: `IECTI2019`/`Frandsen` and `IECTI2005`

**Full Changelog**: [](https://github.com/FraunhoferIWES/foxes/commits/v0.1.4)

## v0.2.0-alpha

- Core
  - Introducing the concept of runners
- Opt
  - New sub package: `foxes.opt`, install by `pip install foxes[opt]`. This introduces the capability to run wind farm optimizations. As examples layout optimization for simple cases are given, see below.
- Models
  - New turbine model: `Thrust2Ct`, calculates ct from thrust values
  - New turbine type: `NullType`, a turbine type with only rotor diameter and hub height data
  - Renamed `PCtSingleFiles` into `PCtTwoFiles`
- Output
  - New output: `WindRoseOutput`, providing a plotly figure that shows a rose-type histogram
  - New output: `AmbientWindRoseOutput`, providing rose-type plotly figures of ambient data (no wake calculation involved)
- Algorithms
  - Improved `Downwind`: Now with option for ambient runs (no wakes)
- Utils
  - New utility: `show_plotly_fig`, opens a window that shows a plotly figure (instead of browser)
  - New runners: `DefaultRunner`, `DaskRunner`. The latter features parallel runs via dask
- Examples
  - Introducing two sub-folders of examples: `foxes` and `foxes.opt`
  - New example: `wind_rose`, calculation of wind rose states data
  - New example: `layout_single_state`, wind farm layout optimization for a single wind state
  - New example: `layout_wind_rose`, wind farm layout optimization for wind rose states
- Notebooks
  - New notebook: `layout_opt.ipynb`, demonstrating a simple layout optimization case

**Full Changelog**: [](https://github.com/FraunhoferIWES/foxes/commits/v0.2.0)

## v0.2.1-alpha

- Input:
  - New input: `MultiHeightStates` and `MultiHeightTimeseries`, for input data at several height levels, e.g. WRF results at one point
- Output:
  - New output: `FarmResultsEval`, calculates sum, mean, min, max over states or turbines for the whole wind farm. Also calculates capacity, efficiency, yield, P75, P90.
  - New output: `ResultsWriter`, writes farm results or pandas data to csv file
  - Renaming: `AmbientRosePlotOutput` is now called `StatesRosePlotOutput`
- Notebooks:
  - New notebook: `multi_height.ipynb`, demonstrating the usage of multi-height wind input data

**Full Changelog**: [](https://github.com/FraunhoferIWES/foxes/commits/v0.2.1)

## v0.2.2-alpha

- Bug fixes
  - Bug fixed in `MultiHeightStates` for wind veer cases

**Full Changelog**: [](https://github.com/FraunhoferIWES/foxes/commits/v0.2.2)

## v0.2.3-alpha

- Input states
  - Improved: `SingleStateStates` now supports profiles
- Turbine models
  - New model `PowerMask`: Derates or boosts a turbine
- Examples
  - New: `power_mask`, demonstrates derating and boost
- Utils
  - New: `cubic_roots`, solves a cubic equation
- Output
  - New: `StateTurbineMap`, creates heat maps for state-turbine data
  - New: `TurbineTypeCurves`, creates power and thrust curve plots
  - Improved: `FarmLayoutOutput` now supports scatter color by variable
- Documentation
  - Adding forgotten `foxes.opt` to API
- Notebooks:
  - Now including results as colored layout plots
  - New notebook: `wind_rose.ipynb`, demonstrating how to calculate wind roses
  - New notebook: `power_mask.ipynb`, showing derating and boost via a power mask

**Full Changelog**: [](https://github.com/FraunhoferIWES/foxes/commits/v0.2.3)

## v0.2.4-alpha

- Bug fixes
  - Hotfix for bug in `TurbineTypeCurves` output

**Full Changelog**: [](https://github.com/FraunhoferIWES/foxes/commits/v0.2.4)

## v0.2.5-alpha

- Core:
  - New: `ExtendedStates`, allows for adding point data models to states
- Input:
  - Improved: `FieldDataNC`, now with support for spatially uniform variables
- Turbine types:
  - New: `WsRho2PCtTwoFiles`, air density dependent power and ct curves
- Turbine models:
  - New: `SectorManagement`, sets variables by range rules on other variables
- Point models:
  - New: `SetUniformData`, set uniform variables (optionally state dependent)
- Examples:
  - New: `sector_management`, demonstrates how to model wind sector management
- Notebooks:
  - New: `sector_man.ipynb`, demonstrates how to model wind sector management
  - New: `data.ipynb`, lists and shows the static data
- Bug fixes:
  - Fix for bug with option `col_models` in farm layout from csv

**Full Changelog**: [](https://github.com/FraunhoferIWES/foxes/commits/v0.2.5)

## v0.2.6-alpha

- Output:
  - Improved: `FarmResultsEval` can now handle NaN data in Einstein sums
  - Improved: `ResultsWriter` can now also write turbine names 
- Turbine models:
  - New: `TableFactors`, multiplies variables with data from a two dimensional table
  - New: `YAW2YAWM`, calculates yaw deltas from absolute yaw and wind direction
  - New `YAWM2YAW`, calculates absolute yaw from yaw deltas and wind direction
  - Generalized: `kTI`, now includes optional offset `kb`
- Turbine types:
  - Changed: Now yaw correction of `P` and `CT` switched on by default
- Point models:
  - New: `TKE2TI`, calculates turbulence intensity from TKE
- Wake models:
  - New: `PorteAgel`, calculates wakes based on the Bastankhah PorteAgel 2016 wake model
- Wake frames:
  - New: `YawedWake`, bends wakes in yawed conditions
- Wake superposition models:
  - Improved: `LinearSuperposition`, now includes options for lower/higher limits of total wake deltas
- Examples:
  - New: `compare_wakes`, compares wake models along horizontal lines
  - New: `yawed_wake`, demonstrates wake bending by yawing a rotor
- Notebooks:
 - New: `yawed_wake.ipynb`, demonstrates wake bending by yawing a rotor
- Bug fixes:
  - Fix for bug with `ExtendedStates`, now it is actually working
  - Fix for bug with wake width in `CrespoHernandezTIWake`
  - Fix for bug with YAW and WD when using the `YAWM2YAW` turbine model
  - Fix for bug in `TurbOPark` wake model, double counting constant offset in sigma

**Full Changelog**: [](https://github.com/FraunhoferIWES/foxes/commits/v0.2.6)

