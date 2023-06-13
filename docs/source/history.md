# History

## v0.1.0-alpha

This is the initial release of **foxes** - ready for testing.

So far not many models have been transferred from the Fraunhofer IWES in-house predecessor *flappy*, they will be added in the following versions. Also optimization is not yet included. We are just getting started here!

Enjoy - we are awaiting comments and issues, thanks for testing.

**Full Changelog**: [https://github.com/FraunhoferIWES/foxes/commits/v0.1.0](https://github.com/FraunhoferIWES/foxes/commits/v0.1.0)

## v0.1.1-alpha

- New code style, created by *black*
- Small fixes, discovered by *flake8*
- Tests now via *pytest* instead of *unittest*
- Introducing github workflow for automatic testing

**Full Changelog**: [https://github.com/FraunhoferIWES/foxes/commits/v0.1.1](https://github.com/FraunhoferIWES/foxes/commits/v0.1.1)

## v0.1.2-alpha

- Adding support for Python 3.9, 3.10

**Full Changelog**: [https://github.com/FraunhoferIWES/foxes/commits/v0.1.2](https://github.com/FraunhoferIWES/foxes/commits/v0.1.2)

## v0.1.3-alpha

- Adding docu: [https://fraunhoferiwes.github.io/foxes.docs/index.html](https://fraunhoferiwes.github.io/foxes.docs/index.html)

**Full Changelog**: [https://github.com/FraunhoferIWES/foxes/commits/v0.1.3](https://github.com/FraunhoferIWES/foxes/commits/v0.1.3)

## v0.1.4-alpha

- Fixes
  - Static data: Adding missing data `wind_rotation.nc` to manifest
- Models
  - New wake model added: `TurbOParkWake` from Orsted
  - New turbine type added: `PCtSingleFiles`, reads power and thrust curves from two separate files
  - New turbulence intensity wake model added: `IECTI2019`/`Frandsen` and `IECTI2005`

**Full Changelog**: [https://github.com/FraunhoferIWES/foxes/commits/v0.1.4](https://github.com/FraunhoferIWES/foxes/commits/v0.1.4)

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

**Full Changelog**: [https://github.com/FraunhoferIWES/foxes/commits/v0.2.0](https://github.com/FraunhoferIWES/foxes/commits/v0.2.0)

## v0.2.1-alpha

- Input:
  - New input: `MultiHeightStates` and `MultiHeightTimeseries`, for input data at several height levels, e.g. WRF results at one point
- Output:
  - New output: `FarmResultsEval`, calculates sum, mean, min, max over states or turbines for the whole wind farm. Also calculates capacity, efficiency, yield, P75, P90.
  - New output: `ResultsWriter`, writes farm results or pandas data to csv file
  - Renaming: `AmbientRosePlotOutput` is now called `StatesRosePlotOutput`
- Notebooks:
  - New notebook: `multi_height.ipynb`, demonstrating the usage of multi-height wind input data

**Full Changelog**: [https://github.com/FraunhoferIWES/foxes/commits/v0.2.1](https://github.com/FraunhoferIWES/foxes/commits/v0.2.1)

## v0.2.2-alpha

- Bug fixes
  - Bug fixed in `MultiHeightStates` for wind veer cases

**Full Changelog**: [https://github.com/FraunhoferIWES/foxes/commits/v0.2.2](https://github.com/FraunhoferIWES/foxes/commits/v0.2.2)

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

**Full Changelog**: [https://github.com/FraunhoferIWES/foxes/commits/v0.2.3](https://github.com/FraunhoferIWES/foxes/commits/v0.2.3)

## v0.2.4-alpha

- Bug fixes
  - Hotfix for bug in `TurbineTypeCurves` output

**Full Changelog**: [https://github.com/FraunhoferIWES/foxes/commits/v0.2.4](https://github.com/FraunhoferIWES/foxes/commits/v0.2.4)

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

**Full Changelog**: [https://github.com/FraunhoferIWES/foxes/commits/v0.2.5](https://github.com/FraunhoferIWES/foxes/commits/v0.2.5)

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

**Full Changelog**: [https://github.com/FraunhoferIWES/foxes/commits/v0.2.6](https://github.com/FraunhoferIWES/foxes/commits/v0.2.6)

## v0.3.0-alpha

- Core:
  - Models now with simplified structure, reduces memory load during calculations
- Algorithms:
  - New: `Iterative`, iterative wind farm calculation until convergence has been reached
- Input:
  - Improved: `FieldDataNC` now accepts xarray Dataset or file pattern str as input
  - New: `ShearedProfile`, Vertical WS profile can be determined with shear exponent
- Output:
  - Improved: `FlowPlots2D` replacing horizontal/vertical --> xy, xz, yz, more intuitive
- Wake models:
  - New: `TurbOParkIX`, integrates wake corrected TI along centreline for wake width sigma.
- Wake frames:
  - Improved: All yawed wake frames now also support centreline data integration
- Notebooks:
  - New: `overview.ipynb`, summarizes the setup steps
- Bug fixes:
  - Fix for bug in `TurbOPark` wake model: Brackets in Eq. (4) were wrong 
  - Fix for bug with long streamlines

**Full Changelog**: [https://github.com/FraunhoferIWES/foxes/commits/v0.3.0](https://github.com/FraunhoferIWES/foxes/commits/v0.3.0)

## v0.3.1-alpha

- Input states:
  - Improved: `FieldDataNC` now supports states subset selection
- Wake frames:
  - Improved: `Streamlines` now with maximal length option
- Turbine models:
  - New: `Calculator`, simple variable calculation via user function
- Opt:
  - Now two abstract problems in core: `FarmOptProblem` and `FarmVarsProblem`, the latter invokes the `SetFarmVars` turbine model 
  - New opt problem: `RegularLayoutOptProblem`, searches for a regular grid layout
  - New opt problem: `GeomRegGrids`, finds regular grid layout by purely geometrical means (no wind farm calculation)
  - New opt problem: `GeomLayout`, turbine positioning based on xy variables, also purely geometrical
  - New opt problem: `GeomLayoutGridded`, a purely geometrical optimization on a gridded background
- Examples:
  - New in `foxes.opt`: Example `layout_regular_grid`, demonstrates regular grid layout optimization
  - New in `foxes.opt`: Example `geom_reggrids`, purely geometrical regular layout optimization
- Utils:
  - New functions for shape file handling: `read_shp`, `shp2csv`, `read_shp_polygons`
  
**Full Changelog**: [https://github.com/FraunhoferIWES/foxes/commits/v0.3.1](https://github.com/FraunhoferIWES/foxes/commits/v0.3.1)

## v0.3.2-alpha

- Bug fixes:
  - Fix for bug in `FarmResultsEval` that affected time range calculations under Windows
  - Bug fixes for `FarmResultsEval` with time series data
    
**Full Changelog**: [https://github.com/FraunhoferIWES/foxes/commits/v0.3.2](https://github.com/FraunhoferIWES/foxes/commits/v0.3.2)

## v0.3.3-alpha

- Utils:
  - Now `geopandas_helpers` can handle interior areas
    
**Full Changelog**: [https://github.com/FraunhoferIWES/foxes/commits/v0.3.3](https://github.com/FraunhoferIWES/foxes/commits/v0.3.3)

## v0.3.4-alpha

- Installation:
  - `foxes` now supports Python 3.11
- Input:
  - New layout input option: `add_from_df`, adding turbines from a pandas DataFrame
  - New interface to [windIO](https://github.com/IEAWindTask37/windIO) case yaml files
- Turbine types:
  - New turbine type `CpCtFile`, reads CP and CT data from file or pandas DataFrame
  - New turbine type `CpCtFromTwo`, reads CP and CT data from two files or pandas DataFrames
  - Improved: Turbine types now calculate `P_nominal` as maximum, if not explicitely given
- Constants:
  - Introducing `P_UNITS`, used in turbine types and output evaluation
- States:
  - Bug fixed in `FieldDataNC` with loading multiple files
- Core:
  - Improved `DataCalcModel`: Now cleaner treatment of runs with `progress_bar=False`. Also now slimmer for `distributed` scheduler

**Full Changelog**: [https://github.com/FraunhoferIWES/foxes/commits/v0.3.4](https://github.com/FraunhoferIWES/foxes/commits/v0.3.4)

## v0.3.5-alpha

- General:
  - Moving identifier-type entries from `foxes.variables` to `foxes.constants`
- Models:
  - New wake superposition model `ProductSuperposition`, computing factorized wake deltas
  - New turbine model: `RotorCentreCalc`, calculates data at rotor centre, irrespective of rotor model
- Bug fixes:
  - Bug fixed that caused problems when restarting the `SectorManagement` turbine model, e.g. for flow plots
- Documentation:
  - Completely new style, fixing issues with incomplete API entries

**Full Changelog**: [https://github.com/FraunhoferIWES/foxes/commits/v0.3.5](https://github.com/FraunhoferIWES/foxes/commits/v0.3.5)

## v0.4.0

- Models:
  - Improved: New option to specify wake growth variable name, such that multiple `kTI` models could be used, resulting in different `k`'s for different wake models
  - New turbine model: `LookupTable`, interpolates data based on a multi-dimensional lookup table
- Utils:
  - Improved `DaskRunner`: Now supports clusters that run the Slurm queueing system
- Examples:
  - New: `timeseries_slurm`, shows how to run foxes on a HPC with Slurm queueing system
- Optimization:
  - Improved: `foxes.opt` is now able to optimize for flow variables (at selected points in space) in addition to turbine variables
- Documentation:
  - Improved API docu, now based on `python-apigen`

**Full Changelog**: [https://github.com/FraunhoferIWES/foxes/commits/v0.4.0](https://github.com/FraunhoferIWES/foxes/commits/v0.4.0)