# Changelog

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

## v0.4.1

- Algorithms:
  - Improved: `Iterative` now iterates through outer loop instead of within chunk
- Models:
  - New wake frame: `Timelines`, propagates wakes for spatially uniform time series
- Tests:
  - New test: `consistency/iterative`, checks if iterative algorithm gives same result
- Examples:
  - New example: `timelines`, demonstrates the usage of the `Timelines` wake frame
  - Improved: All examples were update for the correct usage of the `DaskRunner`
- Notebooks:
  - New notebook: `timelines.ipynb`, showing how to use the `Timelines` wake frame in a notebook
- Data:
  - New states data `timeseries_100.csv.gz`, a short timeseries with timestep 1 min, varying wind direction only
- Output:
  - Improved: `FlowPlots2D` now has the optional argument `runner`, for computing plots with the chosen parallelization settings

**Full Changelog**: [https://github.com/FraunhoferIWES/foxes/commits/v0.4.1](https://github.com/FraunhoferIWES/foxes/commits/v0.4.1)

## v0.4.2

- Bug fixes:
  - Fix of bug that slowed down `TurbOParkIX` by considering all TI wakes instead of only source turbine wake during integration
  - Fix of bug that prevented plotly wind roses to be shown in the documentation
  - Fix in docu that excluded the algorithm models from the API
- Output:
  - New: `Animator`, creates animations based on generators that yield lists of artists
- Examples:
  - Improved: `timelines` now includes turbine REWS signal in animations
- Notebooks:
  - Improved: `timelines.ipynb` now includes turbine REWS signal in animations

**Full Changelog**: [https://github.com/FraunhoferIWES/foxes/commits/v0.4.2](https://github.com/FraunhoferIWES/foxes/commits/v0.4.2)

## v0.4.3

- Output:
  - Improved: `RosePlotOutput` and `StatesRosePlotOutput` now optionally return the binned data
- Models:
  - New vertical profile: `DataProfile`, data based profile from file or pandas DataFrame
  - Improved ti superposition: Now supporting n-th power superposition
  - Improved wake model `TurbOParkIX`: New option for consideration of all wakes in ti integral
- Bug fixes:
  - Fixed bug with `windio` input that resulted in wrong wind rose weights
  - Fixed bug with `FlowPlots2D` with value bounds in contour plots

**Full Changelog**: [https://github.com/FraunhoferIWES/foxes/commits/v0.4.3](https://github.com/FraunhoferIWES/foxes/commits/v0.4.3)

## v0.4.4

- Testing automated publishing to PyPi via workflow

## v0.4.5

- Models:
  - New rotor model: `LevelRotor`, calculates the REWS from different height levels
  - New turbine type: `WsTI2PCtFromTwo`, reads turbulence-dependent ct- and power curves

**Full Changelog**: [https://github.com/FraunhoferIWES/foxes/commits/v0.4.5](https://github.com/FraunhoferIWES/foxes/commits/v0.4.5)

## v0.5

- Dependencies:
  - Dropping support for Python 3.7, due to `netcdf4` dependency
- Core:
  - Simplified model initialization/finalization. Removing the `keep_models` idea, instead all models are now kept in the algorithm's idata memory until finalization. Adding a new model now mostly requires that the `sub_models` and the `load_data` functions are overloaded, if applicable. The `initialize` and `finalize` only need to be addressed explicitly in non-standard cases.
- Algorithms:
  - New algorithm: `Sequential`, step wise evaluation of states for simulation environments that do not support chunking
  - Improved: `Iterative` now supports under-relaxation of parameters
- Output:
  - New sub package: `output.flow_plots_2d` is now a package instead of a module
- Models:
  - This version introduces induction models for modelling blockage effects.
  - New induction model: `RHB`, the classic Rankine-half-body model.
- Examples:
  - New example: `sequential`, demonstrating the usage of the sequential algorithm
  - New example: `induction_RHB`, showing the Rankine-half-body model for blockage
- Notebooks:
  - New notebook: `blockage.ipynb`, demonstrating how to apply the RHB induction model to a wind farm

**Full Changelog**: [https://github.com/FraunhoferIWES/foxes/commits/v0.5](https://github.com/FraunhoferIWES/foxes/commits/v0.5)

## v0.5.1

- Dependencies:
  - The `windio` package is now an optional dependency, to be installed by hand if needed. This is due to windio being not available at conda-forge
- Notebooks:
  - New notebook: `sequential.ipynb`, creating an animation showing state-by-state wake propagation

**Full Changelog**: [https://github.com/FraunhoferIWES/foxes/commits/v0.5.1](https://github.com/FraunhoferIWES/foxes/commits/v0.5.1)

## v0.5.2

- Algorithms:
  - Improved `Downwind`: Now optional arguments that allow calculation of subsets in calc_farm and calc_points, e.g. a selection of states
- Output:
  - New output: `PointCalculator`, a wrapper for the calc_points function of the algorithm
  - New output: `SliceData`, creating arrays, DataFrames or Datasets from 2D slices
  - Improved `FlowPlots2D`: Completely refactored, data based on `SliceData`. New option for creating figures for subsets of states only.
- Vertical profiles:
  - Now optional support for `ustar` input data instead of reference data
- Turbine types:
  - Fix for `Cp`-based models with interpolation within sparse input data
- Wake models:
  - Renaming: `BastankhahWake` class now called `Bastankhah2014`. Models in the model book are also renamed from `Bastankhah` to `Bastankhah2014`
  - Renaming: `PorteAgelWake` class now called `Bastankhah2016`. Models in the model book are also renamed from `PorteAgel` to `Bastankhah2016`
  - Renaming: `RHB` class now called `RankineHalfBody`
  - Fix: `RankineHalfBody` no longer shows a jump at the rotor disc, but a small region of constant deficit instead
  - New default values: `Bastankhah2014` now has default value `sbeta_factor=0.2` (previously 0.25). Models with the previous value are available in the model book as `Bastankhah025` etc.
- Wake superpositions:
  - Restructured: Now simplified classes for WS or TI superposition only (less general but simpler), e.g. `WSLinear` or `TIQuadratic`, etc. Also in the model book the models are now called `ws_linear` or `ti_quadratic`, etc.

**Full Changelog**: [https://github.com/FraunhoferIWES/foxes/commits/v0.5.2](https://github.com/FraunhoferIWES/foxes/commits/v0.5.2)

## v0.5.2.1

- Bug fixes:
  - Bugs fixed that messed up the colorbar and the title in animations
- Notebooks:
  - Improved animations in `timelines.ipynb` and `sequential.ipynb`

**Full Changelog**: [https://github.com/FraunhoferIWES/foxes/commits/v0.5.2.1](https://github.com/FraunhoferIWES/foxes/commits/v0.5.2.1)

## v0.6

- Dependencies:
  - Replacing dependency on `plotly` by dependency on `windrose`, since the latter is lighter
- Core:
  - This version introduces `AxialInductionModel` classes, computing the axial induction factor `a(ct)`
- Wake models:
  - Reformulating `beta` in terms of induction in `Bastankhah2014` and `CrespoHernandez`
  - New default: `Bastankhah2014`, `Bastankhah2016`, `TurbOPark` and `TurbOParkIX` now with default axial induction model `Madsen`
  - New induction wake models: `Rathmann`, `SelfSimilar` and `SelfSimilar2020`, for blockage modelling
  - Introducing `WakeMirror` wrapper around wake models, modelling wake reflection from ground or horizontal plane via virtual mirrored turbines
- Wake frames:
  - Renaming `Streamlines` to `Streamlines2D`, no changes in model book names
- Axial induction models:
  - New induction model: `BetzAxialInduction`, the classic `a = 0.5(1 - sqrt(1-ct))` relation. In the model book this is called `Betz`.
  - New induction model: `MadsenAxialInduction`, a third-order polynomial approximation of `a(ct)`. In the model book this is called `Madsen`.
- Output:
  - Improved: `FlowPlots2D` now includes an option for indicating the rotor disk by a colored line
  - Improved: `RosePlotOutput` no longer depends on `plotly`, but on the new utility `TabWindroseAxes`
- Utils:
  - New: `TabWindroseAxes`, a derivative of `windrose.WindroseAxes` for input data that is based on bins with weights (and not timeseries)
- Notebooks:
  - New: `blockage_comparison.ipynb`, comparing four turbine induction models
- Bug fixes:
  - Fix for bug in `Streamlines2D` when used in combination with `WakeMirror`
- Tests:
  - Fresh `flappy` v0.6.2 test data for all Bastankhah and CrespoHernandez wakes, also without the `sbeta` limitation

**Full Changelog**: [https://github.com/FraunhoferIWES/foxes/commits/v0.6](https://github.com/FraunhoferIWES/foxes/commits/v0.6)

## v0.6.1

- Input:
  - New ambient states: `TabStates`, single tab-file input
- Data
  - New static data: `winds100.tab`, an example tab file
- Examples:
  - New example: `tab_file`, demonstrating the usage of the `TabStates`
- Bug fixes:
  - Bug fixed for `RankineHalfBody` turbine induction model that produced wrong results for wind directions unequal 270 degrees

**Full Changelog**: [https://github.com/FraunhoferIWES/foxes/commits/v0.6.1](https://github.com/FraunhoferIWES/foxes/commits/v0.6.1)

## v0.6.2

- Models:
  - New partial wakes model: `PartialCentre`, evaluating wake deltas at the rotor centre point only
- Inputs:
  - New farm layout input: `add_random`, adds turbines at random positions, respecting a minimal distance
  - New states creation: `random_timseries_data`, creates uniform random timeseries data
- Utils:
  - New function `random_xy_square`, generates random xy positions with minimal distance
- Examples:
  - New example: `random_timeseries`, computes a random farm in a random timeseries. Both sizes are defined by user input
- Bug fixes:
  - Fix for bug in `gen_states_fig_xz` and `gen_states_fig_xz` with parameter `x_direction`, which had no effect on the image

**Full Changelog**: [https://github.com/FraunhoferIWES/foxes/commits/v0.6.2](https://github.com/FraunhoferIWES/foxes/commits/v0.6.2)

## v0.7

- Python versions:
  - Now supporting Python 3.12
- Documentation:
  - New page `Models`, summarizing and explaining the *foxes* model choices.
- Core:
  - Speed-up in comparison with version 0.6.x, by internally handling all turbine data in downwind order, and then translating it back to farm order once computations are complete.
  - Internally, all point evaluation data is now translated into so called "target" data, where each target is understood as being composed of a certain number of target points. During wake computations, these are the points per rotor (as defined by partial wakes models). For computations at user given points, the points are interpreted as targets with a single target point each. Final point output data is then given to the user again with point index coodinates, i.e., in the same format as in previous versions.
  - New data classes: `MData`, `FData`, `TData`, all derived from the foxes `Data` class. These specialize model, farm and target data, respectively, during model calculations.
- Algorithms:
  - All algorithm constructors now take `farm, states, wake_models` as the first three arguments. If no model book is given, the default `ModelBook()` will be used.
  - Partial wakes are now chosen either 
    - by a dictionary, which maps wake model names to model choices (or default choices, if not found),
    - or by a list, where the mapping to the wake models is in order of appearance, 
    - or by a string, in which case all models are either mapped to the given model, or, if that fails with `TypeError`, to their defaults,
    - or by `None`, which means all models are mapped to the default choice.
- Partial wakes:
  - New `PartialSegregated` abstract model, from which the `PartialGrid` model is derived. Segregated models now average background results and wake deltas separatly, and then add the results. Notice that with the choice of `RotorPoints` partial wakes, the mathematically correct average over a discretized rotor is calculated. This is more accurate, but it may be slower than some models (e.g. for `PartialAxiWake` models) or not applicable for some rotor choices (e.g. the `LevelRotor`, where a wake average makes no sense). 
- Outputs:
  - New output `RotorPointPlot`, creating rotor point plots.
- Notebooks:
  - New notebook `rotor_models.ipynb`, visualizing rotor points.
  - New notebook `partial_wakes.ipynb`, verifying partial wakes models.
- Bug fixes:
  - Various fixes here and there.

**Full Changelog**: [https://github.com/FraunhoferIWES/foxes/commits/v0.7](https://github.com/FraunhoferIWES/foxes/commits/v0.7)

## v0.7.1

- Models:
  - Improved `ModelBook`, adding some smartness thanks to model factories. Parameters are parsed from model name strings that follow given patterns. E.g., `Jensen_<superposition>_k<k>` represents all `JensenWake` models with any superposition model choice and and choice of k, and `Jensen_linear_k0075` would be an accepted name choice by the user.
  - New wind speed superpositions `WSLinearLocal`, `WSQuadraticLocal`, `WSPowLocal`, `WSMaxLocal`: Adding dimensionless wind deficits, and then evaluating the overall effect for the ambient results at evaluation points (no scaling with rotor effective data)
- Utils:
  - New utility `Factory`, creating class instances from selections of allowed parameter choises
  - New utility `FDict`, a dictionary that supports factories
- Bug fixes:
  - Bug fixed with `TurbOParkIX`, that prevented it from running
  
**Full Changelog**: [https://github.com/FraunhoferIWES/foxes/commits/v0.7.1](https://github.com/FraunhoferIWES/foxes/commits/v0.7.1)

## v0.7.2

- Bug fixes:
  - Fix for bug with `Factory` that confused templates `A_B<..>` type with `B<..>` type templates 
  
**Full Changelog**: [https://github.com/FraunhoferIWES/foxes/commits/v0.7.2](https://github.com/FraunhoferIWES/foxes/commits/v0.7.2)

## v0.7.3

- Dependencies:
  - Restriction to `numpy<2` due to an incompatibility of dependecy `autograd`
- Core:
  - New model type `GroundModel`, which manages the call of the partial wakes models in case of farm calculations and wake models in case of point calculations
- Inputs:
  - New height dependent states: `MultiHeightNCStates` and `MultiHeightNCTimeseries`, same as `MultiHeightStates` and `MultiHeightTimeseries`, but starting from a netCDF file or `xarray.Dataset` input
- Models:
  - New wake model helper class `WakeK`, handling the `k`, `ka`, `kb` choices for all wake models
  - New ground model `NoGround`, plain call of the partial wakes and wakes models
  - New ground models `WakeMirror` and  `GroundMirror`, replacing the equivalent former wake models. Realizing wake reflection at horizontal planes.
  - New induction model `VortexSheet`, which is a radial implementation of the centreline deficit model in Medici 2012 https://doi.org/10.1002/we.451
- Utils:
  - New utility `WakeKFactory` class, enabling the choice of either `k` or `ka, kb` directly from the wake model names in the model book.
- Inputs:
  - Work on `windio`, but unfinished on their side when it comes to analysis requests
- Examples:
  - Example `multi_height`: Now based on `MultiHeightNCTimeseries`
- Bug fixes:
  - Fox for bug with wake mirrors and partial wakes

  
**Full Changelog**: [https://github.com/FraunhoferIWES/foxes/commits/v0.7.3](https://github.com/FraunhoferIWES/foxes/commits/v0.7.3)

## v0.7.4

- Input:
  - Adding output options to `windio`
- Output:
  - New output `StateTurbineTable`, exporting state-turbine data to NetCDF
  
**Full Changelog**: [https://github.com/FraunhoferIWES/foxes/commits/v0.7.4](https://github.com/FraunhoferIWES/foxes/commits/v0.7.4)

## v0.8

Starting with this version, the previous `foxes.opt` sub-package is is now its own package at github, pypi and conda-forge: [foxes-opt](https://github.com/FraunhoferIWES/foxes-opt), with its own [documentation](https://fraunhoferiwes.github.io/foxes-opt/index.html).

If you are planning to run wind farm optimizations, install it via

```console
pip install foxes[opt]
```

or

```console
pip install foxes-opt
```

or

```console
conda install foxes-opt -c conda-forge
```

Compared to older versions, replace `foxes.opt` by `foxes_opt` in all your scripts - then everything should just run as before.

If you are not running any optimizations, just don't do any of the above and enjoy the lighter version with less dependencies.

**Full Changelog**: [https://github.com/FraunhoferIWES/foxes/commits/v0.8](https://github.com/FraunhoferIWES/foxes/commits/v0.8)

## v0.8.1

- Updated requirements.txt

**Full Changelog**: [https://github.com/FraunhoferIWES/foxes/commits/v0.8.1](https://github.com/FraunhoferIWES/foxes/commits/v0.8.1)

## v0.8.2

- Removing `plotly_helpers.py` from `utils`
- Updating dependencies

**Full Changelog**: [https://github.com/FraunhoferIWES/foxes/commits/v0.8.2](https://github.com/FraunhoferIWES/foxes/commits/v0.8.2)

## v0.8.3

- Outputs:
  - Improved `SliceData` output: Now either specify `resolution` or `n_img_points`, e.g. `n_img_points=(100, 100)` for an image with 100 x 100 points

**Full Changelog**: [https://github.com/FraunhoferIWES/foxes/commits/v0.8.3](https://github.com/FraunhoferIWES/foxes/commits/v0.8.3)

## v1.0

This major version introduces the concept of `Engines` which handle the chunking and parallelization of all *foxes* calculations. The default choice now prefers the [concurrent.futures](https://docs.python.org/3/library/concurrent.futures.html) package and provides a significant speedup compared to previous versions. See the documentation for more details and all engine choices. The `Engines` replace the `Runners` of previous versions.

- Engines
  - New engine `ThreadsEngine` (short `threads`): Sends chunks to threads, based on `concurrent.futures`
  - New engine `ProcessEngine` (short `process`): Sends chunks to processes, based on `concurrent.futures`
  - New engine `MultiprocessEngine` (short `multiprocess`): Sends chunks to a multiprocessing pool
  - New engine `XArrayEngine` (short `xarray`): Runs parallelization via [xarray.apply_ufunc](https://docs.xarray.dev/en/stable/generated/xarray.apply_ufunc.html)
  - New engine `DaskEngine` (short `dask`): Submits chunk calculation functions to `dask`
  - New engine `LocalClusterEngine` (short `local_cluster`): Creates a virtual cluster on the local machine
  - New engine `MPIEngine` (short `mpi`): Parallel computation via `mpi4py`, also multi-node
  - New engine `SlurmClusterEngine` (short `slurm_cluster`): Submits jobs to a SLURM system
  - New engine `NumpyEngine` (short `numpy`): Runs a loop over chunks
  - New engine `SingleChunkEngine` (short `single`): Runs single-chunk calculations
  - New engine `DefaultEngine` (short `default`): Switches between `single` and `process`, depending on the case size
- Inputs:
  - New states `OnePointFlowStates`, `OnePointFlowTimeseries`, `OnePointFlowMultiHeightTimeseries`, `OnePointFlowMultiHeightNCTimeseries`: Generating horizontally inhomogeneous inflow from horizontally homogeneous input data
  - New farm layout option: `add_ring`, adding a ring of turbines
- Models:
  - Wake frame `Timelines` now also accept spatially uniform multi-height states 
  - New wake frame `DynamicWakes`: Dynamic wakes for any kind of timeseries states, compatible with chunking
  - New turbine type `FromLookupTable`, computes power and thrust coefficient from a lookup table
- Outputs:
  - New sub package `seq_plugins`, in case more of these will be added in the future
  - New sequential plugin `SeqWakeDebugPlugin`, adding wake centres and velocity vectors to flow animations, for debugging
- Examples:
  - New example: `dyn_wakes`, similar to `timelines` but with dynamic wakes and `OnePointFlowTimeseries` inflow

**Full Changelog**: [https://github.com/FraunhoferIWES/foxes/commits/v1.0](https://github.com/FraunhoferIWES/foxes/commits/v1.0)

## v1.1

- Python versions:
  - Adding support for Python3.13
- Engines:
  - Default number of processes is now os.cpu_count(), or os.process_cpu_count() for python3.13
  - New engine: `RayEngine` (short name `ray`), runs parallel computations based on  the [Ray package](https://docs.ray.io/en/latest/)
- Inputs:
  - New states `SliceDataNC`, much like `FieldDataNC` but without height dependency
- Models:
  - New turbine type `TBLFile`: Reads power, ct, D, H, P_rated from a *.tbl file
  - Turbine induction models `SelfSimilar`, `SelfSimilar2020`, `Rathmann`, `VortexSheet` now optionally accept any wind speed superposition model, i.e., they are no longer based on hard-coded linear superposition
- Bug fixes:
  - Bug fixed in `WSProduct`, causing zero wind speed at regions not touched by wakes
  - Bugs fixes in `FarmLayoutOutput`, concerning the writing of the layout csv file
  
**Full Changelog**: [https://github.com/FraunhoferIWES/foxes/commits/v1.1](https://github.com/FraunhoferIWES/foxes/commits/v1.1)

## v1.1.1

- Dependencies:
  - Removing optional dependencies `io`. The installation advice will be printed when trying to use the `foxes.input.windio` sub package. The reason is that for now this depends on a personal [fork by kilojoules](https://github.com/kilojoules/windIO), and the inclusion of such dependencies is not supported by PyPi.
  
**Full Changelog**: [https://github.com/FraunhoferIWES/foxes/commits/v1.1.1](https://github.com/FraunhoferIWES/foxes/commits/v1.1.1)

## v1.2

- Command line applications:
  - New command line application: `foxes_yaml`, runs *foxes* by interpreting a `yaml` input parameter file (no script neccessary)
  - New command line application: `foxes_windio`, runs *foxes* by interpreting a windio `yaml` input parameter file (no script neccessary)
- Config:
  - Introducing a dictionary-like configuration object `foxes.config`, setting default data types, the directory choices, etc.
- Inputs:
  - New input package `foxes.input.yaml`, realizing the run via yaml parameter files. This is not following `windio`, but a *foxes* specific structure.
  - Reworked `windio` input package, now at `foxes.input.yaml.windio` and based on `foxes.input.yaml`
- Outputs:
  - Introducing the idea of a default output directory, accessible via `foxes.config.out_dir`. The file writing output classes now refer to this, or a output specific output directory that is given as an argument to the constructor of the output class.
- Examples:
  - New example `yaml_input`, demonstrating the command line application `foxes_yaml` and the new yaml parameter file
- Bug fixes:
  - Fix for `TurbOParkIX` wake model: Now `k` is truly `k`, such that only `ambka` or `ka` choices from `WakeK` correspond to multiplication with `AMB_TI` or `TI`, respectively.
  
**Full Changelog**: [https://github.com/FraunhoferIWES/foxes/commits/v1.2](https://github.com/FraunhoferIWES/foxes/commits/v1.2)

## v1.2.1

- Dependencies:
  - Removing dependency on `windrose` package
- Outputs:
  - Output `FarmResultsEval`: Renaming contraction rule `mean` into `weights`. Introducing new rule `mean_no_weights`.
  - New Output `WindRoseBinPlot`, for example showing variable mean values for wind rose bins
  - Re-implementation of `WindRosePlot`, no longer needs the `windrose` package
- Inputs:
  - Generalizing `ScanWS` states to `ScanStates`, for arbitrary tensor products of variable lists
  - States `FieldDataNC` now also accepts fields without heights, signature (state, x, y) and (state, y, x)
- Bug fixes:
  - Fixing some examples that were computing the turbine efficiency incorrectly

**Full Changelog**: [https://github.com/FraunhoferIWES/foxes/commits/v1.2.1](https://github.com/FraunhoferIWES/foxes/commits/v1.2.1)

## v1.2.2

- Developments required by `foxes_opt` v0.3, related to running also wind farm optimizations by yaml parameter files

**Full Changelog**: [https://github.com/FraunhoferIWES/foxes/commits/v1.2.2](https://github.com/FraunhoferIWES/foxes/commits/v1.2.2)

## v1.2.3

- Dependencies:
  - Switching `windIO` support to fork at [https://github.com/EUFLOW/windIO](https://github.com/EUFLOW/windIO)
  - Removing dependency `netcdf4`, since this or `h5netcdf` are now a dynamic choice, called when NetCDF files are being used
- Config:
  - Now three directories are defined: `work_dir`, `input_dir`, `output_dir`. If not defined, falling back to `work_dir`, defaulting to cwd.
- Inputs:
  - Removing `SliceDataNC`, since now `FieldDataNC` can handle data without height dependency
  - Enhancing `FieldDataNC` for more efficient parallel loading of large data, replacing `pre_load` flag with load mode options `preload`, `lazy`, `fly`
  - Enhancing `windio` inputs for output generation
- Outputs:
  - New output `SlicesData` which combines slices from `SliceData` into a single NetCDF file, resulting in full 3d field output (or 4d, with time/states dimension)
- Models:
  - New point model `Ustar2TI`, estimates TI from friction velocity

**Full Changelog**: [https://github.com/FraunhoferIWES/foxes/commits/v1.2.3](https://github.com/FraunhoferIWES/foxes/commits/v1.2.3)

## v1.2.4

- Config:
  - Fix for bug with `nc_engine` for Python 3.8
- Utils:
  - New util `ReaderWRG`, reading WRG files. Thanks @HassanKassem

**Full Changelog**: [https://github.com/FraunhoferIWES/foxes/commits/v1.2.4](https://github.com/FraunhoferIWES/foxes/commits/v1.2.4)

## v1.2.5

- Models:
  - Fix for bug in `VortexSheet` turbine induction wake model. Thanks @MellBeli

**Full Changelog**: [https://github.com/FraunhoferIWES/foxes/commits/v1.2.5](https://github.com/FraunhoferIWES/foxes/commits/v1.2.5)

## v1.3

- Core:
  - The statistical weights of the states are now no longer required to be known a-priori, but they are computed at target points during the states calculation. So in principle, weights can now depend on the evaluation point, as it is required by the new `WRGStates`, see below.
- Input:
  - Introducing `WRGStates`, reading and running Wind Resource Grid (WRG) data files which describe Weibull factors on a regular horizontal grid.
- Examples:
  - New example `quickstart`, demonstrating _foxes_ in a minimal script
- Bug fixes:
  - Bug fixed with `xarray.Dataset` input instead of file for `FieldDataNC` states

**Full Changelog**: [https://github.com/FraunhoferIWES/foxes/commits/v1.3](https://github.com/FraunhoferIWES/foxes/commits/v1.3)

## v1.4

- Core:
  - This version introduces `WakeDeflection` models. This replaces and generalizes the `YawedWakes` wake frame, such that now any choice of wake frame can be combined with yawed wakes, for example streamlines or dynamic wakes. The wake deflection model is selected through the parameter `wake_deflection` or the algorithm and then applies to all wakes.
  - Wake models, wake deflections or wake superpositions can now induce a horizontal rotation of the waked wind vector. This is realized by inhternally switching from a `(WS, WD)` description to a `UV` vector, and vice-versa at the end of calculations.
  - A new abstract class `WindVectorWakeSuperposition` was instroduced for the purpose of handling `UV` wind wake effects.
- Inputs:
  - New states `WeibullSectors`, reading and interpreting spatially homogeneous Weibull scale and shape parameters for wind direction sectors, from csv or nc files
- Models:
  - New wake deflection model `Bastankhah2016Deflection`, replacing the previous `YawedWakes` wake frame. Notice that this model does not modify the wind speed deficit or the wind direction in the wake, it only changes the wake path. Meant to be used together with the `Bastankhah2016` wake model.
  - New wake deflection model `JimenezDeflection`, which bends the wake and optionally also rotates the waked wind vector along the path. If that option is chosen, this model requires a vector type superposition of the wind deficit.
  - New wake deflection model `NoDeflection`, which switches off wake bending and should be used if no yaw misalignment occurs in the simulation (default).
  - New wake superposition model `WindVectorLinear`, realizing linear vector type wind vector superposition.

**Full Changelog**: [https://github.com/FraunhoferIWES/foxes/commits/v1.4](https://github.com/FraunhoferIWES/foxes/commits/v1.4)

## v1.5

- Dependencies:
  - Introducing minimal package versions of dependencies to pyproject.toml
- Inputs:
  - New abstract states class `DatasetStates`, serving as a common base for NetCDF type input data
  - Renamed `FieldDataNC` as `FieldData`, now with improved data handling
  - New states class `PointCloudData`, interpolating data with irregular point cloud support
  - New states class `WeibullField`, representing Weibull sector data on a regular grid
  - New states class `WeibullPointCloud`, representing Weibull sector data with irregular point cloud support
  - New farm layout input function `add_from_arrays`, for directly adding turbines through one dimensional data lists or arrays
  - Advancing `windio` to version 2.0
- Models:
  - New turbine type `CalculatorType`, for direct infusion of farm data at a turbine through a user function
  - New rotor model `DirectMDataInfusion`, piping stored model data directly into turbines
  - Generalizing the `Calculator` turbine model, now additionally passing `algo, mdata, fdata` to the user function
- Examples:
  - New example `point_cloud`, demonstrating the useage of the `PointCloud` ambient states class
  - New example `weibull_grid`, demonstrating the useage of the `WeibullField` ambient states class
  - New example `weibull_cloud`, demonstrating the useage of the `WeibullPointCloud` ambient states class
  - New example `direct_mdata_infusion`, demonstrating how to infuse states data dirctly into rotor data without interpolation
- Bug fixes:
  - Bug fixed with turbine model `SetFarmVars` that mixed up the turbine ordering
  - Bug fixed with turbine models that are marked as `pre_rotor` that mixed up the turbine ordering

**Full Changelog**: [https://github.com/FraunhoferIWES/foxes/commits/v1.5](https://github.com/FraunhoferIWES/foxes/commits/v1.5)

## v1.5.1

- Examples:
  - Now all examples are making use of an engine via the `with` context, as it should be done in scripts
- Bug fixes:
 - Bug fixed that caused a reset in the `dyn_wakes` wake frame when calling `calc_points`
 - Bug fixed with `write_nc` utility, concerning string type coordinate arrays

**Full Changelog**: [https://github.com/FraunhoferIWES/foxes/commits/v1.5.1](https://github.com/FraunhoferIWES/foxes/commits/v1.5.1)

## v1.5.2

- Models:
  - New farm controller `OpFlagController`, switching off turbines based on state-time dependent boolean input data
  - Wake blockage model `SelfSimilar`: Promoting `alpha` and `beta` to constructor arguments
- Tests:
  - New test `test_set_farm_vars`, checking if correct turbine ordering is maintained
- Notebooks:
  - New notebook `operating_flag.ipynb`, demonstrating the usage of the `OpFlagController` class
- Bug fixes:
  - Bug fixed with negative beta in sqrt for some wake models, Madsen axial induction and high ct values
  - Bug fixed with wake model `bastankhah16` to set default value beta to 0.077 (was 0.07)

**Full Changelog**: [https://github.com/FraunhoferIWES/foxes/commits/v1.5.2](https://github.com/FraunhoferIWES/foxes/commits/v1.5.2)

## v1.6

- Python version:
  - Adding support for Python 3.14
- Core:
  - Wind farms now support turbine locations in terms of longitude, lattitude. They will automatically be translated into
  a uniquely fixed UTM zone, stored in the global config object.
  - Turbines now carry an optional property `wind_farm_name`, intended for cases which model multiple wind farms. The `WindFarm` class offers a property `wind_farm_names` and a function `get_wind_farm_mapping()`, returning a mapping from wind farm names to turbine indices. This is for example used by layout plots with the choice `annotate=3`, which results in printing wind farm names instead of turbine indices.
- Algorithms:
  - New optional dict-type parameter `write_nc` for `algo.calc_farm()` and `algo.calc_points()`, allowing the parallel writing of results into multiple files without ever constructing the complete dataset in memory. If writing results to file is the only intention of the run, this is the recommended approach.
  - Introducing new parameter `population_params` to `Downwind` algorithm, for setting up a vectorized parameter study. This is based on a set of values for selected variables, which are then evaluated for each state. This is realized via the algorithm specific states class `PopulationStates` and turbine model class `PopulationModel`.
  - Reworking the `Iterative` algorithm, such that it now computes only non-converged states in each iteration. In general, this leads to a substantial speed-up.
  - Introducing new boolean parameter `clear_mem` to `algo.calc_farm()`, clearing memory after initiating the calculation. This is intended to be used when there is only one single call of that function, or any of the used models.
- Engines:
  - Now all engines have `submit` and `await_result` functions (which are trivial for serial engines), also usable by the user from the `engine` object, e.g. for parallel plotting during post-processing. 
  - Adding support for parallel results writing via `write_nc` parameter, optionally without ever creating the full dataset in memory. Independent of writing, the futures of individual chunks are now computed directly when needed, no longer in advance. 
  - Engines have a new boolean parameter `progress_bar`, and if set to False, they print the progress to terminal instead. This is recommended for runs on clusters.
  - Dropping `XArrayEngine`, since no longer compatible
- Inputs:
  - New farm layout input `from_eww`, reading wind farm input in EuroWindWakes format
  - New farm layout input `from_wrf`, reading a folder that contains wind farm input data in a format readable by WRF
  - New states `NEWAStates`, directly reading NEWA-WRF data files into foxes
- Models:
  - Addition of `IEA37Gaussian` wake model to the model book, requiring slight generalization of the `Bastankhah2014` wake model class.
  - Adding default parameter wake models to the model book, following the papers: `Bastankhah2014`, `TurbOPark`.
- Utils:
  - Improved `write_nc` utility, now better data compression through the usage of appropriate encoding
- Examples:
  - New example `parameter_study`, demonstrating the setup of a parameter study
- Bug fixes:
  - Fix for max x selection in `FieldDataNC`, thanks @qzheng75
  - Fix for `Sequential` algorithm with non-updated chunk store data
  - Fix for `TurbOPark` wake model in the context of `windio` input files
  - Fix for bug with writing time stamps to NetCDF in `write_nc`

**Full Changelog**: [https://github.com/FraunhoferIWES/foxes/commits/v1.6](https://github.com/FraunhoferIWES/foxes/commits/v1.6)
