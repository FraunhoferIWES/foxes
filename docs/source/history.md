# History

## v0.1.0-alpha

This is the initial release of **foxes** - ready for testing.

So far not many models have been transferred from the Fraunhofer IWES in-house predecessor *flappy*, they will be added in the following versions. Also optimization is not yet included. We are just getting started here!

Enjoy - we are awaiting comments and issues, thanks for testing.

**Full Changelog**: https://github.com/FraunhoferIWES/foxes/commits/v0.1.0

## v0.1.1-alpha

- New code style, created by *black*
- Small fixes, discovered by *flake8*
- Tests now via *pytest* instead of *unittest*
- Introducing github workflow for automatic testing

**Full Changelog**: https://github.com/FraunhoferIWES/foxes/commits/v0.1.1

## v0.1.2-alpha

- Adding support for Python 3.9, 3.10

**Full Changelog**: https://github.com/FraunhoferIWES/foxes/commits/v0.1.2

## v0.1.3-alpha

- Adding docu: https://fraunhoferiwes.github.io/foxes.docs/index.html

**Full Changelog**: https://github.com/FraunhoferIWES/foxes/commits/v0.1.3

## v0.1.4-alpha

- Fixes
    - Static data: Adding missing data `wind_rotation.nc` to manifest
- Models
    - New wake model added: `TurbOParkWake` from Orsted
    - New turbine type added: `PCtSingleFiles`, reads power and thrust curves from two separate files
    - New turbulence intensity wake model added: `IECTI2019`/`Frandsen` and `IECTI2005`

**Full Changelog**: https://github.com/FraunhoferIWES/foxes/commits/v0.1.4

## v0.1.5-alpha

- Core
    - Introducing the concept of runners
- Opt
    - New sub package: `foxes.opt`, install by `pip install foxes[opt]`
- Models
    - New turbine model: `Thrust2Ct`, calculates ct from thrust values
    - New turbine type: `NullType`, a turbine type with only rotor diameter and hub height data
    - New runners: `SimpleRunner`, `DaskRunner`. The latter features parallel runs via dask
- Examples
    - Introducing two sub-folders of examples: `foxes` and `foxes.opt`
    - New example: `wind_rose`, calculation of wind rose states data
    - New example: `layout_single_state`, wind farm layout optimization for a single wind state
    - New example: `layout_wind_rose`, wind farm layout optimization for wind rose states

**Full Changelog**: https://github.com/FraunhoferIWES/foxes/commits/v0.1.5

