---
title: 'FOXES: Farm Optimization and eXtended yield Evaluation Software'
tags:
  - Python
  - wind energy
  - wind turbine wakes
  - wake modelling
  - wind farm optimization
authors:
  - name: Jonas Schmidt
    orcid: 0000-0002-8191-8141
    #equal-contrib: true
    affiliation: 1 # (Multiple affiliations must be quoted)
    corresponding: true # (This is how to denote the corresponding author)
  - name: Lukas Vollmer
    #equal-contrib: true # (This is how you can denote equal contributions between multiple authors)
    #orcid: ???
    affiliation: 1
  - name: Martin Dörenkämper
    orcid: 0000-0002-0210-5733
    affiliation: 1
  - name: Bernhard Stoevesandt
    orcid: 0000-0001-6626-1084
    affiliation: 1
affiliations:
 - name: Fraunhofer IWES, Küpkersweg 70, 26129 Oldenburg, Germany
   index: 1
date: 24 March 2023
bibliography: paper.bib

# Optional fields if submitting to a AAS journal too, see this blog post:
# https://blog.joss.theoj.org/2018/12/a-new-collaboration-with-aas-publishing
#aas-doi: 10.3847/xxxxx <- update this with the DOI from AAS once you know it.
#aas-journal: Astrophysical Journal <- The name of the AAS journal.
---

# Summary

The `Farm Evaluation and eXtended yield Evaluation Software (foxes)` is a Python package that 
calculates wind farm results such as power production or wind turbine wake effects. The meteorological 
input data can be a timeseries or statistical data like wind speed histograms, wind roses or any other 
distribution. Uniform inflow fields are supported as well as horizontal profiles and three dimensional
flow fields from other sources, for example wind fields from mesoscale weather models or wind
atlases [@newa]. Typical applications are wind farm optimization (e.g., layout optimization or wake steering),
wind farm pre- and post-construction analyses; wake model studies, comparison and validation; 
and wind farm simulations invoking complex model interactions.

# Statement of need

The amount of electrical energy that wind farms are able to extract from the kinetic energy
of the wind naturally depends on the meteorological conditions at the site. In particular,
wind speed and wind direction are decisive quantities, and in general those are time and space
dependent fields. Furthermore, for each wind turbine, its location within the wind farm plays an 
important role, since wake effects stemming from up-stream turbines reduce the wind speed and 
increase turbulence at the rotor. Additionally, the rotor height, its size as well as specifics 
related to the wind turbine model are crucial parameters. When it comes to wind farm production 
analysis, the fast evaluation of the annual energy production for long-term time series of 
meteorological conditions is required, and such data often contains many ten thousand entries. 

In addition, wake effects from neighbouring wind farms and wind farm clusters need to be 
considered, such that realistic offshore scenarios can include as many as thousands of wind
turbines. Hence scalability is an issue of central importance for modern numerical wind farm 
and wake modelling approaches. 

`foxes` is built upon the idea of fast vectorized evaluation of the input states, making use 
of the `dask` package [@dask] via the `xarray` package [@xarray]. This means that in general, it 
does not solve differential equations that interconnect the states, but rather relies either on 
analytical models or on lookup-table based models. Exceptions are the included features of 
calculations along streamlines of the background flow and integrations of variables along 
the latter for some models.

`foxes` was designed to be very modular when it comes to modelling aspects of wind farm 
calculations. Users can combine various model types and easily add new models, such that a 
realistic representation of the wind farm's behaviour can be achieved. Such models include, among others, rotor discretization models, wake models and turbine models such as wind sector management, turbine derating or power boost.

Any wind farm calculation variable in `foxes` can be subjected to optimization via the `iwopy` package [@iwopy], with support for vectorized evaluation of optimization variables. This is crucial for the fast evaluation of genetic or other heuristic algorithms, for example for the purpose of solving the turbine positioning optimization problem or finding optimal wind farm control parameters.

Other related open source Python packages that follow a similar agenda, but with differences in 
methods, models and architecture, are `pywake` [@pywake] and `floris` [@floris]. The `foxes` package 
has no code overlap with these packages and has been developed as an independent software for many 
years.

# Acknowledgements

The development of `foxes` and its predecessors flapFOAM and flappy (internal - non public) has been 
supported through multiple publicly funded research projects. We acknowledge in particular the funding 
by the Federal Ministry of Economic Affairs and Climate Action (BMWK) through the projects Smart Wind 
Farms (grant no. 0325851B), GW-Wakes (0325397B) and X-Wakes (03EE3008A) as well as the funding by the 
Federal Ministry of Education and Research (BMBF) in the framework of the project H2Digital (03SF0635).

# References
