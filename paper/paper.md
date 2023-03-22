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
    #orcid: ???
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

The pre and post construction analysis of wind farms requires the fast evaluation of 
the annual energy production for long-term time series which often contain many ten 
thousand entries. In addition, wake effects from neighbouring wind farms and wind farm 
clusters need to be considered, such that realistic off-shore scenarios can include as
many as thousands of wind turbines. Hence scalability is an issue of central importance
for modern numerical wind farm and wake modelling approaches. Furthermore, optimization 
of wind farms and wind farm control parameters are often based on genetic or other 
heuristic algorithms. For such approaches the vectorized evaluation of optimization 
variables is the key for fast and efficient calculations.

# Statement of need

`foxes` is a Python package that calculates wind farm results including wind turbine wake 
effects. The meteorological input data can be a timeseries or statistical data like wind 
roses or other distributions. Uniform inflow fields are supported as well as
horizontal profiles and three dimensional flow fields from other sources, for example
mesoscale simulation results.

`foxes` is build upon the idea of fast vectorized evaluation of the input states, making use 
of the `Dask` package [@dask] via the `xarray` package [@xarray]. This means that in general it 
does not solve differential equations that interconnect the states and rather relies either on 
analytical models or on lookup-table based models. An exception are the included features of 
calculations along streamlines of the background flow, and also integrations of variables along 
the latter for some models.

`foxes` was designed to be very modular when it comes to modelling aspects of wind farm 
calculations. Users can combine various model types and easily add new models, such that a 
realistic representation of the wind farm's behaviour can be achieved. Any wind farm calculation 
variable in `foxes` can be subjected to optimization via the `IWOPY` package [@iwopy], with 
support for vectorized evaluation of optimization variables.

# Mathematics

Single dollars ($) are required for inline mathematics e.g. $f(x) = e^{\pi/x}$

Double dollars make self-standing equations:

$$\Theta(x) = \left\{\begin{array}{l}
0\textrm{ if } x < 0\cr
1\textrm{ else}
\end{array}\right.$$

You can also use plain \LaTeX for equations
\begin{equation}\label{eq:fourier}
\hat f(\omega) = \int_{-\infty}^{\infty} f(x) e^{i\omega x} dx
\end{equation}
and refer to \autoref{eq:fourier} from text.

# Citations

Citations to entries in paper.bib should be in
[rMarkdown](http://rmarkdown.rstudio.com/authoring_bibliographies_and_citations.html)
format.

If you want to cite a software repository URL (e.g. something on GitHub without a preferred
citation) then you can do it with the example BibTeX entry below for @fidgit.

For a quick reference, the following citation commands can be used:
- `@author:2001`  ->  "Author et al. (2001)"
- `[@author:2001]` -> "(Author et al., 2001)"
- `[@author1:2001; @author2:2001]` -> "(Author1 et al., 2001; Author2 et al., 2002)"

# Figures

Figures can be included like this:
![Caption for example figure.\label{fig:example}](figure.png)
and referenced from text using \autoref{fig:example}.

Figure sizes can be customized by adding an optional second parameter:
![Caption for example figure.](figure.png){ width=20% }

# Acknowledgements

The development of `foxes` and its predecessors flapFOAM and flappy (internal - non public) has been supported through multiple publicly funded research projects. We acknowledge in particular the funding by the Federal Ministry of Economic Affairs and Climate Action (BMWK) through the projects Smart Wind Farms (grant no. 0325851B), GW-Wakes (0325397B) and X-Wakes (03EE3008A) as well as the funding by the Federal Ministry of Education and Research (BMBF) in the framework of the project H2Digital (03SF0635).

# References