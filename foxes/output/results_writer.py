import pandas as pd
from xarray import Dataset

import foxes.constants as FC
from foxes.utils import write_nc

from .output import Output


class ResultsWriter(Output):
    """
    Writes results farm data to file.

    Attributes
    ----------
    data: pandas.DataFrame
        The farm results

    :group: output

    """

    def __init__(self, farm_results=None, data=None, **kwargs):
        """
        Constructor.

        Parameters
        ----------
        farm_results: xarray.Dataset, optional
            The farm results, if data is None
        data: pandas.DataFrame, optional
            The data, if farm_results is None
        kwargs: dict, optional
            Additional parameters for the base class

        """
        super().__init__(**kwargs)

        if farm_results is not None and data is None:
            self.data = farm_results.to_dataframe().reset_index()
            self.data[FC.TNAME] = farm_results[FC.TNAME].to_numpy()[
                self.data[FC.TURBINE]
            ]
            self.data.set_index([FC.STATE, FC.TURBINE], inplace=True)
        elif farm_results is None and data is not None:
            self.data = data
        else:
            raise KeyError(
                f"ResultsWriter: Either give 'farm_results' or 'data' arguments"
            )

    def _get_data_vars(self, variables):
        """Helper function for variable gathering"""
        data = self.data
        if variables is None:
            pass
        elif isinstance(variables, dict):
            inds = {
                s: variables.pop(s) for s in self.data.index.names if s in variables
            }
            if len(variables):
                data = data[list(variables.keys())].rename(variables, axis=1)
            if len(inds):
                for s, ns in inds.items():
                    data = data.rename_axis(index={s: ns})
        else:
            data = self.data[list(variables)]

        return data, data.columns.tolist()

    def write_csv(
        self,
        file_path,
        variables=None,
        turbine_names=False,
        state_turbine_table=False,
        verbosity=1,
        **kwargs,
    ):
        """
        Writes a csv file

        Parameters
        ----------
        file_path: str
            Path the the csv file
        variables: dict or list of str, optional
            The variables to be written. If a dict, then
            the keys are the foxes variables and the values
            the column names. If None, then all data will be
            written.
        turbine_names: bool
            Use turbine names instead of turbine indices
        state_turbine_table: bool
            Flag for writing a single variable into turbine columns
            for state rows
        verbosity: int
            The verbosity level, 0 = silent
        kwargs: dict, optional
            Additional parameters for Output.write()

        """
        if verbosity:
            print(f"ResultsWriter: Writing file '{file_path}'")

        data, variables = self._get_data_vars(variables)
        data.reset_index(inplace=True)
        fc2v = kwargs.pop("format_col2var", {})
        tix = FC.TNAME if turbine_names else FC.TURBINE

        if state_turbine_table:
            if len(variables) != 1:
                raise ValueError(
                    f"state_turbine_table can only be written for a single variable, got {variables}"
                )
            v = variables[0]
            cnames = {
                t: f"{v}_T{t:04d}" if tix == FC.TURBINE else f"{v}_{t}"
                for t in data[tix]
            }
            for ti, (t, g) in enumerate(
                data.reset_index().set_index(FC.STATE).groupby(tix)
            ):
                if ti == 0:
                    odata = pd.DataFrame(
                        index=g.index.to_numpy(), columns=list(cnames.values())
                    )
                    odata.index.name = g.index.name
                cname = cnames[t]
                odata[cname] = g[v].to_numpy().copy()
                fc2v[cname] = v
            data = odata
        else:
            data.set_index([FC.STATE, tix], inplace=True)

        fpath = self.get_fpath(file_path)
        super().write(fpath, data, format_col2var=fc2v, **kwargs)

    def write_nc(
        self,
        file_path,
        variables=None,
        turbine_names=False,
        verbosity=1,
        **kwargs,
    ):
        """
        Writes a netCDF file

        Parameters
        ----------
        file_path: str
            Path the the csv file
        variables: dict or list of str, optional
            The variables to be written. If a dict, then
            the keys are the foxes variables and the values
            the column names. If None, then all data will be
            written.
        turbine_names: bool
            Use turbine names instead of turbine indices
        verbosity: int
            The verbosity level, 0 = silent
        kwargs: dict, optional
            Additional parameters for write_nc()

        """
        data, variables = self._get_data_vars(variables)
        states = data.index.get_level_values(0).unique()
        n_states = len(states)
        n_turbines = len(data.index.get_level_values(1).unique())

        crds = {FC.STATE: states}
        if turbine_names:
            idx = pd.IndexSlice
            crds[FC.TURBINE] = self.data.loc[idx[0, :], FC.TNAME].to_numpy()

        ds = Dataset(
            coords=crds,
            data_vars={
                v: (
                    (FC.STATE, FC.TURBINE),
                    data[v].to_numpy().reshape(n_states, n_turbines),
                )
                for v in variables
            },
        )

        fpath = self.get_fpath(file_path)
        write_nc(ds, fpath, verbosity=verbosity, **kwargs)
