from .output import Output


class ResultsWriter(Output):
    """
    Writes results farm data to file.

    Parameters
    ----------
    farm_results : xarray.Dataset, optional
        The farm results, if data is None
    data : pandas.DataFrame, optional
        The data, if farm_results is None

    Attributes
    ----------
    data : pandas.DataFrame
        The farm results

    """

    def __init__(self, farm_results=None, data=None):

        if farm_results is not None and data is None:
            self.data = farm_results.to_dataframe()
        elif farm_results is None and data is not None:
            self.data = data
        else:
            raise KeyError(
                f"ResultsWriter: Either give 'farm_results' or 'data' arguments"
            )

    def write_csv(
        self,
        file_path,
        variables=None,
        verbosity=1,
        **kwargs,
    ):
        """
        Writes a csv file

        Parameters
        ----------
        file_path : str
            Path the the csv file
        variables : dict or list of str, optional
            The variables to be written. If a dict, then
            the keys are the foxes variables and the values
            the column names. If None, then all data will be
            written.
        verbosity : int
            The verbosity level, 0 = silent
        kwargs : dict, optional
            Additional parameters for Output.write

        """
        if verbosity:
            print(f"ResultsWriter: Writing file '{file_path}'")

        if variables is None:
            data = self.data
        elif isinstance(variables, dict):
            inds = {
                s: variables.pop(s) for s in self.data.index.names if s in variables
            }
            data = self.data
            if len(variables):
                data = data[list(variables.keys())].rename(variables, axis=1)
            if len(inds):
                for s, ns in inds.items():
                    l = self.data.index.names.index(s)
                    data = data.rename_axis(index={s: ns})
        else:
            data = self.data[list(variables)]

        super().write(file_path, data, **kwargs)
