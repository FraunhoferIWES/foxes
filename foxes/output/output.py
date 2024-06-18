from pathlib import Path

from foxes.utils import PandasFileHelper, all_subclasses


class Output:
    """
    Base class for foxes output.

    The job of this class is to provide handy
    helper functions.

    Attributes
    ----------
    out_dir: pathlib.Path
        The output file directory
    out_fname_fun: Function, optional
        Modifies file names by f(fname)

    :group: output

    """

    def __init__(self, out_dir=None, out_fname_fun=None):
        """
        Constructor.

        Parameters
        ----------
        out_dir: str, optional
            The output file directory
        out_fname_fun: Function, optional
            Modifies file names by f(fname)

        """
        self.out_dir = Path(out_dir) if out_dir is not None else None
        self.out_fname_fun = out_fname_fun

    def get_fpath(self, fname):
        """
        Gets the total file path

        Parameters
        ----------
        fname: str
            The file name

        Returns
        -------
        fpath: pathlib.Path
            The total file path

        """
        fnm = Path(fname)
        if self.out_fname_fun is not None:
            fnm = self.out_fname_fun(fnm)
        return self.out_dir / fnm if self.out_dir is not None else fnm

    def write(self, file_name, data, format_col2var={}, format_dict={}, **kwargs):
        """
        Writes data to file via pandas.

        The kwargs are forwarded to the underlying pandas writing function.

        Parameters
        ----------
        file_name: str
            The output file name
        data: pandas.DataFrame
            The data
        format_col2var: dict
            Mapping from column names to foxes variables,
            for formatting
        format_dict: dict
            Dictionary with format entries for columns, e.g.
            {FV.P: '{:.4f}'}. Note that the keys are foxes variables

        """
        fdict = {}
        for c in data.columns:
            v = format_col2var.get(c, c)
            if v in format_dict:
                fdict[c] = format_dict[v]
            elif v in PandasFileHelper.DEFAULT_FORMAT_DICT:
                fdict[c] = PandasFileHelper.DEFAULT_FORMAT_DICT[v]

        fpath = self.get_fpath(file_name)
        PandasFileHelper.write_file(data, fpath, fdict, **kwargs)

    @classmethod
    def print_models(cls):
        """
        Prints all model names.
        """
        names = sorted([scls.__name__ for scls in all_subclasses(cls)])
        for n in names:
            print(n)

    @classmethod
    def new(cls, output_type, *args, **kwargs):
        """
        Run-time output model factory.

        Parameters
        ----------
        output_type: string
            The selected derived class name
        args: tuple, optional
            Additional parameters for the constructor
        kwargs: dict, optional
            Additional parameters for the constructor

        """

        if output_type is None:
            return None

        allc = all_subclasses(cls)
        found = output_type in [scls.__name__ for scls in allc]

        if found:
            for scls in allc:
                if scls.__name__ == output_type:
                    return scls(*args, **kwargs)

        else:
            estr = "Output type '{}' is not defined, available types are \n {}".format(
                output_type, sorted([i.__name__ for i in allc])
            )
            raise KeyError(estr)
