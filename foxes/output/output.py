from pathlib import Path

from foxes.config import config, get_path
from foxes.utils import PandasFileHelper, new_instance, all_subclasses


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
        self.out_dir = get_path(out_dir) if out_dir is not None else config.out_dir
        self.out_fname_fun = out_fname_fun

        if not self.out_dir.is_dir():
            print(f"{type(self).__name__}: Creating output dir {self.out_dir}")
            self.out_dir.mkdir(parents=True)

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
        return self.out_dir / fnm if self.out_dir is not None else get_path(fnm)

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
        return new_instance(cls, output_type, *args, **kwargs)
