from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, cast
import pandas as pd

from foxes.config import config, get_output_path
from foxes.utils import PandasFileHelper, new_instance, all_subclasses


class Output:
    """
    Base class for foxes output.

    The job of this class is to provide handy
    helper functions.

    Attributes
    ----------
    out_dir
        The output file directory
    out_fname_fun
        Modifies file names by f(fname)
    nofig
        Do not show figures

    :group: output

    """

    def __init__(
        self,
        out_dir: str | Path | None = None,
        out_fname_fun: Callable[[Path], Path] | None = None,
        nofig: bool = False,
    ) -> None:
        """
        Constructor.

        Parameters
        ----------
        out_dir
            The output file directory
        out_fname_fun
            Modifies file names by f(fname)
        nofig
            Do not show figures

        """
        self.out_dir = (
            get_output_path(out_dir) if out_dir is not None else config.output_dir
        )
        self.out_fname_fun = out_fname_fun
        self.nofig = nofig

        if not self.out_dir.is_dir():
            print(f"{type(self).__name__}: Creating output dir {self.out_dir}")
            self.out_dir.mkdir(parents=True)

    def get_fpath(self, fname: str | Path) -> Path:
        """
        Gets the total file path

        Parameters
        ----------
        fname
            The file name

        Returns
        -------
        fpath
            The total file path

        """
        fnm = Path(fname)
        if self.out_fname_fun is not None:
            fnm = self.out_fname_fun(fnm)
        return self.out_dir / fnm if self.out_dir is not None else get_output_path(fnm)

    def write(
        self,
        file_name: str,
        data: pd.DataFrame,
        format_col2var: dict[str, str] | None = None,
        format_dict: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Writes data to file via pandas.

        The kwargs are forwarded to the underlying pandas writing function.

        Parameters
        ----------
        file_name
            The output file name
        data
            The data
        format_col2var
            Mapping from column names to foxes variables,
            for formatting
        format_dict
            Dictionary with format entries for columns, for example
            ``FV.P`` mapped to ``'{:.4f}'``. Note that the keys are foxes
            variables.

        """
        format_col2var = {} if format_col2var is None else format_col2var
        format_dict = {} if format_dict is None else format_dict
        fdict: dict[str, str] = {}
        for c in data.columns:
            v = format_col2var.get(c, c)
            if v in format_dict:
                fdict[c] = format_dict[v]
            elif v in PandasFileHelper.DEFAULT_FORMAT_DICT:
                fdict[c] = PandasFileHelper.DEFAULT_FORMAT_DICT[v]

        fpath = self.get_fpath(file_name)
        PandasFileHelper.write_file(data, fpath, fdict, **kwargs)

    @classmethod
    def print_models(cls) -> None:
        """
        Prints all model names.
        """
        names = sorted([scls.__name__ for scls in all_subclasses(cls)])
        for n in names:
            print(n)

    @classmethod
    def new(cls, output_type: str, *args: Any, **kwargs: Any) -> "Output":
        """
        Run-time output model factory.

        Parameters
        ----------
        output_type
            The selected derived class name
        args
            Additional parameters for the constructor
        kwargs
            Additional parameters for the constructor

        """
        return cast(Output, new_instance(cls, output_type, *args, **kwargs))
