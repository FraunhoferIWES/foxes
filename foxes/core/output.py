from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, cast
import pandas as pd

from foxes.config import config, get_output_path
from foxes.utils import PandasFileHelper, new_instance, all_subclasses, Dict

from .data_calc_model import DataCalcModel

if TYPE_CHECKING:
    from foxes.core.algorithm import Algorithm
    from foxes.core.data import FData, MData, TData
    from foxes.core.model import Model


class Output(DataCalcModel):
    """
    Base class for foxes output.

    The job of this class is to provide handy
    helper functions.
    """

    def __init__(
        self,
        out_dir: str | Path | None = None,
        out_fname_fun: Callable[[Path], Path] | None = None,
        nofig: bool = False,
        **kwargs: Any,
    ) -> None:
        """
        Parameters
        ----------
        out_dir
            The output file directory
        out_fname_fun
            Modifies file names by f(fname)
        nofig
            Do not show figures
        kwargs
            Additional parameters for the base class
        """
        super().__init__(**kwargs)
        self.out_dir = (
            get_output_path(out_dir) if out_dir is not None else config.output_dir
        )
        self.out_fname_fun = out_fname_fun
        self.nofig = nofig

        if not self.out_dir.is_dir():
            print(f"{type(self).__name__}: Creating output dir {self.out_dir}")
            self.out_dir.mkdir(parents=True)

    def output_coords(self) -> tuple[str, ...]:
        """
        Gets the coordinates of all output arrays

        Returns
        -------
        dims
            The coordinates of all output arrays

        """
        raise NotImplementedError

    def calculate(
        self,
        algo: Algorithm,
        mdata: MData,
        fdata: FData,
        tdata: TData,
        model_results: Any,
        chunk_results: Dict | None = None,
    ) -> Dict:
        """
        The chunk based output calculation.

        This function is executed on a single chunk of data,
        all computations should be based on numpy arrays.

        Parameters
        ----------
        algo
            The calculation algorithm
        mdata
            The model data
        fdata
            The farm data
        tdata
            The target point data
        model_results
            The results from the model calculation for this chunk
        chunk_results
            Existing chunk results

        Returns
        -------
        chunk_results
            The updated chunk results

        """
        if chunk_results is None:
            k = (mdata.chunki_states, mdata.chunki_points)
            chunk_results = Dict(_name=f"chunk_results_{k}")
        return chunk_results

    def update_results(
        self,
        algo: Algorithm,
        n_chunks_states: int,
        n_chunks_targets: int,
        new_chunk_results: Dict[tuple[int, int], Dict],
        output_results: Dict,
    ) -> None:
        """
        Update output results with latest new chunk results.

        Parameters
        ----------
        algo
            The algorithm
        n_chunks_states
            The number of state chunks
        n_chunks_targets
            The number of target chunks
        new_chunk_results
            New chunk results, keys: (states_chunk_index, target_chunk_index),
            values: dict with chunk results
        output_results
            Existing output results to be updated

        """
        pass

    def finalize_results(
        self,
        algo: Algorithm,
        n_chunks_states: int,
        n_chunks_targets: int,
        output_results: Dict,
    ) -> None:
        """
        Final update of the output results, after all chunks have been processed.

        Parameters
        ----------
        algo
            The algorithm
        n_chunks_states
            The number of state chunks
        n_chunks_targets
            The number of target chunks
        output_results
            Existing output results to be updated

        """
        pass

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


class OutputList(Output):
    """A list of outputs"""

    def __init__(
        self,
        *args: Any,
        outputs: list[Output] = [],
        **kwargs: Any,
    ) -> None:
        """
        Parameters
        ----------
        args
            Additional parameters for the base class constructor
        outputs
            The outputs
        kwargs
            Additional parameters for the base class
        """
        super().__init__(*args, **kwargs)
        self.outputs = outputs

    def add_output(self, output: Output):
        """
        Adds an output to the list

        Parameters
        ----------
        output
            The output to be added

        """
        self.outputs.append(output)

    def sub_models(self) -> list[Model]:
        """
        Return the list of all sub-models.

        Returns
        -------
        smdls
            All sub-models.

        """
        return self.outputs

    def calculate(
        self,
        algo: Algorithm,
        mdata: MData,
        fdata: FData,
        tdata: TData,
        model_results: Any,
        chunk_results: Dict | None = None,
    ) -> Dict:
        """
        The chunk based output calculation.

        This function is executed on a single chunk of data,
        all computations should be based on numpy arrays.

        Parameters
        ----------
        algo
            The calculation algorithm
        mdata
            The model data
        fdata
            The farm data
        tdata
            The target point data
        model_results
            The results from the model calculation for this chunk
        chunk_results
            Existing chunk results

        Returns
        -------
        chunk_results
            The updated chunk results

        """
        chunk_results = super().calculate(
            algo=algo,
            mdata=mdata,
            fdata=fdata,
            tdata=tdata,
            model_results=model_results,
            chunk_results=chunk_results,
        )
        for o in self.outputs:
            chunk_results = o.calculate(
                algo=algo,
                mdata=mdata,
                fdata=fdata,
                tdata=tdata,
                model_results=model_results,
                chunk_results=chunk_results,
            )
        return chunk_results

    def update_results(
        self,
        algo: Algorithm,
        n_chunks_states: int,
        n_chunks_targets: int,
        new_chunk_results: Dict[tuple[int, int], Dict],
        output_results: Dict,
    ) -> None:
        """
        Update output results with latest new chunk results.

        Parameters
        ----------
        algo
            The algorithm
        n_chunks_states
            The number of state chunks
        n_chunks_targets
            The number of target chunks
        new_chunk_results
            New chunk results, keys: (states_chunk_index, target_chunk_index),
            values: dict with chunk results
        output_results
            Existing output results to be updated

        """
        for o in self.outputs:
            o.update_results(
                algo,
                n_chunks_states,
                n_chunks_targets,
                new_chunk_results,
                output_results,
            )

    def finalize_results(
        self,
        algo: Algorithm,
        n_chunks_states: int,
        n_chunks_targets: int,
        output_results: Dict,
    ) -> None:
        """
        Final update of the output results, after all chunks have been processed.

        Parameters
        ----------
        algo
            The algorithm
        n_chunks_states
            The number of state chunks
        n_chunks_targets
            The number of target chunks
        output_results
            Existing output results to be updated

        """
        for o in self.outputs:
            o.finalize_results(
                algo,
                n_chunks_states,
                n_chunks_targets,
                output_results,
            )
