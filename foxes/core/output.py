"""Output model lifecycle, chunk result composition and file path helpers."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, cast
import pandas as pd
import numpy as np

from foxes.config import config, get_output_path
from foxes.utils import PandasFileHelper, new_instance, all_subclasses, Dict

from .data_calc_model import DataCalcModel

if TYPE_CHECKING:
    from foxes.core.algorithm import Algorithm
    from foxes.core.data import FData, MData, TData
    from foxes.core.model import Model


class Output(DataCalcModel):
    """Base contract for chunk-local output and unified calculation results.

    Engine workers call ``calculate`` on prepared NumPy data. The engine then
    calls ``update_results`` with ordered chunk payloads and ``finalize_results``
    after all chunks have been processed. Outputs share result mappings and must
    consume only their own pending keys.
    """

    def __init__(
        self,
        out_dir: str | Path | None = None,
        out_fname_fun: Callable[[Path], Path] | None = None,
        nofig: bool = False,
        mdata_variables: Sequence[str] | None = (),
        fdata_variables: list[str] | None = None,
        tdata_variables: list[str] | None = None,
        show: bool = False,
        **kwargs: Any,
    ) -> None:
        """Configure input requirements, output paths and optional figure display.

        Parameters
        ----------
        out_dir
            Output directory, resolved relative to the configured output path.
            ``None`` uses ``config.output_dir``. Missing directories are created.
        out_fname_fun
            Transform each filename before joining it to the output directory.
        nofig
            Suppress figure creation for plotting outputs; takes precedence over
            ``show``.
        mdata_variables
            Required model variables during ``calculate``; ``None`` requests all,
            and an empty sequence requests none. The sequence is copied.
        fdata_variables
            Required farm variables; ``None`` requests all. The list is copied.
        tdata_variables
            Required target variables; ``None`` requests all. The list is copied.
        show
            Display figures for plotting outputs during parent-side finalization.
            Display blocks until the windows close. Numerical results are
            unchanged, and figure creation does not itself enable display.
        kwargs
            Additional parameters for ``DataCalcModel``.
        """
        # Resolve paths and copy input requirements for this output instance.
        super().__init__(**kwargs)
        self.out_dir = (
            get_output_path(out_dir) if out_dir is not None else config.output_dir
        )
        self.out_fname_fun = out_fname_fun
        self.nofig = nofig
        self.show = show
        self._mdata_variables = (
            None if mdata_variables is None else list(mdata_variables)
        )
        self._fdata_variables = (
            None if fdata_variables is None else list(fdata_variables)
        )
        self._tdata_variables = (
            None if tdata_variables is None else list(tdata_variables)
        )

        # Create the requested output directory without changing global configuration.
        if not self.out_dir.is_dir():
            print(f"{type(self).__name__}: Creating output dir {self.out_dir}")
            self.out_dir.mkdir(parents=True)

    def _show_figures(self) -> None:
        """Display completed plots from finalization when explicitly enabled.

        Call after writing configured files and while any displayed animation
        remains referenced locally. Blocking display keeps animation playback
        alive without retaining figures or scientific data on the output model.
        """
        if self.show and not self.nofig:
            import matplotlib.pyplot as plt

            plt.show(block=True)

    @property
    def mdata_variables(self) -> list[str] | None:
        """The required model variables.

        Returns
        -------
        variables
            Model variable names, or ``None`` to request all variables.
        """
        # Expose the requirements consumed by output preparation.
        return self._mdata_variables

    @property
    def fdata_variables(self) -> list[str] | None:
        """The required farm variables.

        Returns
        -------
        variables
            Farm variable names, or ``None`` to request all variables.
        """
        # Expose the requirements consumed by output preparation.
        return self._fdata_variables

    @property
    def tdata_variables(self) -> list[str] | None:
        """The required target variables.

        Returns
        -------
        variables
            Target variable names, or ``None`` to request all variables.
        """
        # Expose the requirements consumed by output preparation.
        return self._tdata_variables

    def output_coords(self) -> tuple[str, ...]:
        """
        Gets the coordinates of all output arrays

        Returns
        -------
        dims
            The coordinates of all output arrays

        """
        raise NotImplementedError

    def get_calc_points(self, algo: Algorithm) -> np.ndarray | None:
        """Return the calculation points required by this initialized output.

        Parameters
        ----------
        algo
            Initialized algorithm owning the output's loaded calculation data.

        Returns
        -------
        points
            Static coordinates shaped ``(n_points, 3)`` or state-dependent
            coordinates shaped ``(n_states, n_points, 3)``. ``None`` means
            no point calculation is requested.

        Notes
        -----
        ``calc_outputs`` obtains these points before dispatching engine chunks.
        Farm outputs evaluate their points in bounded slices per state chunk.
        ``calc_points`` ignores this hook and uses its supplied points or
        precomputed point results.
        """
        return None

    def calculate(  # type: ignore[override]
        self,
        algo: Algorithm,
        mdata: MData,
        fdata: FData,
        tdata: TData | None,
        model_results: Any,
        chunk_results: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        """Create or preserve the shared result mapping for one chunk.

        Overrides calculate output from prepared NumPy arrays rather than the
        variadic scientific-model inputs of ``DataCalcModel.calculate``.

        Parameters
        ----------
        algo
            The calculation algorithm
        mdata
            The model data
        fdata
            The farm data
        tdata
            The target data, or ``None`` for farm calculations.
        model_results
            The results from the model calculation for this chunk
        chunk_results
            Existing chunk results, keys are result object names

        Returns
        -------
        chunk_results
            The updated chunk results, keys are result object names

        """
        # Preserve existing output results when chaining multiple output models.
        if chunk_results is None:
            k = (mdata.chunki_states, mdata.chunki_points)
            chunk_results = Dict(_name=f"chunk_results_{k}")
        return chunk_results

    def update_results(
        self,
        algo: Algorithm,
        n_chunks_states: int,
        n_chunks_targets: int,
        new_chunk_results: Dict[str, Dict[tuple[int, int], Any]],
        output_results: Dict[str, Any],
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
            New chunk results, keys are result object names.
            Values are dicts with keys (states_chunk_index, target_chunk_index)
        output_results
            Output results to be updated, keys are result object names

        """
        pass

    def finalize_results(
        self,
        algo: Algorithm,
        n_chunks_states: int,
        n_chunks_targets: int,
        output_results: Dict[str, Any],
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
            Output results to be updated, keys are result object names

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
    """An ordered output chain with merged chunk input requirements."""

    def __init__(
        self,
        *args: Any,
        outputs: list[Output] | None = None,
        **kwargs: Any,
    ) -> None:
        """Configure an independent list of outputs.

        Parameters
        ----------
        args
            Additional parameters for the base class constructor
        outputs
            Outputs to execute in order, copied into an owned list.
            ``None`` creates an empty chain.
        kwargs
            Additional parameters for the base class
        """
        # Initialize common output options before collecting child requirements.
        super().__init__(*args, **kwargs)

        # Own the list so separate output chains cannot share mutable state.
        self._outputs = [] if outputs is None else list(outputs)
        self._update_variables()

    def _update_variables(self) -> None:
        """Combine child variable requirements, with ``None`` requesting all."""
        # Merge requirements in declaration order rather than set iteration order.
        for vrs in ["_mdata_variables", "_fdata_variables", "_tdata_variables"]:
            v: list[str] | None = []
            for o in self._outputs:
                vrs0 = getattr(o, vrs)
                if vrs0 is None:
                    v = None
                    break
                else:
                    v += vrs0
            if v is not None:
                v = list(dict.fromkeys(v))
            setattr(self, vrs, v)

    @property
    def outputs(self) -> list[Output]:
        """
        The ordered child outputs.

        Returns
        -------
        outputs
            The owned list of output instances. Use ``add_output`` when adding
            a child so the combined variable requirements are recalculated.

        """
        # Expose the child order used by the engine output hooks.
        return self._outputs

    def get_calc_points(self, algo: Algorithm) -> np.ndarray | None:
        """Return the common calculation points required by child outputs.

        Parameters
        ----------
        algo
            Initialized algorithm owning the children's loaded calculation data.

        Returns
        -------
        points
            Shared points for all point-dependent outputs, or ``None`` when
            none of the child outputs requests calculation points.

        Raises
        ------
        ValueError
            If child outputs request different calculation points.
        """
        # A single calc_points call can only evaluate one shared point set.
        points = None
        for output in self.outputs:
            output_points = output.get_calc_points(algo)
            if output_points is None:
                continue
            if points is None:
                points = output_points
            elif not np.array_equal(points, output_points):
                raise ValueError(
                    f"{self.name}: Child outputs request different calculation points"
                )
        return points

    def add_output(self, output: Output) -> None:
        """
        Add an output and recompute the combined input requirements.

        Parameters
        ----------
        output
            The output to be added

        """
        # Keep preparation requirements synchronized with the output chain.
        self._outputs.append(output)
        self._update_variables()

    def sub_models(self) -> list[Model]:
        """
        Return the list of all sub-models.

        Returns
        -------
        smdls
            All sub-models.

        """
        # Return child models for initialization and running-state propagation.
        return [output for output in self.outputs]

    def calculate(  # type: ignore[override]
        self,
        algo: Algorithm,
        mdata: MData,
        fdata: FData,
        tdata: TData | None,
        model_results: Any,
        chunk_results: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
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
            The target point data, or ``None`` for farm calculations.
        model_results
            The results from the model calculation for this chunk
        chunk_results
            Existing chunk results, keys are result object names

        Returns
        -------
        chunk_results
            The updated chunk results, keys are result object names

        """
        # Create one result mapping shared by all outputs in this chunk.
        chunk_results = super().calculate(
            algo=algo,
            mdata=mdata,
            fdata=fdata,
            tdata=tdata,
            model_results=model_results,
            chunk_results=chunk_results,
        )
        # Execute children in declaration order while retaining earlier payloads.
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
        new_chunk_results: Dict[str, Dict[tuple[int, int], Any]],
        output_results: Dict[str, Any],
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
            New chunk results, keys are result object names.
            Values are dicts with keys (states_chunk_index, target_chunk_index)
        output_results
            Output results to be updated, keys are result object names

        """
        # Let each child consume only its own pending output key.
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
        output_results: Dict[str, Any],
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
            Output results to be updated, keys are result object names

        """
        # Complete child unification and outstanding writes in declaration order.
        for o in self.outputs:
            o.finalize_results(
                algo,
                n_chunks_states,
                n_chunks_targets,
                output_results,
            )
