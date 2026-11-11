from __future__ import annotations
# mypy: disable-error-code=override

import numpy as np
from typing import TYPE_CHECKING

from foxes.core import DataCalcModel, OutputList
from foxes.models.turbine_models.evaluation import TurbineEvaluation
import foxes.variables as FV
import foxes.constants as FC

if TYPE_CHECKING:
    from foxes.core.model import LoadedData, Model
    from foxes.core.algorithm import Algorithm
    from foxes.core.data import FData, MData, TData
    from foxes.core.output import Output


class PrepareForOutput(DataCalcModel):
    """Evaluate requested metrics, filter chunk fields and restore farm order.

    Evaluation uses the incoming calculation order before filtering or
    permutation. ``FV.ORDER`` maps downwind columns to original turbine
    indices; its inverse restores physical fields to original farm order.
    The order array itself remains downwind-sequence metadata.
    """

    def __init__(self, output: Output | None, reorder: bool = True) -> None:
        """Configure output filtering and final turbine reordering.

        Parameters
        ----------
        output
            The output object.
        reorder
            Restore physical state-turbine fields to original farm order after
            evaluation. Intermediate Iterative results disable this so the
            next wake pass receives downwind-ordered arrays. Final output
            preparation enables it again; ``FV.ORDER`` itself is never permuted.
        """
        super().__init__()
        self.output = output
        self.reorder = reorder
        self._eval: TurbineEvaluation | None = None

    def output_coords(self) -> tuple[str, ...]:
        """
        Gets the coordinates of all output arrays

        Returns
        -------
        dims
            The coordinates of all output arrays

        """
        raise NotImplementedError

    def output_farm_vars(self, algo: Algorithm) -> list[str]:
        """
        The variables which are being modified by the model.

        Parameters
        ----------
        algo
            The calculation algorithm

        Returns
        -------
        output_vars
            The output variable names

        """
        return []

    def initialize(
        self,
        algo: Algorithm,
        loaded_data: LoadedData | None = None,
        force: bool = False,
        verbosity: int = 0,
    ) -> LoadedData:
        """Initialize evaluation for explicitly requested missing farm metrics.

        Parameters
        ----------
        algo
            The calculation algorithm.
        loaded_data
            Data that has already been loaded, to be extended by this function.
            Keys are "coords", a dict with entries ``dim_name_str -> dim_array``;
            "data_vars", a dict with entries ``name_str -> (dim_tuple, data_ndarray)``;
            and "extra_data", a dict with non-array additional data.
        force
            Overwrite existing data.
        verbosity
            The verbosity level, where 0 is silent.

        Returns
        -------
        loaded_data
            The loaded data, containing the keys "coords", "data_vars", and
            "extra_data".

        Raises
        ------
        ValueError
            If requested probability yields lack a configured uncertainty.

        Notes
        -----
        An unrestricted farm-variable selection retains available fields;
        it does not request additional derived metrics.
        """
        # Only explicit missing fields request additional metric calculations.
        self._eval = None
        eval_vars: set[str] = set()
        pending = [] if self.output is None else [self.output]
        while pending:
            output = pending.pop()
            if isinstance(output, OutputList):
                pending.extend(output.outputs)
            elif output.fdata_variables is not None:
                eval_vars.update(
                    set(output.fdata_variables) & TurbineEvaluation.SUPPORTED_VARIABLES
                )
        if eval_vars:
            eval_vars.difference_update(algo.farm_vars)
            if eval_vars:
                self._eval = TurbineEvaluation(variables=sorted(eval_vars))

        return super().initialize(
            algo=algo,
            loaded_data=loaded_data,
            force=force,
            verbosity=verbosity,
        )

    def sub_models(self) -> list[Model]:
        """
        Return the list of all sub-models.

        Returns
        -------
        smdls
            All sub-models.

        """
        return [self._eval] if self._eval else []

    def calculate(
        self,
        algo: Algorithm,
        mdata: MData,
        fdata: FData,
        tdata: TData | None = None,
    ) -> dict[str, np.ndarray]:
        """
        Filter chunk fields and restore selected arrays to original farm order.

        Parameters
        ----------
        algo
            The running calculation algorithm.
        mdata
            Model chunk, filtered in place according to the output requirements.
        fdata
            Farm chunk in downwind order, filtered in place. ``FV.ORDER`` maps
            calculation columns to original turbine indices when reordering is
            required.
        tdata
            Target chunk, filtered in place when supplied. Physical point fields
            use ``(FC.STATE, FC.TARGET, FC.TPOINT)``.

        Returns
        -------
        results
            Empty mapping; retained arrays and dimension metadata are installed
            directly in the supplied containers. Without an output, all entries
            and their dimension metadata are removed.

        Raises
        ------
        KeyError
            If a retained nonuniform state-turbine field needs reordering but
            ``FV.ORDER`` is absent. No order is required when ``reorder`` is false
            or retained fields do not need permutation.

        Notes
        -----
        State-turbine arrays must lead with ``(FC.STATE, FC.TURBINE)``;
        trailing dimensions are preserved. Model configuration is not mutated.

        """
        # case not output:
        if self.output is None:
            for data in (mdata, fdata, tdata):
                if data is not None:
                    for v in list(data.keys()):
                        del data[v], data.dims[v]
            return {}

        # get variables:
        vrs_mdata = self.output.mdata_variables
        vrs_fdata = self.output.fdata_variables
        vrs_tdata = self.output.tdata_variables

        # apply evaluation:
        if self._eval is not None:
            res = self._eval.calculate(
                algo=algo,
                mdata=mdata,
                fdata=fdata,
            )
            fdata.update(res)
            del res

        # loop over targets:
        order = fdata.get(FV.ORDER)
        ssel = None
        order_inv = None
        for vrs, data in zip((vrs_mdata, vrs_fdata, vrs_tdata), (mdata, fdata, tdata)):
            if data is None:
                continue
            all_vrs = list(set(data.keys()) - set(data.sizes.keys()))
            if vrs is None:
                vrs = all_vrs
            for v in all_vrs:
                dta = data.pop(v)
                dms = data.dims.pop(v)
                if v in vrs:
                    if FC.STATE in dms and FC.TURBINE in dms:
                        assert dms[:2] == (FC.STATE, FC.TURBINE), (
                            f"{self.name}: Require dimensions of '{v}' to match ({FC.STATE}, {FC.TURBINE}, ...), got {dms}"
                        )

                        # Reorder physical fields, retaining the downwind order metadata.
                        if (
                            v != FV.ORDER
                            and self.reorder
                            and dta.shape[1] > 1
                            and np.any(dta != dta[0, 0, None, None])
                        ):
                            if ssel is None:
                                if order is None:
                                    raise KeyError(
                                        f"{self.name}: Farm reordering requires '{FV.ORDER}'"
                                    )
                                ssel = np.broadcast_to(
                                    np.arange(order.shape[0])[:, None], order.shape
                                )
                                order_inv = np.empty_like(order)
                                order_inv[ssel, order] = np.arange(order.shape[1])[
                                    None, :
                                ]
                            dta = dta[ssel, order_inv, ...]

                    data[v] = dta
                    data.dims[v] = dms

        return {}

    def finalize(self, algo: Algorithm, verbosity: int = 0) -> None:
        """
        Finalize the model.

        Parameters
        ----------
        algo
            The calculation algorithm.
        verbosity
            The verbosity level; ``0`` is silent.

        """
        super().finalize(algo=algo, verbosity=verbosity)
        self._eval = None
