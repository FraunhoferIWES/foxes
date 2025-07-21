import numpy as np

from foxes.core import TData
import foxes.variables as FV
import foxes.constants as FC

from .centre import CentreRotor


class DirectMDataInfusion(CentreRotor):
    """
    Direct data infusion of data stored under mdata.

    Attributes
    ----------
    svars2mdvars: dict
        A mapping of state variables to mdata variables.
    mdata_vars: list of str
        The mdata variables to be used for infusion. By default,
        all mdata variables are searched.
    turbine_coord: str, optional
        The mdata coordinate that represents turbine names. By default,
        the second coordinate is used as a candidate if the mdata variable
        has three dimensions.

    :group: models.turbine_types

    """

    def __init__(
        self,
        svars2mdvars=None,
        mdata_vars=None,
        turbine_coord=None,
        **kwargs,
    ):
        """
        Constructor.

        Parameters
        ----------
        svars2mdvars: dict, optional
            A mapping of state variables to mdata variables.
        mdata_vars: list of str, optional
            The mdata variables to be used for infusion. By default,
            all mdata variables are searched.
        turbine_coord: str, optional
            The mdata coordinate that represents turbine names. By default,
            the second coordinate is used as a candidate if the mdata variable
            has three dimensions.
        kwargs: dict, optional
            Additional parameters for RotorModel class

        """
        super().__init__(**kwargs)
        self.svars2mdvars = svars2mdvars
        self.mdata_vars = mdata_vars
        self.turbine_coord = turbine_coord

    def output_farm_vars(self, algo):
        """
        The variables which are being modified by the model.

        Parameters
        ----------
        algo: foxes.core.Algorithm
            The calculation algorithm

        Returns
        -------
        output_vars: list of str
            The output variable names

        """
        if self.svars2mdvars is None:
            self.svars2mdvars = {v: v for v in algo.states.output_point_vars(algo)}

        if self.calc_vars is None:
            vrs = list(self.svars2mdvars.keys())
            assert FV.WEIGHT not in vrs, (
                f"Rotor '{self.name}': svars2mdvars keys {vrs}  contain '{FV.WEIGHT}', please remove"
            )

            if FV.WS in vrs:
                self.calc_vars = [FV.REWS] + [v for v in vrs if v != FV.WS]
            else:
                self.calc_vars = vrs

            if algo.farm_controller.needs_rews2() and FV.REWS2 not in self.calc_vars:
                self.calc_vars.append(FV.REWS2)
            if algo.farm_controller.needs_rews3() and FV.REWS3 not in self.calc_vars:
                self.calc_vars.append(FV.REWS3)

            self.calc_vars = sorted(self.calc_vars)

        if FV.WEIGHT not in self.calc_vars:
            self.calc_vars.append(FV.WEIGHT)
            if FV.WEIGHT not in self.svars2mdvars:
                self.svars2mdvars[FV.WEIGHT] = FV.WEIGHT

        return self.calc_vars

    def calculate(
        self,
        algo,
        mdata,
        fdata,
        rpoints=None,
        rpoint_weights=None,
        store=False,
        downwind_index=None,
    ):
        """
        Calculate ambient rotor effective results.

        Parameters
        ----------
        algo: foxes.core.Algorithm
            The calculation algorithm
        mdata: foxes.core.MData
            The model data
        fdata: foxes.core.FData
            The farm data
        rpoints: numpy.ndarray, optional
            The rotor points, or None for automatic for
            this rotor. Shape: (n_states, n_turbines, n_rpoints, 3)
        rpoint_weights: numpy.ndarray, optional
            The rotor point weights, or None for automatic
            for this rotor. Shape: (n_rpoints,)
        store: bool, optional
            Flag for storing ambient rotor point results
        downwind_index: int, optional
            Only compute for index in the downwind order

        Returns
        -------
        results: dict
            results dict. Keys: Variable name str. Values:
            numpy.ndarray with results, shape: (n_states, n_turbines)

        """
        self.ensure_output_vars(algo, fdata)

        if rpoints is None:
            rpoints = mdata.get(
                FC.ROTOR_POINTS, self.get_rotor_points(algo, mdata, fdata)
            )
        if downwind_index is not None:
            rpoints = rpoints[:, downwind_index, None]
        if rpoint_weights is None:
            rpoint_weights = mdata.get_item(FC.TWEIGHTS, self.rotor_point_weights())

        tdata = TData.from_tpoints(rpoints, rpoint_weights)
        sres = {}
        mdvs = self.mdata_vars if self.mdata_vars is not None else list(mdata.keys())
        for v, w in self.svars2mdvars.items():
            tdata.add(
                v,
                data=np.full_like(rpoints[..., 0], np.nan),
                dims=(FC.STATE, FC.TARGET, FC.TPOINT),
            )

            # check fixed variables
            if hasattr(algo.states, "fixed_vars") and v in algo.states.fixed_vars:
                tdata[v][:] = algo.states.fixed_vars[v]
                sres[v] = tdata[v]
                continue

            # search in mdata variables
            tcoord = self.turbine_coord
            for mdv in mdvs:
                assert mdv in mdata and mdv in mdata.dims, (
                    f"Rotor '{self.name}': mdata variable '{mdv}' not found in mdata {list(mdata.keys())} with dims {list(mdata.dims.keys())}"
                )
                mdat = mdata[mdv]
                dims = mdata.dims[mdv]

                # skip coordinates in the search:
                if dims == (mdv,):
                    continue

                # find variable index in last data array dimension
                vc = dims[-1]
                assert vc in mdata and vc in mdata.dims and mdata.dims[vc] == (vc,), (
                    f"Rotor '{self.name}': mdata coordinate '{vc}' not in mdata or wrong dimensions {mdata.dims}, expected '{(vc,)}'"
                )
                vrs = list(mdata[vc])
                if w in vrs:
                    i = vrs.index(w)
                    mdat = mdat[..., i]
                    dims = dims[:-1]

                    # pure state dependent variable
                    if dims == (FC.STATE,):
                        tdata[v][:] = mdat[:, None, None]

                    # state and turbine dependent variable
                    elif len(dims) == 2 and dims[0] == FC.STATE:
                        assert mdat.shape[1] == mdata.n_turbines, (
                            f"Rotor '{self.name}': mdata variable '{mdv}' has dimensions {dims} and unexpected shape {mdat.shape} for variable '{w}', expected ({mdata.n_states}, {mdata.n_turbines}) for this wind farm with {mdata.n_turbines} turbines"
                        )
                        if tcoord is None:
                            tcoord = dims[1]
                        assert dims[1] == tcoord, (
                            f"Rotor '{self.name}': mdata variable '{mdv}' has unexpected dimensions {dims} for variable '{w}', expected ({FC.STATE}, {tcoord})"
                        )

                        tdata[v][:] = mdat[:, :, None]

                    else:
                        if tcoord is None:
                            tcoord = "<turbine>"
                        raise ValueError(
                            f"Rotor '{self.name}': mdata variable '{mdv}' has unexpected dimensions {dims} for variable '{w}' at position {i}, expected ({FC.STATE},) or ({FC.STATE}, {tcoord})"
                        )

                    sres[v] = tdata[v]
                    break

            if v not in sres:
                raise ValueError(
                    f"Rotor '{self.name}': mdata variable '{w}' not found in any of the mdata variables {mdvs}"
                )

        if store:
            algo.add_to_chunk_store(FC.ROTOR_POINTS, rpoints, mdata=mdata)
            algo.add_to_chunk_store(FC.ROTOR_WEIGHTS, rpoint_weights, mdata=mdata)
            algo.add_to_chunk_store(FC.AMB_ROTOR_RES, sres, mdata=mdata)
            algo.add_to_chunk_store(FC.WEIGHT_RES, tdata[FV.WEIGHT], mdata=mdata)

        self.eval_rpoint_results(
            algo,
            mdata,
            fdata,
            tdata,
            rpoint_weights,
            downwind_index,
            copy_to_ambient=True,
        )

        return {v: fdata[v] for v in self.output_farm_vars(algo)}
