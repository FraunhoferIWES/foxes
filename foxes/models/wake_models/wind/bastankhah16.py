from __future__ import annotations

import numpy as np
from typing import TYPE_CHECKING, Any

from foxes.models.wake_models.dist_sliced import DistSlicedWakeModel
from foxes.core import Model, WakeK
from foxes.config import config
import foxes.variables as FV
import foxes.constants as FC

if TYPE_CHECKING:
    from foxes.core.algorithm import Algorithm
    from foxes.core.data import FData, MData, TData
    from foxes.core.model import LoadedData, Model


class Bastankhah2016Model(Model):
    """
    Common calculations for the wake model and the wake
    frame, such that code repetitions can be avoided.

    Notes
    -----
    Reference:
    "Experimental and theoretical study of wind turbine wakes in yawed conditions"
    Majid Bastankhah, Fernando Porté-Agel
    https://doi.org/10.1017/jfm.2016.595
    """

    MDATA_KEY = "Bastankhah2016Model"
    PARS = "pars"
    CHECK = "check"
    ST_SEL = "st_sel"
    X0 = "x0"

    NEAR = "near"
    R_PC = "r_pc"
    R_PC_S = "r_pc_s"
    AMPL_NEAR = "ampl_near"
    DELTA_NEAR = "delta_near"

    AMPL_FAR = "ampl_far"
    SIGMA_Y_FAR = "sigma_y_far"
    SIGMA_Z_FAR = "sigma_z_far"
    DELTA_FAR = "delta_far"

    def __init__(self, alpha: float, beta: float, induction: str) -> None:
        """
        Parameters
        ----------
        alpha
            model parameter used to determine onset of far wake region
        beta
            model parameter used to determine onset of far wake region
        induction
            The induction model
        """
        super().__init__()
        self.induction = induction
        setattr(self, FV.PA_ALPHA, alpha)
        setattr(self, FV.PA_BETA, beta)

    def sub_models(self) -> list[Model]:
        """
        List of all sub-models

        Returns
        -------
        smdls
            All sub models

        """
        return [] if isinstance(self.induction, str) else [self.induction]

    def initialize(
        self,
        algo: Algorithm,
        loaded_data: LoadedData | None = None,
        force: bool = False,
        verbosity: int = 0,
    ) -> LoadedData:
        """
        Initializes the model.

        Parameters
        ----------
        algo
            The calculation algorithm
        loaded_data
            Data that has already been loaded, to be extended by this function.
            Keys are "coords", a dict with entries `dim_name_str -> dim_array`;
            "data_vars", a dict with entries `name_str -> (dim_tuple, data_ndarray)`;
            and "extra_data", a dict with non-array additional data.
        force
            Overwrite existing data
        verbosity
            The verbosity level, 0 = silent

        Returns
        -------
        loaded_data
            The loaded data, containing keys "coords", "data_vars", and "extra_data".
            Keys are "coords", a dict with entries `dim_name_str -> dim_array`;
            "data_vars", a dict with entries `name_str -> (dim_tuple, data_ndarray)`;
            and "extra_data", a dict with non-array additional data.

        """
        if isinstance(self.induction, str):
            self.induction = algo.mbook.axial_induction[self.induction]
        return super().initialize(
            algo, loaded_data=loaded_data, force=force, verbosity=verbosity
        )

    @property
    def pars(self) -> dict[str, float | str]:
        """
        Dictionary of the model parameters

        Returns
        -------
        dict :
            Dictionary of the model parameters

        """
        alpha = getattr(self, FV.PA_ALPHA)
        beta = getattr(self, FV.PA_BETA)
        iname = (
            self.induction if isinstance(self.induction, str) else self.induction.name
        )
        return dict(alpha=alpha, beta=beta, induction=iname)

    def calc_data(
        self,
        algo: Algorithm,
        mdata: MData,
        fdata: FData,
        tdata: TData,
        downwind_index: int,
        x: np.ndarray,
        gamma: np.ndarray,
        k: np.ndarray,
    ) -> None:
        """
        Calculate common model data, store it in mdata.

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
        downwind_index
            The index in the downwind order
        x
            The x values, shape: (n_states, n_targets)
        gamma
            The YAWM angles in radiants, shape: (n_states, n_targets)
        k
            The k parameter values, shape: (n_states, n_targets)

        """
        # store parameters:
        assert not isinstance(self.induction, str)
        out: dict[str, Any] = {self.PARS: self.pars}
        out[self.CHECK] = (
            mdata.states_i0(counter=True),
            mdata.n_states,
            downwind_index,
            hash(x.tobytes()),
        )

        # get D:
        D = super().get_data(
            FV.D,
            target=FC.STATE_TARGET,
            lookup="w",
            algo=algo,
            fdata=fdata,
            tdata=tdata,
            downwind_index=downwind_index,
            upcast=True,
        )

        # get ct:
        ct = super().get_data(
            FV.CT,
            target=FC.STATE_TARGET,
            lookup="w",
            algo=algo,
            fdata=fdata,
            tdata=tdata,
            downwind_index=downwind_index,
            upcast=True,
        )

        # select targets:
        st_sel = (x > 1e-8) & (ct > 1e-8)
        if np.any(st_sel):
            # get ws:
            ws = super().get_data(
                FV.REWS,
                target=FC.STATE_TARGET,
                lookup="w",
                algo=algo,
                fdata=fdata,
                tdata=tdata,
                downwind_index=downwind_index,
                upcast=False,
                selection=st_sel,
            )

            # get TI:
            ti = super().get_data(
                FV.TI,
                target=FC.STATE_TARGET,
                lookup="w",
                algo=algo,
                fdata=fdata,
                tdata=tdata,
                downwind_index=downwind_index,
                upcast=False,
                selection=st_sel,
            )

            # get alpha:
            alpha = super().get_data(
                FV.PA_ALPHA,
                target=FC.STATE_TARGET,
                lookup="ws",
                algo=algo,
                fdata=fdata,
                tdata=tdata,
                downwind_index=downwind_index,
                upcast=False,
                selection=st_sel,
            )

            # get beta:
            beta = super().get_data(
                FV.PA_BETA,
                target=FC.STATE_TARGET,
                lookup="ws",
                algo=algo,
                fdata=fdata,
                tdata=tdata,
                downwind_index=downwind_index,
                upcast=False,
                selection=st_sel,
            )

            # apply filter:
            x = x[st_sel]
            D = D[st_sel]
            ct = ct[st_sel]
            k = k[st_sel]
            gamma = gamma[st_sel]

            # calc theta_c0, Eq. (6.12):
            cosg = np.cos(gamma)
            twoac = 2 * self.induction.ct2a(ct * cosg)
            theta = 0.3 * gamma / cosg * twoac

            # calculate x0, Eq. (7.3):
            twoa = 2 * self.induction.ct2a(ct)
            x0 = D * cosg * (2 - twoa) / (np.sqrt(2) * (4 * alpha * ti + beta * twoa))
            out[self.X0] = x0

            # calcuate sigma, Eq. (7.2):
            sigma_y0 = D * cosg / np.sqrt(8)
            simga_z0 = D / np.sqrt(8)
            sigma_y = k * (x - x0) + sigma_y0
            sigma_z = k * (x - x0) + simga_z0

            # calc near wake data:
            near = x < x0
            out[self.NEAR] = near
            if np.any(near):
                # apply filter:
                ctn = ct[near]
                cosgn = cosg[near]
                twoan = twoa[near]
                twoacn = twoac[near]

                # initial velocity deficits, Eq. (6.4):
                uR = 0.5 * ctn * cosgn / twoacn

                # constant potential core value, Eq. (6.7):
                u0 = 1 - twoan

                # compute potential core shape, for later, Eq. (6.13):
                d = x[near] / x0[near]
                r_pc_0 = 0.5 * D[near] * np.sqrt(uR / u0)  # radius at x=0
                r_pc = r_pc_0 - d * r_pc_0  # potential core radius

                # memorize near wake data:
                out[self.R_PC] = r_pc
                out[self.R_PC_S] = d * sigma_y0[near]
                out[self.DELTA_NEAR] = theta[near] * x[near]
                out[self.AMPL_NEAR] = u0 - 1

                # cleanup:
                del ctn, cosgn, uR, u0, d, r_pc_0, r_pc, twoan, twoacn

            # calc far wake data:
            far = ~near
            if np.any(far):
                # apply filter:
                ws = ws[far]
                ct = ct[far]
                sigma_y = sigma_y[far]
                sigma_z = sigma_z[far]
                cosg = cosg[far]
                D = D[far]
                theta = theta[far]
                x0 = x0[far]
                k = k[far]
                twoa = twoa[far]

                # calculate delta, Eq. (7.4):
                sqct = np.sqrt(ct)
                sqsd = np.sqrt(8 * sigma_y * sigma_z / (cosg * D**2))
                delta = theta * x0 + (
                    D
                    * theta
                    / 14.7
                    * np.sqrt(cosg / (ct * k**2))
                    * (2.9 + 1.3 * (1 - twoa) - ct)
                    * np.log(
                        ((1.6 + sqct) * (1.6 * sqsd - sqct))
                        / ((1.6 - sqct) * (1.6 * sqsd + sqct))
                    )
                )

                # calculate amplitude, Eq. (7.1):
                ct_eff = ct * cosg * D**2 / (8 * sigma_y * sigma_z)
                ampl = np.maximum(-2 * self.induction.ct2a(ct_eff), -1)

                # memorize far wake data:
                out[self.AMPL_FAR] = ampl
                out[self.DELTA_FAR] = delta
                out[self.SIGMA_Y_FAR] = sigma_y
                out[self.SIGMA_Z_FAR] = sigma_z

        # update mdata:
        out[self.ST_SEL] = st_sel
        mdata.add(self.MDATA_KEY, out, None)

    def has_data(self, mdata: MData, downwind_index: int, x: np.ndarray) -> bool:
        """
        Check if data exists

        Parameters
        ----------
        mdata
            The model data
        downwind_index
            For each state, one turbine index for the
            wake causing turbine. Shape: (n_states,)
        x
            The x values, shape: (n_states, n_points)

        Returns
        -------
        check
            True if data exists

        """
        check = (
            mdata.states_i0(counter=True),
            mdata.n_states,
            downwind_index,
            hash(x.tobytes()),
        )
        return self.MDATA_KEY in mdata and mdata[self.MDATA_KEY][self.CHECK] == check

    def get_cached_data(self, key: str, mdata: MData) -> Any:
        """
        Return data entry

        Parameters
        ----------
        key
            The data key
        mdata
            The model data

        Returns
        -------
        data
            The data

        """
        return mdata[self.MDATA_KEY][key]

    def clean(self, mdata: MData) -> None:
        """
        Clean all data
        """
        del mdata[self.MDATA_KEY]


class Bastankhah2016(DistSlicedWakeModel):
    """
    The Bastankhah 2016 wake model

    Notes
    -----
    Reference:
    "Experimental and theoretical study of wind turbine wakes in yawed conditions"
    Majid Bastankhah, Fernando Porté-Agel
    https://doi.org/10.1017/jfm.2016.595
    """

    def __init__(
        self,
        superposition: str,
        alpha: float = 0.58,
        beta: float = 0.077,
        induction: str = "Madsen",
        **wake_k: Any,
    ) -> None:
        """
        Parameters
        ----------
        superposition
            The wind deficit superposition
        ct_max
            The maximal value for ct, values beyond will be limited
            to this number, by default 0.9999
        alpha
            model parameter used to determine onset of far wake region
        beta
            model parameter used to determine onset of far wake region
        induction
            The induction model
        wake_k
            Parameters for the WakeK class
        """
        super().__init__(wind_superposition=superposition)

        self.model: Bastankhah2016Model | None = None
        self.alpha = alpha
        self.beta = beta
        self.induction = induction
        self.wake_k = WakeK(**wake_k)

        setattr(self, FV.YAWM, 0.0)

    def __repr__(self) -> str:
        iname = self.induction
        s = f"{type(self).__name__}"
        s += f"({self.wind_superposition}, induction={iname}, "
        s += self.wake_k.repr() + ")"
        return s

    def sub_models(self) -> list[Model]:
        """
        List of all sub-models

        Returns
        -------
        smdls
            Names of all sub models

        """
        smdls = super().sub_models() + [self.wake_k]
        if self.model is not None:
            smdls.append(self.model)
        return smdls

    def waked_variables(self) -> list[str]:
        """
        Returns a list of variable names that are affected by this wake model.

        Returns
        -------
        waked_variables
            A list of variable names affected by this wake model.

        """
        return [FV.WS]

    def initialize(
        self,
        algo: Algorithm,
        loaded_data: LoadedData | None = None,
        force: bool = False,
        verbosity: int = 0,
    ) -> LoadedData:
        """
        Initializes the model.

        Parameters
        ----------
        algo
            The calculation algorithm
        loaded_data
            Data that has already been loaded, to be extended by this function.
            Keys are "coords", a dict with entries `dim_name_str -> dim_array`;
            "data_vars", a dict with entries `name_str -> (dim_tuple, data_ndarray)`;
            and "extra_data", a dict with non-array additional data.
        force
            Overwrite existing data
        verbosity
            The verbosity level, 0 = silent

        Returns
        -------
        loaded_data
            The loaded data, containing keys "coords", "data_vars", and "extra_data".
            Keys are "coords", a dict with entries `dim_name_str -> dim_array`;
            "data_vars", a dict with entries `name_str -> (dim_tuple, data_ndarray)`;
            and "extra_data", a dict with non-array additional data.

        """
        if not self.initialized:
            self.model = Bastankhah2016Model(
                alpha=self.alpha, beta=self.beta, induction=self.induction
            )
        return super().initialize(
            algo, loaded_data=loaded_data, force=force, verbosity=verbosity
        )

    def calc_wakes_x_yz(
        self,
        algo: Algorithm,
        mdata: MData,
        fdata: FData,
        tdata: TData,
        downwind_index: int,
        x: np.ndarray,
        yz: np.ndarray,
    ) -> tuple[dict[str, np.ndarray], np.ndarray]:
        """
        Calculate wake deltas.

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
        downwind_index
            The index in the downwind order
        x
            The x values, shape: (n_states, n_targets)
        yz
            The yz values for each x value, shape:
            (n_states, n_targets, n_yz_per_target, 2)

        Returns
        -------
        wdeltas
            The wake deltas. Key: variable name str,
            value
        st_sel
            The state-target selection, for which the wake
            is non-zero, shape: (n_states, n_targets)

        """
        # prepare:
        assert self.model is not None
        n_y_per_z = yz.shape[2]

        # calculate model data:
        if not self.model.has_data(mdata, downwind_index, x):
            # get gamma:
            gamma = self.get_data(
                FV.YAWM,
                FC.STATE_TARGET,
                lookup="ws",
                algo=algo,
                fdata=fdata,
                tdata=tdata,
                upcast=True,
                downwind_index=downwind_index,
            )
            gamma = gamma * np.pi / 180

            # get k:
            k = self.wake_k(
                FC.STATE_TARGET,
                algo=algo,
                fdata=fdata,
                tdata=tdata,
                upcast=True,
                downwind_index=downwind_index,
            )

            # run calculation:
            self.model.calc_data(algo, mdata, fdata, tdata, downwind_index, x, gamma, k)

        # select targets:
        st_sel = self.model.get_cached_data(Bastankhah2016Model.ST_SEL, mdata)
        n_sp_sel = np.sum(st_sel)
        wdeltas = {FV.WS: np.zeros((n_sp_sel, n_y_per_z), dtype=config.dtype_double)}
        if np.any(st_sel):
            # apply filter:
            yz = yz[st_sel]

            # collect data:
            near = self.model.get_cached_data(Bastankhah2016Model.NEAR, mdata)
            far = ~near

            # near wake:
            if np.any(near):
                # collect data:
                ampl = self.model.get_cached_data(Bastankhah2016Model.AMPL_NEAR, mdata)
                r_pc = self.model.get_cached_data(Bastankhah2016Model.R_PC, mdata)
                s = self.model.get_cached_data(Bastankhah2016Model.R_PC_S, mdata)

                # radial dependency:
                r = np.linalg.norm(yz[near], axis=-1)
                rfactor = np.ones_like(r)
                sel_oc = np.where(r > r_pc[:, None])
                r = r[sel_oc]
                r_pc = r_pc[sel_oc[0]]
                s = s[sel_oc[0]]
                rfactor[sel_oc] = np.exp(-0.5 * ((r - r_pc) / s) ** 2)

                # set deficit, Eq. (6.13):
                wdeltas[FV.WS][near] = ampl[:, None] * rfactor

            # far wake:
            if np.any(far):
                # apply filter:
                yz = yz[far]

                # collect data:
                ampl = self.model.get_cached_data(Bastankhah2016Model.AMPL_FAR, mdata)[
                    :, None
                ]
                sigma_y = self.model.get_cached_data(
                    Bastankhah2016Model.SIGMA_Y_FAR, mdata
                )[:, None]
                sigma_z = self.model.get_cached_data(
                    Bastankhah2016Model.SIGMA_Z_FAR, mdata
                )[:, None]

                # set deficit, Eq. (7.1):
                y = yz[..., 0]
                z = yz[..., 1]
                wdeltas[FV.WS][far] = ampl * (
                    np.exp(-0.5 * (y / sigma_y) ** 2)
                    * np.exp(-0.5 * (z / sigma_z) ** 2)
                )

        # wake deflection causes wind vector rotation:
        if FC.WDEFL_ROT_ANGLE in tdata:
            dwd_defl = tdata.pop(FC.WDEFL_ROT_ANGLE)
            if FV.WD not in wdeltas:
                wdeltas[FV.WD] = np.zeros_like(wdeltas[FV.WS])
                wdeltas[FV.WD][:] = dwd_defl[st_sel]
            else:
                wdeltas[FV.WD] += dwd_defl[st_sel]

        # wake deflection causes wind speed reduction:
        if FC.WDEFL_DWS_FACTOR in tdata:
            dws_defl = tdata.pop(FC.WDEFL_DWS_FACTOR)
            if FV.WS not in wdeltas:
                raise AssertionError(
                    f"Wake model '{self.name}': Expecting '{FV.WS}' in wdeltas, found {list(wdeltas.keys())}"
                )
            else:
                wdeltas[FV.WS] *= dws_defl[st_sel]

        return wdeltas, st_sel
