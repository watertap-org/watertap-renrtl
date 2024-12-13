#################################################################################
# WaterTAP Copyright (c) 2020-2023, The Regents of the University of California,
# through Lawrence Berkeley National Laboratory, Oak Ridge National Laboratory,
# National Renewable Energy Laboratory, and National Energy Technology
# Laboratory (subject to receipt of any required approvals from the U.S. Dept.
# of Energy). All rights reserved.
#
# Copyright 2023-2024, National Technology & Engineering Solutions of Sandia,
# LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the
# U.S. Government retains certain rights in this software.
#
# Copyright 2024, Nazia Aslam, Matthew D. Stuber, George M. Bollas, and the University of Connecticut.
#
# Please see the files COPYRIGHT.md and LICENSE.md for full copyright and license
# information, respectively. These files are also available online at the URL
# "https://github.com/watertap-org/watertap/"
#
# Authors: Nazia Aslam from the University of Connecticut and Soraya Rawlings
#################################################################################
"""
Property package for LiBr 
        
References:
[1] "Water." Wikipedia, The Free Encyclopedia. Wikipedia, The Free Encyclopedia, 
last modified June 5, 2024. https://en.wikipedia.org/wiki/Water.
        
[2] "Lithium bromide." Wikipedia, The Free Encyclopedia. Wikipedia, 
The Free Encyclopedia, last modified June 1, 2024. https://en.wikipedia.org/wiki/Lithium_bromide.
        
[3] Sharqawy, Mostafa H.; Lienhard, John H.; Zubair, Syed M. (2010). 
Thermophysical properties of seawater: a review of existing correlations and data. 
Desalination and Water Treatment, 16(1-3), 354-380. doi:10.5004/dwt.2010.1079
           
[4] Hellmann, H. M., & Grossman, G. (1996). 
Improved property data correlations of absorption fluids for computer simulation of heat pump cycles. 
ASHRAE Transactions, 102(1), 980-997.
        
[5] G.A. Florides, S.A. Kalogirou, S.A. Tassou, L.C. Wrobel, 
Design and construction of a LiBr-water absorption machine,
Energy Conversion & Management, 2002. doi: 10.1016/S0196-8904(03)00006-2

"""
# Import Pyomo library and components
import pyomo.environ as pyo
from pyomo.environ import (
    Constraint,
    Expression,
    Reals,
    NonNegativeReals,
    Param,
    Suffix,
    value,
    log,
    log10,
    exp,
    check_optimal_termination,
)
from pyomo.environ import units as pyunits

# Import IDAES cores
import idaes.logger as idaeslog
from idaes.core import (
    declare_process_block_class,
    MaterialFlowBasis,
    PhysicalParameterBlock,
    StateBlockData,
    StateBlock,
    MaterialBalanceType,
    EnergyBalanceType,
)
from idaes.core.base.components import Solute, Solvent
from idaes.core.base.phases import LiquidPhase
from idaes.core.util.constants import Constants
from idaes.core.util.initialization import (
    fix_state_vars,
    revert_state_vars,
    solve_indexed_blocks,
)
from idaes.core.solvers import get_solver
from idaes.core.util.model_statistics import (
    degrees_of_freedom,
    number_unfixed_variables,
)
from idaes.core.util.exceptions import (
    ConfigurationError,
    InitializationError,
    PropertyPackageError,
)
import idaes.core.util.scaling as iscale
from watertap.core.util.scaling import transform_property_constraints

# Set up logger
_log = idaeslog.getLogger(__name__)


@declare_process_block_class("LiBrParameterBlock")
class LiBrParameterData(PhysicalParameterBlock):
    """Parameter block for a LiBr property package."""

    CONFIG = PhysicalParameterBlock.CONFIG()

    def build(self):
        """
        Callable method for Block construction.
        """
        super(LiBrParameterData, self).build()

        self._state_block_class = LiBrStateBlock

        # components
        self.H2O = Solvent()
        self.TDS = Solute()

        # phases
        self.Liq = LiquidPhase()

        # Parameters
        mw_comp_data = {
            "H2O": 18.01528e-3,  # from ref [1]
            "TDS": 86.845e-3,  # from ref [2]
        }

        self.mw_comp = Param(
            self.component_list,
            initialize=mw_comp_data,
            units=pyunits.kg / pyunits.mol,
            doc="Molecular weight",
        )
        # Density of pure water (kg/m3)
        # Validity: 0 < t < 180 oC; 0 < S < 0.16 kg/kg
        # from ref [3]

        dens_units = pyunits.kg / pyunits.m**3
        t_inv_units = pyunits.K**-1
        s_inv_units = pyunits.kg / pyunits.g

        self.dens_mass_param_A1 = pyo.Param(
            within=Reals,
            initialize=9.999e2,
            units=dens_units,
            doc="Mass density parameter A1 for pure water",
        )
        self.dens_mass_param_A2 = pyo.Param(
            within=Reals,
            initialize=2.034e-2,
            units=dens_units * t_inv_units,
            doc="Mass density parameter A2 for pure water",
        )
        self.dens_mass_param_A3 = pyo.Param(
            within=Reals,
            initialize=-6.162e-3,
            units=dens_units * t_inv_units**2,
            doc="Mass density parameter A3 for pure water",
        )
        self.dens_mass_param_A4 = pyo.Param(
            within=Reals,
            initialize=2.261e-5,
            units=dens_units * t_inv_units**3,
            doc="Mass density parameter A4 for pure water",
        )
        self.dens_mass_param_A5 = pyo.Param(
            within=Reals,
            initialize=-4.657e-8,
            units=dens_units * t_inv_units**4,
            doc="Mass density parameter A5 for pure water",
        )

        # Density of LiBr solution (kg/m3)
        # Validity: 0 < t < 200 oC; 0.2 < X < 0.65
        # from ref [4]
        self.dens_mass_param_B1 = pyo.Param(
            within=Reals,
            initialize=1145.36,
            units=dens_units,
            doc="Mass density parameter B1 for LiBr",
        )
        self.dens_mass_param_B2 = pyo.Param(
            within=Reals,
            initialize=470.84,
            units=dens_units,
            doc="Mass density parameter B2 for LiBr",
        )
        self.dens_mass_param_B3 = pyo.Param(
            within=Reals,
            initialize=1374.79,
            units=dens_units,
            doc="Mass density parameter B3",
        )
        self.dens_mass_param_B4 = pyo.Param(
            within=Reals,
            initialize=0.333393,
            units=dens_units / pyunits.K,
            doc="Mass density parameter B4",
        )

        self.dens_mass_param_B5 = pyo.Param(
            within=Reals,
            initialize=0.571749,
            units=dens_units / pyunits.K,
            doc="Mass density parameter B5",
        )

        # Absolute viscosity of LiBr solution (Pa*s)
        # Validity: 0 < t < 200 oC; 0.45 < X < 0.65
        # from ref [5]
        visc_d_units = pyunits.Pa * pyunits.s

        self.visc_d_param_A = pyo.Param(
            within=Reals,
            initialize=-494.122,
            units=visc_d_units,
            doc="Dynamic viscosity parameter A",
        )
        self.visc_d_param_B = pyo.Param(
            within=Reals,
            initialize=16.3967,
            units=visc_d_units,
            doc="Dynamic viscosity parameter B",
        )
        self.visc_d_param_C = pyo.Param(
            within=Reals,
            initialize=0.14511,
            units=visc_d_units,
            doc="Dynamic viscosity parameter C",
        )
        self.visc_d_param_D = pyo.Param(
            within=Reals,
            initialize=28606.4,
            units=visc_d_units * t_inv_units,
            doc="Dynamic viscosity parameter D",
        )

        self.visc_d_param_E = pyo.Param(
            within=Reals,
            initialize=934.568,
            units=visc_d_units * t_inv_units,
            doc="Dynamic viscosity parameter D",
        )
        self.visc_d_param_F = pyo.Param(
            within=Reals,
            initialize=8.52755,
            units=visc_d_units * t_inv_units,
            doc="Dynamic viscosity parameter F",
        )
        self.visc_d_param_G = pyo.Param(
            within=Reals,
            initialize=70.3848,
            units=visc_d_units,
            doc="Dynamic viscosity parameter G",
        )
        self.visc_d_param_H = pyo.Param(
            within=Reals,
            initialize=2.35014,
            units=visc_d_units,
            doc="Dynamic viscosity parameter H",
        )
        self.visc_d_param_I = pyo.Param(
            within=Reals,
            initialize=0.0207809,
            units=visc_d_units,
            doc="Dynamic viscosity parameter I",
        )

        # Saturation temp (boiling temp) in K of a LiBr solution given pressure and mass fraction of LiBr
        # from ref [4]
        a_list = ["a1", "a2", "a3", "a4", "a5", "a6", "a7", "a8", "a9", "a10", "a11"]
        b_list = ["b1", "b2", "b3", "b4", "b5", "b6", "b7", "b8", "b9", "b10", "b11"]
        self.set_a = pyo.Set(initialize=a_list)
        self.set_b = pyo.Set(initialize=b_list)
        self.temperature_sat_param_a = {
            "a1": 0,
            "a2": 16.634856,
            "a3": -553.38169,
            "a4": 11228.336,
            "a5": -110283.9,
            "a6": 621094.64,
            "a7": -2111256.7,
            "a8": 4385190.1,
            "a9": -5409811.5,
            "a10": 3626674.2,
            "a11": -1015305.9,
        }
        self.temperature_sat_param_b = {
            "b1": 1,
            "b2": -0.068242821,
            "b3": 5.873619,
            "b4": -102.78186,
            "b5": 930.32374,
            "b6": -4822.394,
            "b7": 15189.038,
            "b8": -29412.863,
            "b9": 34100.528,
            "b10": -21671.48,
            "b11": 5799.56,
        }

        # Refrigerant saturation pressure at refrence saturation temperature
        # Validity: 0.45 < X < 0.70
        # from ref [5]
        self.pressure_sat_param_A1 = pyo.Param(
            within=Reals,
            initialize=7.05,
            units=pyunits.dimensionless,
            doc="Saturation pressure parameter A1",
        )

        self.pressure_sat_param_A2 = pyo.Param(
            within=Reals,
            initialize=-1596.49,
            units=pyunits.K,
            doc="Saturation pressure parameter A2",
        )

        self.pressure_sat_param_A3 = pyo.Param(
            within=Reals,
            initialize=-104095.5,
            units=pyunits.K**2,
            doc="Saturation pressure parameter A3",
        )

        # Water vapor saturation temperature (K) at given pressure
        # Pressure P(start = 101325.0,min=0.01)
        # reverse Antoinne's equation

        t_units = pyunits.K
        self.temperature_sat_solvent_param_A1 = pyo.Param(
            within=Reals,
            initialize=42.67776,
            units=t_units,
            doc="Parameter A1",
        )
        self.temperature_sat_solvent_param_A2 = pyo.Param(
            within=Reals,
            initialize=3892.7,
            units=t_units,
            doc="Parameter A2",
        )
        self.temperature_sat_solvent_param_A3 = pyo.Param(
            within=Reals,
            initialize=9.48654,
            units=pyunits.dimensionless,
            doc="Parameter A3",
        )

        # Heat capacity of LiBr solution (J/(kg K)
        # Assumption: 0.4 < X < 0.7
        # from ref [5]
        cp_units = pyunits.J / (pyunits.kg * pyunits.K)
        self.cp_phase_param_A1 = pyo.Param(
            within=Reals,
            initialize=0.0976,
            units=cp_units,
            doc="Specific heat of LiBr parameter A1",
        )
        self.cp_phase_param_A2 = pyo.Param(
            within=Reals,
            initialize=37.512,
            units=cp_units,
            doc="Specific heat of LiBr parameter A2",
        )
        self.cp_phase_param_A3 = pyo.Param(
            within=Reals,
            initialize=3825.4,
            units=cp_units,
            doc="Specific heat of LiBr parameter A3",
        )

        # Thermal conductivity(W/(m K)) of LiBr solution at T(K) and X(g LiBr/g soln)
        # from ref [5]
        therm_cond_units = pyunits.W / pyunits.m / pyunits.K
        self.therm_cond_phase_param_1 = pyo.Param(
            within=Reals,
            initialize=-0.3081,
            units=therm_cond_units,
            doc="Thermal conductivity of LiBr parameter 1",
        )
        self.therm_cond_phase_param_2 = pyo.Param(
            within=Reals,
            initialize=0.62979,
            units=therm_cond_units,
            doc="Thermal conductivity of LiBr parameter 2",
        )
        self.therm_cond_phase_param_3 = pyo.Param(
            within=Reals,
            initialize=-0.3191795,
            units=therm_cond_units,
            doc="Thermal conductivity of LiBr parameter 3",
        )
        self.therm_cond_phase_param_4 = pyo.Param(
            within=Reals,
            initialize=0.65388,
            units=therm_cond_units,
            doc="Thermal conductivity of LiBr parameter 4",
        )
        self.therm_cond_phase_param_5 = pyo.Param(
            within=Reals,
            initialize=-0.291897,
            units=therm_cond_units,
            doc="Thermal conductivity of LiBr parameter 5",
        )
        self.therm_cond_phase_param_6 = pyo.Param(
            within=Reals,
            initialize=0.59821,
            units=therm_cond_units,
            doc="Thermal conductivity of LiBr parameter 6",
        )

        # Traditional parameters are the only Vars currently on the block and should be fixed
        for v in self.component_objects(pyo.Var):
            v.fix()

        # ---default scaling---
        self.set_default_scaling("temperature", 1e-2)
        self.set_default_scaling("pressure", 1e-6)
        self.set_default_scaling("dens_mass_phase", 1e-3, index="Liq")
        self.set_default_scaling("dens_mass_solvent", 1e-3)
        self.set_default_scaling("visc_d_phase", 1e3, index="Liq")
        self.set_default_scaling("enth_mass_phase", 1e-5, index="Liq")
        self.set_default_scaling("temperature_sat", 1e-2)
        self.set_default_scaling("temperature_sat_solvent", 1e-2)
        self.set_default_scaling("pressure_sat", 1e-6)
        self.set_default_scaling("cp_mass_phase", 1e-3, index="Liq")
        self.set_default_scaling("therm_cond_phase", 1e0, index="Liq")

    @classmethod
    def define_metadata(cls, obj):
        """Define properties supported and units."""
        obj.add_properties(
            {
                "flow_mass_phase_comp": {"method": None},
                "temperature": {"method": None},
                "pressure": {"method": None},
                "mass_frac_phase_comp": {"method": "_mass_frac_phase_comp"},
                "dens_mass_phase": {"method": "_dens_mass_phase"},
                "flow_vol_phase": {"method": "_flow_vol_phase"},
                "flow_vol": {"method": "_flow_vol"},
                "conc_mass_phase_comp": {"method": "_conc_mass_phase_comp"},
                "flow_mol_phase_comp": {"method": "_flow_mol_phase_comp"},
                "mole_frac_phase_comp": {"method": "_mole_frac_phase_comp"},
                "molality_phase_comp": {"method": "_molality_phase_comp"},
                "visc_d_phase": {"method": "_visc_d_phase"},
                "enth_mass_phase": {"method": "_enth_mass_phase"},
                "temperature_sat": {"method": "_temperature_sat"},
                "temperature_sat_solvent": {"method": "_temperature_sat_solvent"},
                "pressure_sat": {"method": "_pressure_sat"},
                "cp_mass_phase": {"method": "_cp_mass_phase"},
                "therm_cond_phase": {"method": "_therm_cond_phase"},
            }
        )

        obj.define_custom_properties(
            {
                "dens_mass_solvent": {"method": "_dens_mass_solvent"},
                "enth_flow": {"method": "_enth_flow"},
            }
        )

        obj.add_default_units(
            {
                "time": pyunits.s,
                "length": pyunits.m,
                "mass": pyunits.kg,
                "amount": pyunits.mol,
                "temperature": pyunits.K,
            }
        )


# This Class contains methods which should be applied to Property Blocks as a whole, rather than individual elements of indexed Property Blocks.
class _LiBrStateBlock(StateBlock):
    def fix_initialization_states(self):
        """
        Fixes state variables for state blocks.
        Returns:
            None
        """
        # Fix state variables
        fix_state_vars(self)

        # Constraint on water concentration at outlet - unfix in these cases
        for b in self.values():
            if b.config.defined_state is False:
                b.conc_mol_comp["H2O"].unfix()

    def initialize(
        self,
        state_args=None,
        state_vars_fixed=False,
        hold_state=False,
        outlvl=idaeslog.NOTSET,
        solver=None,
        optarg=None,
    ):
        """
        Initialization routine for property package.
        Keyword Arguments:
            state_args : Dictionary with initial guesses for the state vars
                         chosen. Note that if this method is triggered
                         through the control volume, and if initial guesses
                         were not provided at the unit model level, the
                         control volume passes the inlet values as initial
                         guess.The keys for the state_args dictionary are:
                         flow_mass_phase_comp : value at which to initialize
                                               phase component flows
                         pressure : value at which to initialize pressure
                         temperature : value at which to initialize temperature
            outlvl : sets output level of initialization routine
            optarg : solver options dictionary object (default={})
            state_vars_fixed: Flag to denote if state vars have already been
                              fixed.
                              - True - states have already been fixed by the
                                       control volume 1D. Control volume 0D
                                       does not fix the state vars, so will
                                       be False if this state block is used
                                       with 0D blocks.
                             - False - states have not been fixed. The state
                                       block will deal with fixing/unfixing.
            solver : Solver object to use during initialization if None is provided
                     it will use the default solver for IDAES (default = None)
            hold_state : flag indicating whether the initialization routine
                         should unfix any state variables fixed during
                         initialization (default=False).
                         - True - states variables are not unfixed, and
                                 a dict of returned containing flags for
                                 which states were fixed during
                                 initialization.
                        - False - state variables are unfixed after
                                 initialization by calling the
                                 release_state method
        Returns:
            If hold_states is True, returns a dict containing flags for
            which states were fixed during initialization.
        """
        # Get loggers
        init_log = idaeslog.getInitLogger(self.name, outlvl, tag="properties")
        solve_log = idaeslog.getSolveLogger(self.name, outlvl, tag="properties")

        # Set solver and options
        opt = get_solver(solver, optarg)

        # Fix state variables
        flags = fix_state_vars(self, state_args)
        # Check when the state vars are fixed already result in dof 0
        for k in self.keys():
            dof = degrees_of_freedom(self[k])
            if dof != 0:
                raise PropertyPackageError(
                    "State vars fixed but degrees of "
                    "freedom for state block is not "
                    "zero during initialization."
                )

        # ---------------------------------------------------------------------
        skip_solve = True  # skip solve if only state variables are present
        for k in self.keys():
            if number_unfixed_variables(self[k]) != 0:
                skip_solve = False

        if not skip_solve:
            # Initialize properties
            with idaeslog.solver_log(solve_log, idaeslog.DEBUG) as slc:
                results = solve_indexed_blocks(opt, [self], tee=slc.tee)
            init_log.info_high(
                "Property initialization: {}.".format(idaeslog.condition(results))
            )

        # If input block, return flags, else release state
        if state_vars_fixed is False:
            if hold_state is True:
                return flags
            else:
                self.release_state(flags)

        if (not skip_solve) and (not check_optimal_termination(results)):
            raise InitializationError(
                f"{self.name} failed to initialize successfully. Please "
                f"check the output logs for more information."
            )

    def release_state(self, flags, outlvl=idaeslog.NOTSET):
        """
        Method to release state variables fixed during initialisation.
        Keyword Arguments:
            flags : dict containing information of which state variables
                    were fixed during initialization, and should now be
                    unfixed. This dict is returned by initialize if
                    hold_state=True.
            outlvl : sets output level of of logging
        """
        # Unfix state variables
        init_log = idaeslog.getInitLogger(self.name, outlvl, tag="properties")
        revert_state_vars(self, flags)
        init_log.info("{} State Released.".format(self.name))

    def calculate_state(
        self,
        var_args=None,
        hold_state=False,
        outlvl=idaeslog.NOTSET,
        solver=None,
        optarg=None,
    ):
        """
        Solves state blocks given a set of variables and their values. These variables can
        be state variables or properties. This method is typically used before
        initialization to solve for state variables because non-state variables (i.e. properties)
        cannot be fixed in initialization routines.
        Keyword Arguments:
            var_args : dictionary with variables and their values, they can be state variables or properties
                       {(VAR_NAME, INDEX): VALUE}
            hold_state : flag indicating whether all of the state variables should be fixed after calculate state.
                         True - State variables will be fixed.
                         False - State variables will remain unfixed, unless already fixed.
            outlvl : idaes logger object that sets output level of solve call (default=idaeslog.NOTSET)
            solver : solver name string if None is provided the default solver
                     for IDAES will be used (default = None)
            optarg : solver options dictionary object (default={})
        Returns:
            results object from state block solve
        """
        # Get logger
        solve_log = idaeslog.getSolveLogger(self.name, level=outlvl, tag="properties")

        # Initialize at current state values (not user provided)
        self.initialize(solver=solver, optarg=optarg, outlvl=outlvl)

        # Set solver and options
        opt = get_solver(solver, optarg)

        # Fix variables and check degrees of freedom
        flags = (
            {}
        )  # Dictionary noting which variables were fixed and their previous state
        for k in self.keys():
            sb = self[k]
            for (v_name, ind), val in var_args.items():
                var = getattr(sb, v_name)
                if iscale.get_scaling_factor(var[ind]) is None:
                    _log.warning(
                        "While using the calculate_state method on {sb_name}, variable {v_name} "
                        "was provided as an argument in var_args, but it does not have a scaling "
                        "factor. This suggests that the calculate_scaling_factor method has not been "
                        "used or the variable was created on demand after the scaling factors were "
                        "calculated. It is recommended to touch all relevant variables (i.e. call "
                        "them or set an initial value) before using the calculate_scaling_factor "
                        "method.".format(v_name=v_name, sb_name=sb.name)
                    )
                if var[ind].is_fixed():
                    flags[(k, v_name, ind)] = True
                    if value(var[ind]) != val:
                        raise ConfigurationError(
                            "While using the calculate_state method on {sb_name}, {v_name} was "
                            "fixed to a value {val}, but it was already fixed to value {val_2}. "
                            "Unfix the variable before calling the calculate_state "
                            "method or update var_args."
                            "".format(
                                sb_name=sb.name,
                                v_name=var.name,
                                val=val,
                                val_2=value(var[ind]),
                            )
                        )
                else:
                    flags[(k, v_name, ind)] = False
                    var[ind].fix(val)

            if degrees_of_freedom(sb) != 0:
                raise RuntimeError(
                    "While using the calculate_state method on {sb_name}, the degrees "
                    "of freedom were {dof}, but 0 is required. Check var_args and ensure "
                    "the correct fixed variables are provided."
                    "".format(sb_name=sb.name, dof=degrees_of_freedom(sb))
                )

        # Solve
        with idaeslog.solver_log(solve_log, idaeslog.DEBUG) as slc:
            results = solve_indexed_blocks(opt, [self], tee=slc.tee)
            solve_log.info_high(
                "Calculate state: {}.".format(idaeslog.condition(results))
            )

        if not check_optimal_termination(results):
            _log.warning(
                "While using the calculate_state method on {sb_name}, the solver failed "
                "to converge to an optimal solution. This suggests that the user provided "
                "infeasible inputs, or that the model is poorly scaled, poorly initialized, "
                "or degenerate."
            )

        # unfix all variables fixed with var_args
        for (k, v_name, ind), previously_fixed in flags.items():
            if not previously_fixed:
                var = getattr(self[k], v_name)
                var[ind].unfix()

        # fix state variables if hold_state
        if hold_state:
            fix_state_vars(self)

        return results


@declare_process_block_class("LiBrStateBlock", block_class=_LiBrStateBlock)
class LiBrStateBlockData(StateBlockData):
    """A LiBr property package."""

    def build(self):
        """Callable method for Block construction."""
        super().build()

        self.scaling_factor = Suffix(direction=Suffix.EXPORT)

        # Add state variables
        self.flow_mass_phase_comp = pyo.Var(
            self.params.phase_list,
            self.params.component_list,
            initialize={("Liq", "H2O"): 0.65, ("Liq", "TDS"): 0.35},
            bounds=(0.0, None),
            domain=NonNegativeReals,
            units=pyunits.kg / pyunits.s,
            doc="Mass flow rate",
        )

        self.temperature = pyo.Var(
            initialize=298.15,
            bounds=(273.15, 1000),
            domain=NonNegativeReals,
            units=pyunits.K,
            doc="Temperature",
        )

        self.pressure = pyo.Var(
            initialize=101325,
            bounds=(1e3, 5e7),
            domain=NonNegativeReals,
            units=pyunits.Pa,
            doc="Pressure",
        )

    # -----------------------------------------------------------------------------
    # Property Methods
    def _mass_frac_phase_comp(self):
        self.mass_frac_phase_comp = pyo.Var(
            self.params.phase_list,
            self.params.component_list,
            initialize=0.1,
            bounds=(0.0, None),
            units=pyunits.dimensionless,
            doc="Mass fraction",
        )

        def rule_mass_frac_phase_comp(b, p, j):
            return b.mass_frac_phase_comp[p, j] == b.flow_mass_phase_comp[p, j] / sum(
                b.flow_mass_phase_comp[p, j] for j in b.params.component_list
            )

        self.eq_mass_frac_phase_comp = Constraint(
            self.params.phase_list,
            self.params.component_list,
            rule=rule_mass_frac_phase_comp,
        )

    # Density of LiBr solution (kg/m3)
    # Validity: 0 < t < 200 oC; 0.2 < X < 0.65
    # from ref [4]
    def _dens_mass_phase(self):
        self.dens_mass_phase = pyo.Var(
            self.params.phase_list,
            initialize=1e3,
            bounds=(1, 1e6),
            units=pyunits.kg / pyunits.m**3,
            doc="Mass density of LiBr in water",
        )

        def rule_dens_mass_phase(b, p):
            t = b.temperature - 273.15 * pyunits.K
            s = b.mass_frac_phase_comp[p, "TDS"]
            dens_mass = (
                b.params.dens_mass_param_B1
                + b.params.dens_mass_param_B2 * s
                + b.params.dens_mass_param_B3 * s**2
                - (b.params.dens_mass_param_B4 + b.params.dens_mass_param_B5 * s) * t
            )
            return b.dens_mass_phase[p] == dens_mass

        self.eq_dens_mass_phase = Constraint(
            self.params.phase_list, rule=rule_dens_mass_phase
        )

    def _dens_mass_solvent(self):
        self.dens_mass_solvent = pyo.Var(
            initialize=1e3,
            bounds=(1, 1e6),
            units=pyunits.kg * pyunits.m**-3,
            doc="Mass density of pure water",
        )

        # from ref [3]
        def rule_dens_mass_solvent(b):
            t = b.temperature - 273.15
            dens_mass_w = (
                b.params.dens_mass_param_A1
                + b.params.dens_mass_param_A2 * t
                + b.params.dens_mass_param_A3 * t**2
                + b.params.dens_mass_param_A4 * t**3
                + b.params.dens_mass_param_A5 * t**4
            )
            return b.dens_mass_solvent == dens_mass_w

        self.eq_dens_mass_solvent = Constraint(rule=rule_dens_mass_solvent)

    def _flow_vol_phase(self):
        self.flow_vol_phase = pyo.Var(
            self.params.phase_list,
            initialize=1,
            bounds=(0.0, None),
            units=pyunits.m**3 / pyunits.s,
            doc="Volumetric flow rate",
        )

        def rule_flow_vol_phase(b, p):
            return (
                b.flow_vol_phase[p]
                == sum(b.flow_mass_phase_comp[p, j] for j in b.params.component_list)
                / b.dens_mass_phase[p]
            )

        self.eq_flow_vol_phase = Constraint(
            self.params.phase_list, rule=rule_flow_vol_phase
        )

    def _flow_vol(self):
        def rule_flow_vol(b):
            return sum(b.flow_vol_phase[p] for p in b.params.phase_list)

        self.flow_vol = Expression(rule=rule_flow_vol)

    def _conc_mass_phase_comp(self):
        self.conc_mass_phase_comp = pyo.Var(
            self.params.phase_list,
            self.params.component_list,
            initialize=10,
            bounds=(0.0, 1e6),
            units=pyunits.kg * pyunits.m**-3,
            doc="Mass concentration",
        )

        def rule_conc_mass_phase_comp(b, p, j):
            return (
                b.conc_mass_phase_comp[p, j]
                == b.dens_mass_phase[p] * b.mass_frac_phase_comp[p, j]
            )

        self.eq_conc_mass_phase_comp = Constraint(
            self.params.phase_list,
            self.params.component_list,
            rule=rule_conc_mass_phase_comp,
        )

    def _flow_mol_phase_comp(self):
        self.flow_mol_phase_comp = pyo.Var(
            self.params.phase_list,
            self.params.component_list,
            initialize=100,
            bounds=(0.0, None),
            units=pyunits.mol / pyunits.s,
            doc="Molar flowrate",
        )

        def rule_flow_mol_phase_comp(b, p, j):
            return (
                b.flow_mol_phase_comp[p, j]
                == b.flow_mass_phase_comp[p, j] / b.params.mw_comp[j]
            )

        self.eq_flow_mol_phase_comp = Constraint(
            self.params.phase_list,
            self.params.component_list,
            rule=rule_flow_mol_phase_comp,
        )

    def _mole_frac_phase_comp(self):
        self.mole_frac_phase_comp = pyo.Var(
            self.params.phase_list,
            self.params.component_list,
            initialize=0.1,
            bounds=(0.0, None),
            units=pyunits.dimensionless,
            doc="Mole fraction",
        )

        def rule_mole_frac_phase_comp(b, p, j):
            return b.mole_frac_phase_comp[p, j] == b.flow_mol_phase_comp[p, j] / sum(
                b.flow_mol_phase_comp[p, j] for j in b.params.component_list
            )

        self.eq_mole_frac_phase_comp = Constraint(
            self.params.phase_list,
            self.params.component_list,
            rule=rule_mole_frac_phase_comp,
        )

    def _molality_phase_comp(self):
        self.molality_phase_comp = pyo.Var(
            self.params.phase_list,
            ["TDS"],
            initialize=1,
            bounds=(0.0, 1e6),
            units=pyunits.mole / pyunits.kg,
            doc="Molality",
        )

        def rule_molality_phase_comp(b, p, j):
            return (
                self.molality_phase_comp[p, j]
                == b.mass_frac_phase_comp[p, j]
                / (1 - b.mass_frac_phase_comp[p, j])
                / b.params.mw_comp[j]
            )

        self.eq_molality_phase_comp = Constraint(
            self.params.phase_list, ["TDS"], rule=rule_molality_phase_comp
        )

    # Absolute viscosity of LiBr solution (Pa*s)
    # Validity: 0 < t < 200 oC; 0.45 < X < 0.65
    # from ref [5]
    def _visc_d_phase(self):
        self.visc_d_phase = pyo.Var(
            self.params.phase_list,
            initialize=1e-3,
            bounds=(0.0, 1),
            units=pyunits.Pa * pyunits.s,
            doc="Viscosity",
        )

        def rule_visc_d_phase(b, p):
            t = b.temperature  # in K
            s = b.mass_frac_phase_comp[p, "TDS"]
            factor_visc = 100 * pyunits.dimensionless
            A1 = (
                b.params.visc_d_param_A
                + b.params.visc_d_param_B * factor_visc * s
                - b.params.visc_d_param_C * (factor_visc * s) ** 2
            )
            A2 = (
                b.params.visc_d_param_D
                - b.params.visc_d_param_E * factor_visc * s
                + b.params.visc_d_param_F * (factor_visc * s) ** 2
            )
            A3 = (
                b.params.visc_d_param_G
                - b.params.visc_d_param_H * factor_visc * s
                + b.params.visc_d_param_I * (factor_visc * s) ** 2
            )
            B = A1 + (A2 / t) + A3 * pyo.log(t)
            return b.visc_d_phase[p] == 0.001 * pyunits.dimensionless * pyo.exp(B)

        self.eq_visc_d_phase = Constraint(
            self.params.phase_list, rule=rule_visc_d_phase
        )

    # Enthalpy of LiBr solution (kJ/kg)
    # Assumptions: subsaturated, incompressible H(T,P)=H(T) with the same reference state as the steam tables
    # Validity: 0.4 < X < 0.7
    def _enth_mass_phase(self):
        self.enth_mass_phase = pyo.Var(
            self.params.phase_list,
            initialize=1e6,
            bounds=(1, 1e8),
            units=pyunits.J * pyunits.kg**-1,
            doc="Enthalpy",
        )

        def rule_enth_mass_phase(b, p):
            t = b.temperature  # in K
            X = b.mass_frac_phase_comp[p, "TDS"]
            t0 = 273.15 * pyunits.K
            cp = b.cp_mass_phase["Liq"]  # in J/(kg K)

            h_libr = cp * (t - t0)

            return b.enth_mass_phase[p] == h_libr

        self.eq_enth_mass_phase = Constraint(
            self.params.phase_list, rule=rule_enth_mass_phase
        )

    def _enth_flow(self):
        # Enthalpy flow expression for get_enthalpy_flow_terms method

        def rule_enth_flow(b):  # enthalpy flow [J/s]
            return (
                sum(b.flow_mass_phase_comp["Liq", j] for j in b.params.component_list)
                * b.enth_mass_phase["Liq"]
            )

        self.enth_flow = Expression(rule=rule_enth_flow)

    # Water vapor saturation temperature (K) at given pressure
    # Pressure P(start = 101325.0,min=0.01)
    # reverse Antoinne's equation
    def _temperature_sat_solvent(self):
        self.temperature_sat_solvent = pyo.Var(
            initialize=298.15,
            bounds=(1, 1e3),
            units=pyunits.K,
            doc="Vapor temperature of water",
        )

        def rule_temperature_sat_solvent(b):
            factor_pa = 1000000 * pyunits.Pa
            p = b.pressure
            tsat_w = (
                b.params.temperature_sat_solvent_param_A1
                - b.params.temperature_sat_solvent_param_A2
                / (pyo.log(p / factor_pa) - b.params.temperature_sat_solvent_param_A3)
            )

            return b.temperature_sat_solvent == tsat_w

        self.eq_temperature_sat_solvent = Constraint(rule=rule_temperature_sat_solvent)

    # Saturation temp (boiling temp) in K of a LiBr solution given pressure and mass fraction of LiBr
    # from ref [4]
    def _temperature_sat(self):
        self.temperature_sat = pyo.Var(
            initialize=298.15, bounds=(1, 1e3), units=pyunits.K, doc="Vapor temperature"
        )

        def rule_temperature_sat(b):
            t = b.temperature
            tref = b.temperature_sat_solvent  # water vapor saturation temperature
            s = b.mass_frac_phase_comp["Liq", "TDS"]
            s1 = (
                b.params.temperature_sat_param_a["a1"]
                + b.params.temperature_sat_param_a["a2"] * s
                + b.params.temperature_sat_param_a["a3"] * s**2
                + b.params.temperature_sat_param_a["a4"] * s**3
                + b.params.temperature_sat_param_a["a5"] * s**4
                + b.params.temperature_sat_param_a["a6"] * s**5
                + b.params.temperature_sat_param_a["a7"] * s**6
                + b.params.temperature_sat_param_a["a8"] * s**7
                + b.params.temperature_sat_param_a["a9"] * s**8
                + b.params.temperature_sat_param_a["a10"] * s**9
                + b.params.temperature_sat_param_a["a11"] * s**10
            )
            s2 = (
                b.params.temperature_sat_param_b["b1"]
                + b.params.temperature_sat_param_b["b2"] * s
                + b.params.temperature_sat_param_b["b3"] * s**2
                + b.params.temperature_sat_param_b["b4"] * s**3
                + b.params.temperature_sat_param_b["b5"] * s**4
                + b.params.temperature_sat_param_b["b6"] * s**5
                + b.params.temperature_sat_param_b["b7"] * s**6
                + b.params.temperature_sat_param_b["b8"] * s**7
                + b.params.temperature_sat_param_b["b9"] * s**8
                + b.params.temperature_sat_param_b["b10"] * s**9
                + b.params.temperature_sat_param_b["b11"] * s**10
            )
            tsat = (
                s1 * pyunits.K + (tref - 273.15 * pyunits.K) * s2 + 273.15 * pyunits.K
            )

            return b.temperature_sat == tsat

        self.eq_temperature_sat = Constraint(rule=rule_temperature_sat)

    # Saturation pressure of refrigerant
    # from ref [5]
    def _pressure_sat(self):
        self.pressure_sat = pyo.Var(
            initialize=1e6, bounds=(1, 1e10), units=pyunits.Pa, doc="Vapor pressure"
        )

        def rule_pressure_sat(b):
            tsat = b.temperature_sat  # units in K
            scaling = 1000 * pyunits.dimensionless

            return (
                b.pressure_sat
                == (
                    10
                    ** (
                        (
                            b.params.pressure_sat_param_A1
                            + b.params.pressure_sat_param_A2 / (tsat)
                            + b.params.pressure_sat_param_A3 / (tsat) ** 2
                        )
                        / scaling
                    )
                )
                * pyunits.Pa
            )

        self.eq_pressure_sat = Constraint(rule=rule_pressure_sat)

    # Heat capacity of LiBr solution (J/(kg K)
    # from ref [5]
    def _cp_mass_phase(self):
        self.cp_mass_phase = pyo.Var(
            self.params.phase_list,
            initialize=4e3,
            bounds=(0.0, 1e8),
            units=pyunits.J / pyunits.kg / pyunits.K,
            doc="Specific heat capacity",
        )

        def rule_cp_mass_phase(b, p):
            s = b.mass_frac_phase_comp[p, "TDS"]
            factor = 100
            cp = (
                b.params.cp_phase_param_A1 * (s * factor) ** 2
                - b.params.cp_phase_param_A2 * s * factor
                + b.params.cp_phase_param_A3
            )

            return b.cp_mass_phase[p] == cp

        self.eq_cp_mass_phase = Constraint(
            self.params.phase_list, rule=rule_cp_mass_phase
        )

    # Thermal conductivity(W/(m K)) of LiBr solution at T(K) and X(g LiBr/g soln)
    # from ref [5]
    def _therm_cond_phase(self):
        self.therm_cond_phase = pyo.Var(
            self.params.phase_list,
            initialize=0.4,
            bounds=(0.0, 1),
            units=pyunits.W / pyunits.m / pyunits.K,
            doc="Thermal conductivity",
        )

        def rule_therm_cond_phase(b, p):
            t = b.temperature
            s = b.mass_frac_phase_comp[p, "TDS"]
            K1 = (
                b.params.therm_cond_phase_param_1 * s
                + b.params.therm_cond_phase_param_2
            )
            K3 = (
                b.params.therm_cond_phase_param_5 * s
                + b.params.therm_cond_phase_param_6
            )
            D2 = (K3 - K1) * (313 * pyunits.K - t) / (20 * pyunits.K)

            K = K1 + D2

            # If T >= 313, use this expression:
            # K2 = (
            #     b.params.therm_cond_phase_param_3 * s
            #     + b.params.therm_cond_phase_param_4
            # )
            # K = K1 + D1
            # D1 = (
            #     (K2 - K1) * (t - 313 * pyunits.K) / (20 * pyunits.K)
            # )
            return b.therm_cond_phase[p] == K

        self.eq_therm_cond_phase = Constraint(
            self.params.phase_list, rule=rule_therm_cond_phase
        )

    # -----------------------------------------------------------------------------
    # General Methods

    # NOTE: For scaling in the control volume to work properly, these
    # methods must return a pyomo Var or Expression

    def get_material_flow_terms(self, p, j):
        """Create material flow terms for control volume."""
        return self.flow_mass_phase_comp[p, j]

    def get_enthalpy_flow_terms(self, p):
        """Create enthalpy flow terms."""
        return self.enth_flow

    def default_material_balance_type(self):
        return MaterialBalanceType.componentTotal

    def default_energy_balance_type(self):
        return EnergyBalanceType.enthalpyTotal

    def get_material_flow_basis(self):
        return MaterialFlowBasis.mass

    def define_state_vars(self):
        """Define state vars."""
        return {
            "flow_mass_phase_comp": self.flow_mass_phase_comp,
            "temperature": self.temperature,
            "pressure": self.pressure,
        }

    # -----------------------------------------------------------------------------
    # Scaling methods
    def calculate_scaling_factors(self):
        super().calculate_scaling_factors()

        # setting scaling factors for variables

        # default scaling factors have already been set with
        # idaes.core.property_base.calculate_scaling_factors()
        # for the following variables: flow_mass_phase_comp, pressure,
        # temperature, dens_mass_phase, visc_d_phase, osm_coeff, and enth_mass_phase

        # these variables should have user input
        if iscale.get_scaling_factor(self.flow_mass_phase_comp["Liq", "H2O"]) is None:
            sf = iscale.get_scaling_factor(
                self.flow_mass_phase_comp["Liq", "H2O"], default=1e0, warning=True
            )
            iscale.set_scaling_factor(self.flow_mass_phase_comp["Liq", "H2O"], sf)

        if iscale.get_scaling_factor(self.flow_mass_phase_comp["Liq", "TDS"]) is None:
            sf = iscale.get_scaling_factor(
                self.flow_mass_phase_comp["Liq", "TDS"], default=1e0, warning=True
            )
            iscale.set_scaling_factor(self.flow_mass_phase_comp["Liq", "TDS"], sf)

        # scaling factors for parameters
        for j, v in self.params.mw_comp.items():
            if iscale.get_scaling_factor(v) is None:
                iscale.set_scaling_factor(self.params.mw_comp, 1e2)

        # these variables do not typically require user input,
        # will not override if the user does provide the scaling factor
        if self.is_property_constructed("mass_frac_phase_comp"):
            for j in self.params.component_list:
                if (
                    iscale.get_scaling_factor(self.mass_frac_phase_comp["Liq", j])
                    is None
                ):
                    if j == "TDS":
                        sf = iscale.get_scaling_factor(
                            self.flow_mass_phase_comp["Liq", j]
                        ) / iscale.get_scaling_factor(
                            self.flow_mass_phase_comp["Liq", "H2O"]
                        )
                        iscale.set_scaling_factor(
                            self.mass_frac_phase_comp["Liq", j], sf
                        )
                    elif j == "H2O":
                        iscale.set_scaling_factor(
                            self.mass_frac_phase_comp["Liq", j], 1
                        )

        if self.is_property_constructed("flow_vol_phase"):
            sf = iscale.get_scaling_factor(
                self.flow_mass_phase_comp["Liq", "H2O"]
            ) / iscale.get_scaling_factor(self.dens_mass_phase["Liq"])
            iscale.set_scaling_factor(self.flow_vol_phase, sf)

        if self.is_property_constructed("flow_vol"):
            sf = iscale.get_scaling_factor(self.flow_vol_phase)
            iscale.set_scaling_factor(self.flow_vol, sf)

        if self.is_property_constructed("conc_mass_phase_comp"):
            for j in self.params.component_list:
                sf_dens = iscale.get_scaling_factor(self.dens_mass_phase["Liq"])
                if (
                    iscale.get_scaling_factor(self.conc_mass_phase_comp["Liq", j])
                    is None
                ):
                    if j == "H2O":
                        # solvents typically have a mass fraction between 0.5-1
                        iscale.set_scaling_factor(
                            self.conc_mass_phase_comp["Liq", j], sf_dens
                        )
                    elif j == "TDS":
                        iscale.set_scaling_factor(
                            self.conc_mass_phase_comp["Liq", j],
                            sf_dens
                            * iscale.get_scaling_factor(
                                self.mass_frac_phase_comp["Liq", j]
                            ),
                        )

        if self.is_property_constructed("flow_mol_phase_comp"):
            for j in self.params.component_list:
                if (
                    iscale.get_scaling_factor(self.flow_mol_phase_comp["Liq", j])
                    is None
                ):
                    sf = iscale.get_scaling_factor(self.flow_mass_phase_comp["Liq", j])
                    sf /= iscale.get_scaling_factor(self.params.mw_comp[j])
                    iscale.set_scaling_factor(self.flow_mol_phase_comp["Liq", j], sf)

        if self.is_property_constructed("mole_frac_phase_comp"):
            for j in self.params.component_list:
                if (
                    iscale.get_scaling_factor(self.mole_frac_phase_comp["Liq", j])
                    is None
                ):
                    if j == "TDS":
                        sf = iscale.get_scaling_factor(
                            self.flow_mol_phase_comp["Liq", j]
                        ) / iscale.get_scaling_factor(
                            self.flow_mol_phase_comp["Liq", "H2O"]
                        )
                        iscale.set_scaling_factor(
                            self.mole_frac_phase_comp["Liq", j], sf
                        )
                    elif j == "H2O":
                        iscale.set_scaling_factor(
                            self.mole_frac_phase_comp["Liq", j], 1
                        )

        if self.is_property_constructed("molality_phase_comp"):
            for j in self.params.component_list:
                if isinstance(getattr(self.params, j), Solute):
                    if (
                        iscale.get_scaling_factor(self.molality_phase_comp["Liq", j])
                        is None
                    ):
                        sf = iscale.get_scaling_factor(
                            self.mass_frac_phase_comp["Liq", j]
                        )
                        sf /= iscale.get_scaling_factor(self.params.mw_comp[j])
                        iscale.set_scaling_factor(
                            self.molality_phase_comp["Liq", j], sf
                        )

        if self.is_property_constructed("enth_flow"):
            iscale.set_scaling_factor(
                self.enth_flow,
                iscale.get_scaling_factor(self.flow_mass_phase_comp["Liq", "H2O"])
                * iscale.get_scaling_factor(self.enth_mass_phase["Liq"]),
            )

        # transforming constraints
        # property relationships with no index, simple constraint
        v_str_lst_simple = [
            "dens_mass_solvent",
            "pressure_sat",
        ]
        for v_str in v_str_lst_simple:
            if self.is_property_constructed(v_str):
                v = getattr(self, v_str)
                sf = iscale.get_scaling_factor(v, default=1, warning=True)
                c = getattr(self, "eq_" + v_str)
                iscale.constraint_scaling_transform(c, sf)

        # transforming constraints
        transform_property_constraints(self)
