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
# Author: Nazia Aslam from the University of Connecticut
#################################################################################
"""
References:
[1] Robinson, R. A., & Stokes, R. H. Electrolyte Solutions. Academic Press, New York, 1959. xv + 559 pp.

[2] Stuber, M. D., Sullivan, C., Kirk, S. A., Farrand, J. A., Schillaci, P. V., Fojtasek, B. D., & Mandell, A. H. (2015). 
Pilot demonstration of concentrated solar-powered desalination of subsurface agricultural drainage water and 
other brackish groundwater sources. Desalination, 355, 186-196. https://doi.org/10.1016/j.desal.2014.10.037

This is a closed-loop 3MED-AHP model configuration. A break-point is placed between the absorber tube outlet and the pump inlet to 
account for the necessary concentration increase following the absorber.
The model uses experimental conditions from [2] and validates well at the listed water recoveries below for single electrolyte systems. 

The following changes need to be made to run specific conditions at 70% water recovery:
1. Ideal
def set_scaling(m):
            if "temperature" in var.name:
                iscale.set_scaling_factor(var, 1e-2) 

2. r-eNRTL(constant)
def set_scaling(m):
            if "temperature" in var.name:
                iscale.set_scaling_factor(var, 1e-2) 

3. r-eNRTL(stepwise)
def set_scaling(m):
            if "temperature" in var.name:
                iscale.set_scaling_factor(var, 1e-1) 

4. IDAES e-NRTL 
def set_scaling(m):
            if "temperature" in var.name:
                iscale.set_scaling_factor(var, 1e-3) 

5. multi r-eNRTL(constant)
def set_scaling(m):
            if "temperature" in var.name:
                iscale.set_scaling_factor(var, 1e2)
"""
import logging

# Import pyomo components
import pyomo.environ as pyo
from pyomo.environ import (
    ConcreteModel,
    TransformationFactory,
    Block,
    Constraint,
    Expression,
    Objective,
    minimize,
    Param,
    value,
    Set,
    RangeSet,
    log,
    exp,
    Var,
)
from pyomo.network import Arc
from pyomo.environ import units as pyunits

# Import IDAES components
import idaes.core.util.scaling as iscale
import idaes.logger as idaeslog
from idaes.core import FlowsheetBlock
from idaes.models.properties.modular_properties.base.generic_property import (
    GenericParameterBlock,
)
from idaes.models.unit_models import Feed
from idaes.core.solvers.get_solver import get_solver
from idaes.core.util.model_statistics import degrees_of_freedom
from idaes.core.util.initialization import propagate_state
from idaes.core import MaterialBalanceType
from idaes.models.unit_models import PressureChanger
from idaes.models.unit_models.pressure_changer import ThermodynamicAssumption
from idaes.models.unit_models import Mixer, MomentumMixingType, Separator
from idaes.models.unit_models.separator import SplittingType
from idaes.models.unit_models.heat_exchanger import (
    HeatExchanger,
    HeatExchangerFlowPattern,
)
from idaes.models.unit_models.translator import Translator

# Import WaterTAP components
from watertap.unit_models.mvc.components import Evaporator, Condenser
from watertap.unit_models.mvc.components.lmtd_chen_callback import (
    delta_temperature_chen_callback,
)

# Import property packages
import watertap.property_models.seawater_prop_pack as props_sw
import watertap.property_models.water_prop_pack as props_w
import LiBr_prop_pack as props_libr

# Import configuration dictionaries
import LiBr_enrtl_config_FpcTPupt

logging.basicConfig(level=logging.INFO)
logging.getLogger("pyomo.repn.plugins.nl_writer").setLevel(logging.ERROR)


# solve_nonideal gives the option to solve an ideal and nonideal case for the MED loop of the system
# solve_nonideal_AHP gives the option to solve an ideal and nonideal case for the AHP loop of the system
# If solve_nonideal is set to true, eNRTL is used to calculate the activity coefficients of solvent and solutes;
# when set to False, the model is solved assuming an ideal system with an activity coefficient of 1 for the solvent.
solve_nonideal = True  # 3MED loop
solve_nonideal_AHP = True  # AHP loop

# run_multi toggles between single-component and multi-component electrolyte systems. To use the multi-refined eNRTL, run_multi=True. To use the single refined eNRTL, run_multi=False.
# NOTE: Make sure the config files, LiBr_enrtl_config_FpcTPupt and enrtl_config_FpcTP or renrtl_multi_config are using the same refined eNRTL model
run_multi = True

if run_multi:
    import renrtl_multi_config  # multi electrolytes
else:
    import enrtl_config_FpcTP  # single electrolyte


def populate_enrtl_state_vars_single(blk, base="FpcTP"):  # for MED loop
    blk.temperature = 298.15
    blk.pressure = 101325

    if base == "FpcTP":
        feed_flow_mass = 0.25  # kg/s
        feed_mass_frac_comp = {"Na+": 0.009127, "Cl-": 0.01407}
        feed_mass_frac_comp["H2O"] = 1 - sum(x for x in feed_mass_frac_comp.values())
        mw_comp = {"H2O": 18.015e-3, "Na+": 22.990e-3, "Cl-": 35.453e-3}

        for j in feed_mass_frac_comp:
            blk.flow_mol_phase_comp["Liq", j] = (
                feed_flow_mass * feed_mass_frac_comp[j] / mw_comp[j]
            )
            if j == "H2O":
                blk.flow_mol_phase_comp["Liq", j] /= 2


def populate_enrtl_state_vars_multi(blk, base="FpcTP"):  # for MED loop
    """Initialize state variables"""
    blk.temperature = 27 + 273.15
    blk.pressure = 101325

    if base == "FpcTP":
        feed_flow_mass = 0.25  # kg/s
        feed_mass_frac_comp = {"Na+": 0.003022, "Cl-": 0.004601, "SO4_2-": 0.01263}
        feed_mass_frac_comp["H2O"] = 1 - sum(x for x in feed_mass_frac_comp.values())
        mw_comp = {
            "H2O": 18.015e-3,
            "Na+": 22.990e-3,
            "Cl-": 35.453e-3,
            "SO4_2-": 96.064e-3,
        }

        for j in feed_mass_frac_comp:
            blk.flow_mol_phase_comp["Liq", j] = (
                feed_flow_mass * feed_mass_frac_comp[j] / mw_comp[j]
            )
            if j == "H2O":
                blk.flow_mol_phase_comp["Liq", j] /= 2


def populate_enrtl_state_vars_gen(blk, base="FpcTP"):  # for AHP loop
    blk.temperature = 180 + 273.15
    blk.pressure = 12000

    if base == "FpcTP":
        feed_flow_mass = 1  # kg/s
        feed_mass_frac_comp = {"Li+": 0.0043945, "Br-": 0.0506055}
        feed_mass_frac_comp["H2O"] = 1 - sum(x for x in feed_mass_frac_comp.values())
        mw_comp = {"H2O": 18.015e-3, "Li+": 6.941e-3, "Br-": 79.904e-3}

        for j in feed_mass_frac_comp:
            blk.flow_mol_phase_comp["Liq", j] = (
                feed_flow_mass * feed_mass_frac_comp[j] / mw_comp[j]
            )
            if j == "H2O":
                blk.flow_mol_phase_comp["Liq", j] /= 2


def create_model():
    m = ConcreteModel("Three-effect Distillation with Absorption Heat Pump Loop")
    m.fs = FlowsheetBlock(dynamic=False)

    # Add property packages for water, seawater, and lithium bromide
    m.fs.properties_vapor = props_w.WaterParameterBlock()
    m.fs.properties_feed_sw = props_sw.SeawaterParameterBlock()
    m.fs.properties_feed = props_libr.LiBrParameterBlock()

    m.fs.feed_sw = Feed(property_package=m.fs.properties_feed_sw)  # Seawater
    m.fs.feed = Feed(property_package=m.fs.properties_feed)  # LiBr

    # Declare unit models
    m.fs.generator = Evaporator(
        property_package_feed=m.fs.properties_feed,
        property_package_vapor=m.fs.properties_vapor,
    )

    m.fs.economizer = HeatExchanger(
        delta_temperature_callback=delta_temperature_chen_callback,
        hot_side_name="tube",
        cold_side_name="shell",
        tube={"property_package": m.fs.properties_feed},
        shell={"property_package": m.fs.properties_feed},
        flow_pattern=HeatExchangerFlowPattern.crossflow,
    )

    m.fs.expansion_valve = PressureChanger(
        property_package=m.fs.properties_feed,
        material_balance_type=MaterialBalanceType.componentTotal,
        thermodynamic_assumption=ThermodynamicAssumption.pump,
    )

    m.fs.mixer = Mixer(
        property_package=m.fs.properties_feed,
        num_inlets=2,
        momentum_mixing_type=MomentumMixingType.minimize,
    )

    m.fs.absorber = HeatExchanger(
        delta_temperature_callback=delta_temperature_chen_callback,
        hot_side_name="tube",
        cold_side_name="shell",
        tube={
            "property_package": m.fs.properties_feed
        },  # LiBr absorbent enters tube to preheat SW
        shell={"property_package": m.fs.properties_feed_sw},  # SW enters the shell
        flow_pattern=HeatExchangerFlowPattern.crossflow,
    )

    m.fs.pump = PressureChanger(
        property_package=m.fs.properties_feed,
        material_balance_type=MaterialBalanceType.componentTotal,
        thermodynamic_assumption=ThermodynamicAssumption.pump,
    )

    # Note: the evaporator unit is a customized unit that includes a complete condenser
    m.fs.num_evaporators = 3
    m.fs.set_evaporators = RangeSet(m.fs.num_evaporators)
    m.fs.set_condensers = RangeSet(m.fs.num_evaporators + 1)

    m.fs.evaporator = Evaporator(
        m.fs.set_evaporators,
        property_package_feed=m.fs.properties_feed_sw,
        property_package_vapor=m.fs.properties_vapor,
    )

    m.fs.condenser = Condenser(
        m.fs.set_condensers, property_package=m.fs.properties_vapor
    )

    m.fs.separator = Separator(
        property_package=m.fs.properties_vapor,
        outlet_list=["outlet_1", "outlet_2"],
        split_basis=SplittingType.totalFlow,
    )

    # The water property package from the outlet of separator cannot mix with LiBr when entering the AHP loop,thus a translator block is added to link the two streams
    m.fs.tblock = Translator(
        inlet_property_package=m.fs.properties_vapor,
        outlet_property_package=m.fs.properties_feed,
    )

    @m.fs.tblock.Constraint()
    def eq_flow_mass_comp_H2O(b):
        return b.properties_out[0].flow_mass_phase_comp["Liq", "H2O"] == (
            b.properties_in[0].flow_mass_phase_comp["Liq", "H2O"]
            + b.properties_in[0].flow_mass_phase_comp["Vap", "H2O"]
        )

    @m.fs.tblock.Constraint()
    def eq_temperature(b):
        return b.properties_in[0].temperature == b.properties_out[0].temperature

    @m.fs.tblock.Constraint()
    def eq_pressure(b):
        return b.properties_out[0].pressure == b.properties_in[0].pressure

    # Add variable to calculate molal concentration of solute.
    # The upper bound is included to ensure that the molality of the electrolyte solution is within saturation and the concentration limits of eNRTL tau and alpha parameters.

    # The eNRTL method is applied in the MED and the AHP loop
    # MED Loop
    m.fs.molal_conc_solute = pyo.Var(
        m.fs.set_evaporators,
        initialize=2,
        bounds=(1e-3, 6),
        units=pyunits.mol / pyunits.kg,
        doc="Molal concentration of solute",
    )

    @m.fs.Constraint(
        m.fs.set_evaporators,
        doc="Molal concentration of solute in solvent in mol of TDS/kg of H2O",
    )
    def rule_molal_conc_solute(b, e):
        return m.fs.molal_conc_solute[e] == (
            (
                b.evaporator[e].properties_brine[0].flow_mass_phase_comp["Liq", "TDS"]
                / b.properties_feed.mw_comp["TDS"]  # to convert it to mol/s
            )
            / b.evaporator[e].properties_brine[0].flow_mass_phase_comp["Liq", "H2O"]
        )

    # AHP Loop
    m.fs.molal_conc_solute_gen = pyo.Var(
        initialize=2,
        bounds=(1e-3, 50),
        units=pyunits.mol / pyunits.kg,
        doc="Molal concentration of solute",
    )

    @m.fs.Constraint(
        doc="Molal concentration of solute in solvent in mol of TDS/kg of H2O"
    )
    def rule_molal_conc_solute_gen(b):
        return b.molal_conc_solute_gen == (
            (
                b.generator.properties_brine[0].flow_mass_phase_comp["Liq", "TDS"]
                / b.properties_feed.mw_comp["TDS"]  # to convert it to mol/s
            )
            / b.generator.properties_brine[0].flow_mass_phase_comp["Liq", "H2O"]
        )

    # Add eNRTL method to calculate the activity coefficients for the electrolyte solution.
    # Note that since water is the only solvent participating in the vapor-liquid equilibrium,the activity coefficient is of water

    # MED Loop
    if solve_nonideal:
        # Add activity coefficient as a global variable in each evaporator.
        m.fs.act_coeff = pyo.Var(
            m.fs.set_evaporators,
            initialize=1,
            units=pyunits.dimensionless,
            bounds=(0, 20),
        )

        # Declare a block to include the generic properties needed by eNRTL as a state block.
        m.fs.enrtl_state = Block(m.fs.set_evaporators)

        if run_multi:
            # Declare a Generic Parameter Block that calls a configuration file that includes eNRTL as the equation of state method.
            # Use the multi-component configuration
            m.fs.prop_enrtl = GenericParameterBlock(**renrtl_multi_config.configuration)

            # Declare a set for the ions in the electrolyte solution and the stoichiometric coefficient of the ions in the solute molecule
            # Calculate the total mass of the solute molecule and the mass ratio of each ion in the solute molecule
            m.fs.set_ions_multi = Set(initialize=["Na+", "Cl-", "SO4_2-"])
            m.fs.ion_coeff_multi = {"Na+": 3, "Cl-": 1, "SO4_2-": 1}

            m.fs.set_ions_nacl = Set(initialize=["Na+", "Cl-"])
            m.fs.ion_coeff_nacl = {"Na+": 1, "Cl-": 1}

            m.fs.set_ions_na2so4 = Set(initialize=["Na+", "SO4_2-"])
            m.fs.ion_coeff_na2so4 = {"Na+": 2, "SO4_2-": 1}

            for e in m.fs.set_evaporators:
                m.fs.enrtl_state[e].properties = m.fs.prop_enrtl.build_state_block([0])
                add_enrtl_method_multi(m, n_evap=e)

        else:
            # Declare a Generic Parameter Block that calls a configuration file that includes eNRTL as the equation of state method.
            # Use the single-component configuration
            m.fs.prop_enrtl = GenericParameterBlock(**enrtl_config_FpcTP.configuration)

            # Declare a set for the ions in the electrolyte solution and the stoichiometric coefficient of the ions in the solute molecule
            # Calculate the total mass of the solute molecule and the mass ratio of each ion in the solute molecule

            m.fs.set_ions_single = Set(initialize=["Na+", "Cl-"])
            m.fs.ion_coeff_single = {"Na+": 1, "Cl-": 1}

            for e in m.fs.set_evaporators:
                m.fs.enrtl_state[e].properties = m.fs.prop_enrtl.build_state_block([0])
                add_enrtl_method_single(m, n_evap=e)

        # Save the calculated activity coefficient in the global activity coefficient variable.
        @m.fs.Constraint(
            m.fs.set_evaporators, doc="eNRTL activity coefficient for water"
        )
        def eNRTL_activity_coefficient(b, e):
            return (
                b.act_coeff[e]
                == m.fs.enrtl_state[e].properties[0].act_coeff_phase_comp["Liq", "H2O"]
            )

    else:
        # Add the activity coefficient as a parameter with a value of 1
        m.fs.act_coeff = pyo.Param(
            m.fs.set_evaporators, initialize=1, units=pyunits.dimensionless
        )

    # Deactivate equilibrium equation from evaporators.
    # Note that when deactivated, one DOF appears for each evaporator.
    for e in m.fs.set_evaporators:
        m.fs.evaporator[e].eq_brine_pressure.deactivate()

    # Add vapor-liquid equilibrium equation.
    @m.fs.Constraint(m.fs.set_evaporators, doc="Vapor-liquid equilibrium equation")
    def _eq_phase_equilibrium(b, e):
        return (
            1  # mole fraction of water in vapor phase
            * b.evaporator[e].properties_brine[0].pressure
        ) == (
            m.fs.act_coeff[e]
            * b.evaporator[e].properties_brine[0].mole_frac_phase_comp["Liq", "H2O"]
            * b.evaporator[e].properties_vapor[0].pressure_sat
        )

    # AHP Loop
    if solve_nonideal_AHP:
        # Add activity coefficient as a global variable in each evaporator.
        m.fs.act_coeff_gen = pyo.Var(
            initialize=1, units=pyunits.dimensionless, bounds=(1e-5, 100)
        )

        # Declare a block to include the generic properties needed by eNRTL as a state block.
        m.fs.enrtl_state_gen = Block()

        # Declare a Generic Parameter Block that calls the LiBr configuration file that includes eNRTL as the equation of state method.
        m.fs.prop_enrtl_gen = GenericParameterBlock(
            **LiBr_enrtl_config_FpcTPupt.configuration
        )

        m.fs.set_ions_AHP = Set(initialize=["Li+", "Br-"])
        m.fs.ion_coeff_AHP = {"Li+": 1, "Br-": 1}

        m.fs.enrtl_state_gen.properties = m.fs.prop_enrtl_gen.build_state_block([0])

        add_enrtl_method_AHP(m)

        # Save the calculated activity coefficient in the global activity coefficient variable.
        @m.fs.Constraint(doc="eNRTL activity coefficient for water")
        def eNRTL_activity_coefficient_AHP(b):
            return (
                b.act_coeff_gen
                == m.fs.enrtl_state_gen.properties[0].act_coeff_phase_comp["Liq", "H2O"]
            )

    else:
        # Add the activity coefficient as a parameter with a value of 1
        m.fs.act_coeff_gen = pyo.Param(initialize=1, units=pyunits.dimensionless)

    # Deactivate equilibrium equation from generator
    # Note that when deactivated, one DOF appears
    m.fs.generator.eq_brine_pressure.deactivate()

    # Add vapor-liquid equilibrium equation.
    @m.fs.Constraint(doc="Vapor-liquid equilibrium equation")
    def _eq_phase_equilibrium_gen(b):
        return (
            1  # mole fraction of water in vapor phase
            * b.generator.properties_brine[0].pressure
        ) == (
            m.fs.act_coeff_gen
            * b.generator.properties_brine[0].mole_frac_phase_comp["Liq", "H2O"]
            * b.generator.properties_vapor[0].pressure_sat
        )

    create_arcs(m)
    TransformationFactory("network.expand_arcs").apply_to(m)

    return m


def create_arcs(m):
    # Create arcs to connect units in the flowsheet

    m.fs.pump_to_economizer = Arc(
        source=m.fs.pump.outlet,
        destination=m.fs.economizer.shell_inlet,
        doc="Connect pump outlet to economizer shell inlet",
    )

    m.fs.economizer_to_generator = Arc(
        source=m.fs.economizer.shell_outlet,
        destination=m.fs.generator.inlet_feed,
        doc="Connect economizer shell outlet to steam generator inlet",
    )

    m.fs.generator_to_economizer = Arc(
        source=m.fs.generator.outlet_brine,
        destination=m.fs.economizer.tube_inlet,
        doc="Connect steam generator brine outlet to economizer tube inlet",
    )

    m.fs.economizer_to_valve = Arc(
        source=m.fs.economizer.tube_outlet,
        destination=m.fs.expansion_valve.inlet,
        doc="Connect economizer tube outlet to expansion valve inlet",
    )

    m.fs.valve_to_mixer = Arc(
        source=m.fs.expansion_valve.outlet,
        destination=m.fs.mixer.inlet_2,
        doc="Connect expansion valve outlet to mixer inlet 2",
    )

    m.fs.mixer_to_absorber = Arc(
        source=m.fs.mixer.outlet,
        destination=m.fs.absorber.tube_inlet,
        doc="Connect mixer outlet to absorber tube inlet",
    )

    m.fs.feed_to_absorber = Arc(
        source=m.fs.feed_sw.outlet,
        destination=m.fs.absorber.shell_inlet,
        doc="Connect seawater feed to absorber shell inlet",
    )

    # There is a break-point between the absorber tube outlet and the pump inlet

    m.fs.generator_to_condenser = Arc(
        source=m.fs.generator.outlet_vapor,
        destination=m.fs.condenser[1].inlet,
        doc="Connect steam generator vapor outlet to condenser 1 inlet",
    )

    m.fs.absorber_to_evaporator_feed = Arc(
        source=m.fs.absorber.shell_outlet,
        destination=m.fs.evaporator[1].inlet_feed,
        doc="Connect absorber shell outlet to evaporator 1 inlet",
    )

    m.fs.evap1brine_to_evap2feed = Arc(
        source=m.fs.evaporator[1].outlet_brine,
        destination=m.fs.evaporator[2].inlet_feed,
        doc="Connect evaporator 1 brine outlet to evaporator 2 inlet",
    )
    m.fs.evap1vapor_to_cond2 = Arc(
        source=m.fs.evaporator[1].outlet_vapor,
        destination=m.fs.condenser[2].inlet,
        doc="Connect evaporator 1 vapor outlet to condenser 2 inlet",
    )

    m.fs.evap2vapor_to_cond3 = Arc(
        source=m.fs.evaporator[2].outlet_vapor,
        destination=m.fs.condenser[3].inlet,
        doc="Connect evaporator 2 vapor outlet to condenser 3 inlet",
    )

    m.fs.evap2brine_to_evap3feed = Arc(
        source=m.fs.evaporator[2].outlet_brine,
        destination=m.fs.evaporator[3].inlet_feed,
        doc="Connect evaporator 2 brine outlet to evaporator 3 inlet",
    )

    m.fs.evap3vapor_to_separator = Arc(
        source=m.fs.evaporator[3].outlet_vapor,
        destination=m.fs.separator.inlet,
        doc="Connect evaporator 3 vapor outlet to separator inlet",
    )

    m.fs.separator_to_condenser = Arc(
        source=m.fs.separator.outlet_2,
        destination=m.fs.condenser[4].inlet,
        doc="Connect separator outlet 2 to condenser 4 inlet",
    )

    m.fs.separator_to_tblock = Arc(
        source=m.fs.separator.outlet_1,
        destination=m.fs.tblock.inlet,
        doc="Connect separator outlet 1 to translator block inlet",
    )

    m.fs.tblock_to_mixer = Arc(
        source=m.fs.tblock.outlet,
        destination=m.fs.mixer.inlet_1,
        doc="Connect translator block outlet to mixer inlet 1",
    )


# MED Loop
def add_enrtl_method_single(m, n_evap=None):
    sb_enrtl = m.fs.enrtl_state[n_evap].properties[0]  # renaming the block

    # Populate eNRTL state block
    populate_enrtl_state_vars_single(sb_enrtl, base="FpcTP")

    # Calculate the total mass of the solute molecule and the mass ratio of each ion in the solute molecule.
    m.fs.enrtl_state[n_evap].mol_mass_ion_molecule = sum(
        m.fs.ion_coeff_single[j] * sb_enrtl.mw_comp[j] for j in m.fs.set_ions_single
    )
    m.fs.enrtl_state[n_evap].mass_ratio_ion = {
        "Na+": sb_enrtl.mw_comp["Na+"] / m.fs.enrtl_state[n_evap].mol_mass_ion_molecule,
        "Cl-": sb_enrtl.mw_comp["Cl-"] / m.fs.enrtl_state[n_evap].mol_mass_ion_molecule,
    }

    # Add constraints to link the outlet temperature, pressure, and mass flowrate of the evaporator brine with the eNRTL properties block.
    # Note that, since the flow from the seawater property package is in terms of total TDS, we convert the flow from TDS to the respective ions in the seawater.
    m.fs.enrtl_state[n_evap].eq_enrtl_temperature = Constraint(
        expr=(
            sb_enrtl.temperature
            == m.fs.evaporator[n_evap].properties_brine[0].temperature
        )
    )

    m.fs.enrtl_state[n_evap].eq_enrtl_pressure = Constraint(
        expr=(sb_enrtl.pressure == m.fs.evaporator[n_evap].properties_brine[0].pressure)
    )

    m.fs.enrtl_state[n_evap].eq_enrtl_flow_mass_H2O = Constraint(
        expr=(
            sb_enrtl.flow_mass_phase_comp["Liq", "H2O"]
            == m.fs.evaporator[n_evap]
            .properties_brine[0]
            .flow_mass_phase_comp["Liq", "H2O"]
        )
    )

    def enrtl_flow_mass_ion_comp(b, j):
        return sb_enrtl.flow_mass_phase_comp["Liq", j] == (
            (
                m.fs.evaporator[n_evap]
                .properties_brine[0]
                .flow_mass_phase_comp["Liq", "TDS"]
                * b.mass_ratio_ion[j]
            )
        )

    m.fs.enrtl_state[n_evap].enrtl_flow_mass_ion_comp_eq = Constraint(
        m.fs.set_ions_single, rule=enrtl_flow_mass_ion_comp
    )

    # Add expression to calculate the mean ionic activity coefficient.
    # It can be commented when the variable is not used.
    # The equation is taken from reference [1], page 28.
    # Note that gamma value is used (not log gamma), so to convert it to molal basis we return the log(act_coeff).
    m.fs.enrtl_state[n_evap].mean_act_coeff = Expression(
        expr=log(
            (
                sb_enrtl.act_coeff_phase_comp["Liq", "Na+"]
                ** m.fs.ion_coeff_single["Na+"]
                * sb_enrtl.act_coeff_phase_comp["Liq", "Cl-"]
                ** m.fs.ion_coeff_single["Cl-"]
            )
            ** (1 / sum(m.fs.ion_coeff_single[j] for j in m.fs.set_ions_single))
        )
    )

    # Add expressions to convert mean ionic activity coefficient to molal basis.
    m.fs.enrtl_state[n_evap].conv_mole_frac_to_molal = Expression(
        expr=log(
            1
            + (sb_enrtl.mw_comp["H2O"] * 2 * m.fs.molal_conc_solute[n_evap])
            / 1  # 1 kg of solvent
        )
    )
    m.fs.enrtl_state[n_evap].molal_mean_act_coeff = Expression(
        expr=(
            m.fs.enrtl_state[n_evap].mean_act_coeff
            - m.fs.enrtl_state[n_evap].conv_mole_frac_to_molal
        )
    )


def add_enrtl_method_multi(m, n_evap=None):
    sb_enrtl = m.fs.enrtl_state[n_evap].properties[0]  # renaming the block

    # Populate eNRTL state block
    populate_enrtl_state_vars_multi(sb_enrtl, base="FpcTP")

    # Calculate the total mass of the solute molecule and the mass ratio of each ion in the solute molecule.
    m.fs.enrtl_state[n_evap].mol_mass_ion_molecule_nacl = sum(
        m.fs.ion_coeff_nacl[j] * sb_enrtl.mw_comp[j] for j in m.fs.set_ions_nacl
    )

    m.fs.enrtl_state[n_evap].mol_mass_ion_molecule_na2so4 = sum(
        m.fs.ion_coeff_na2so4[j] * sb_enrtl.mw_comp[j] for j in m.fs.set_ions_na2so4
    )

    m.fs.enrtl_state[n_evap].mass_ratio_ion = {
        "Na+": sb_enrtl.mw_comp["Na+"]
        / (
            m.fs.enrtl_state[n_evap].mol_mass_ion_molecule_nacl
            + m.fs.enrtl_state[n_evap].mol_mass_ion_molecule_na2so4
        ),
        "Cl-": sb_enrtl.mw_comp["Cl-"]
        / m.fs.enrtl_state[n_evap].mol_mass_ion_molecule_nacl,
        "SO4_2-": sb_enrtl.mw_comp["SO4_2-"]
        / m.fs.enrtl_state[n_evap].mol_mass_ion_molecule_na2so4,
    }

    # Add constraints to link the outlet temperature, pressure, and mass flowrate of the evaporator brine with the eNRTL properties block.
    # Note that, since the flow from the seawater property package is in terms of total TDS, we convert the flow from TDS to the respective ions in the seawater.
    m.fs.enrtl_state[n_evap].eq_enrtl_temperature = Constraint(
        expr=(
            sb_enrtl.temperature
            == m.fs.evaporator[n_evap].properties_brine[0].temperature
        )
    )

    m.fs.enrtl_state[n_evap].eq_enrtl_pressure = Constraint(
        expr=(sb_enrtl.pressure == m.fs.evaporator[n_evap].properties_brine[0].pressure)
    )

    m.fs.enrtl_state[n_evap].eq_enrtl_flow_mass_H2O = Constraint(
        expr=(
            sb_enrtl.flow_mass_phase_comp["Liq", "H2O"]
            == m.fs.evaporator[n_evap]
            .properties_brine[0]
            .flow_mass_phase_comp["Liq", "H2O"]
        )
    )

    def enrtl_flow_mass_ion_comp(b, j):
        return sb_enrtl.flow_mass_phase_comp["Liq", j] == (
            (
                m.fs.evaporator[n_evap]
                .properties_brine[0]
                .flow_mass_phase_comp["Liq", "TDS"]
                * b.mass_ratio_ion[j]
            )
        )

    m.fs.enrtl_state[n_evap].enrtl_flow_mass_ion_comp_eq = Constraint(
        m.fs.set_ions_multi, rule=enrtl_flow_mass_ion_comp
    )

    # Add expression to calculate the mean ionic activity coefficient.
    # It can be commented when the variable is not used.
    # The equation is taken from reference [1], page 28.
    # Note that gamma value is used (not log gamma), so to convert it to molal basis we return the log(act_coeff).
    m.fs.enrtl_state[n_evap].mean_act_coeff = Expression(
        expr=log(
            (
                sb_enrtl.act_coeff_phase_comp["Liq", "Na+"]
                ** m.fs.ion_coeff_multi["Na+"]
                * sb_enrtl.act_coeff_phase_comp["Liq", "Cl-"]
                ** m.fs.ion_coeff_multi["Cl-"]
                * sb_enrtl.act_coeff_phase_comp["Liq", "SO4_2-"]
                ** m.fs.ion_coeff_multi["SO4_2-"]
            )
            ** (1 / sum(m.fs.ion_coeff_multi[j] for j in m.fs.set_ions_multi))
        )
    )

    # Add expressions to convert mean ionic activity coefficient to molal basis.
    m.fs.enrtl_state[n_evap].conv_mole_frac_to_molal = Expression(
        expr=log(
            1
            + (sb_enrtl.mw_comp["H2O"] * 2 * m.fs.molal_conc_solute[n_evap])
            / 1  # 1 kg of solvent
        )
    )
    m.fs.enrtl_state[n_evap].molal_mean_act_coeff = Expression(
        expr=(
            m.fs.enrtl_state[n_evap].mean_act_coeff
            - m.fs.enrtl_state[n_evap].conv_mole_frac_to_molal
        )
    )


def add_enrtl_method_AHP(m):
    sb_enrtl_gen = m.fs.enrtl_state_gen.properties[0]  # renaming the block

    # Populate eNRTL state block
    populate_enrtl_state_vars_gen(sb_enrtl_gen, base="FpcTP")

    # Calculate the total mass of the solute molecule and the mass ratio of each ion in the solute molecule.
    m.fs.enrtl_state_gen.mol_mass_ion_molecule = sum(
        m.fs.ion_coeff_AHP[j] * sb_enrtl_gen.mw_comp[j] for j in m.fs.set_ions_AHP
    )
    m.fs.enrtl_state_gen.mass_ratio_ion = {
        "Li+": sb_enrtl_gen.mw_comp["Li+"] / m.fs.enrtl_state_gen.mol_mass_ion_molecule,
        "Br-": sb_enrtl_gen.mw_comp["Br-"] / m.fs.enrtl_state_gen.mol_mass_ion_molecule,
    }

    # Add constraints to link the outlet temperature, pressure, and mass flowrate of the generator brine with the eNRTL properties block.
    # Note that, since the flow from the seawater property package is in terms of total TDS, we convert the flow from TDS to the respective ions in the seawater.
    m.fs.enrtl_state_gen.eq_enrtl_temperature = Constraint(
        expr=(
            sb_enrtl_gen.temperature == m.fs.generator.properties_brine[0].temperature
        )
    )

    m.fs.enrtl_state_gen.eq_enrtl_pressure = Constraint(
        expr=(sb_enrtl_gen.pressure == m.fs.generator.properties_brine[0].pressure)
    )

    m.fs.enrtl_state_gen.eq_enrtl_flow_mass_H2O = Constraint(
        expr=(
            sb_enrtl_gen.flow_mass_phase_comp["Liq", "H2O"]
            == m.fs.generator.properties_brine[0].flow_mass_phase_comp["Liq", "H2O"]
        )
    )

    def enrtl_flow_mass_ion_comp(b, j):
        return sb_enrtl_gen.flow_mass_phase_comp["Liq", j] == (
            (
                m.fs.generator.properties_brine[0].flow_mass_phase_comp["Liq", "TDS"]
                * b.mass_ratio_ion[j]
            )
        )

    m.fs.enrtl_state_gen.enrtl_flow_mass_ion_comp_eq = Constraint(
        m.fs.set_ions_AHP, rule=enrtl_flow_mass_ion_comp
    )

    # Add expression to calculate the mean ionic activity coefficient.
    # It can be commented when the variable is not used.
    # The equation is taken from reference [1], page 28.
    # Note that gamma value is used (not log gamma), so to convert it to molal basis we return the log(act_coeff).
    m.fs.enrtl_state_gen.mean_act_coeff = Expression(
        expr=log(
            (
                sb_enrtl_gen.act_coeff_phase_comp["Liq", "Li+"]
                ** m.fs.ion_coeff_AHP["Li+"]
                * sb_enrtl_gen.act_coeff_phase_comp["Liq", "Br-"]
                ** m.fs.ion_coeff_AHP["Br-"]
            )
            ** (1 / sum(m.fs.ion_coeff_AHP[j] for j in m.fs.set_ions_AHP))
        )
    )

    # Add expressions to convert mean ionic activity coefficient to
    # molal basis.
    m.fs.enrtl_state_gen.conv_mole_frac_to_molal = Expression(
        expr=log(
            1
            + (sb_enrtl_gen.mw_comp["H2O"] * 2 * m.fs.molal_conc_solute_gen)
            / 1  # 1 kg of solvent
        )
    )
    m.fs.enrtl_state_gen.molal_mean_act_coeff = Expression(
        expr=(
            m.fs.enrtl_state_gen.mean_act_coeff
            - m.fs.enrtl_state_gen.conv_mole_frac_to_molal
        )
    )


def set_scaling(m):
    # Scaling factors are added for all the variables
    for var in m.fs.component_data_objects(pyo.Var, descend_into=True):
        if "temperature" in var.name:
            iscale.set_scaling_factor(var, 1e2)
        if "lmtd" in var.name:
            iscale.set_scaling_factor(var, 1e-1)
        if "delta_temperature_in" in var.name:
            iscale.set_scaling_factor(var, 1e-1)
        if "delta_temperature_out" in var.name:
            iscale.set_scaling_factor(var, 1e-1)
        if "pressure" in var.name:
            iscale.set_scaling_factor(var, 1e-5)
        if "dens_mass_" in var.name:
            iscale.set_scaling_factor(var, 1e-3)
        if "flow_mass_phase_comp" in var.name:
            iscale.set_scaling_factor(var, 1e1)
        if "flow_mol_phase_comp" in var.name:
            iscale.set_scaling_factor(var, 1e-2)
        if "area" in var.name:
            iscale.set_scaling_factor(var, 1e-2)
        if "heat_transfer" in var.name:
            iscale.set_scaling_factor(var, 1e-5)
        if "heat" in var.name:
            iscale.set_scaling_factor(var, 1e-5)
        if "U" in var.name:
            iscale.set_scaling_factor(var, 1e-3)
        if "work" in var.name:
            iscale.set_scaling_factor(var, 1e-5)
        if "split_fraction" in var.name:
            iscale.set_scaling_factor(var, 1e-1)

    # Calculate scaling factors
    iscale.calculate_scaling_factors(m)


def set_model_inputs(m):
    # Feed
    # Assumed a constant feedflowrate of seawater to be 0.25kg/s
    # TDS flowrate = [(TDS concentration=23,000ppm * Seawater feed flowrate=0.25)]/1.0e6
    m.fs.feed_sw.properties[0].flow_mass_phase_comp["Liq", "H2O"].fix(0.24)  # kg/s
    m.fs.feed_sw.properties[0].flow_mass_phase_comp["Liq", "TDS"].fix(0.0058)  # kg/s
    m.fs.feed_sw.properties[0].temperature.fix(27 + 273.15)  # K
    m.fs.feed_sw.properties[0].pressure.fix(101325)  # Pa

    # Inlet data for pump
    m.fs.pump.inlet.flow_mass_phase_comp[0, "Liq", "H2O"].fix(0.45)  # kg/s
    m.fs.pump.inlet.flow_mass_phase_comp[0, "Liq", "TDS"].fix(0.55)  # kg/s
    m.fs.pump.inlet.temperature.fix(
        150 + 273.15
    )  # K, set point of heat transfer fluid from solar array to generator in [2] is 180degC
    m.fs.pump.inlet.pressure.fix(10000)  # Pa

    m.fs.pump.deltaP.fix(2e3)  # Pa
    m.fs.pump.efficiency_pump.fix(0.7)

    # Inlet data for economizer
    m.fs.economizer.tube_inlet.flow_mass_phase_comp[0, "Liq", "H2O"].fix(0.35)  # kg/s
    m.fs.economizer.tube_inlet.flow_mass_phase_comp[0, "Liq", "TDS"].fix(0.65)  # kg/s
    m.fs.economizer.tube_inlet.temperature.fix(200 + 273.15)  # K
    m.fs.economizer.tube_inlet.pressure.fix(30000)  # Pa

    m.fs.economizer.area.fix(40)  # m^2
    m.fs.economizer.overall_heat_transfer_coefficient.fix(600)  # W/K-m^2
    m.fs.economizer.crossflow_factor.fix(0.5)

    # Inlet data for generator
    m.fs.generator.outlet_vapor.pressure[0].fix(30e3)  # Pa
    m.fs.generator.U.fix(500)  # W/K-m^2
    m.fs.generator.area.fix(10)  # m^2
    m.fs.generator.heat_transfer.fix(111e3)  # W, average Qin from table 4 of [2]
    m.fs.generator.delta_temperature_in.fix(10)  # K

    # Inlet data for expansion Valve
    m.fs.expansion_valve.deltaP.fix(-20e3)  # Pa
    m.fs.expansion_valve.efficiency_pump.fix(0.7)

    # Inlet data for mixer
    # These are approximate values and are unfixed later during initialization to close the loop between MED and AHP
    m.fs.mixer.inlet_1.flow_mass_phase_comp[0, "Liq", "H2O"].fix(0.15)  # kg/s
    m.fs.mixer.inlet_1.flow_mass_phase_comp[0, "Liq", "TDS"].fix(0)  # kg/s
    m.fs.mixer.inlet_1.pressure.fix(31000)  # Pa
    m.fs.mixer.inlet_1.temperature.fix(65 + 273.15)  # K

    # Inlet data for absorber
    m.fs.absorber.overall_heat_transfer_coefficient.fix(500)  # W/K-m^2
    m.fs.absorber.shell_outlet.temperature.fix(75 + 273.15)  # K
    m.fs.absorber.crossflow_factor.fix(0.5)

    # Inlet data for condenser[1]
    m.fs.condenser[1].outlet.temperature[0].fix(51 + 273.15)  # K

    # Inlet data for Evaporator[1]
    m.fs.evaporator[1].outlet_brine.temperature[0].fix(52 + 273.15)  # K
    m.fs.evaporator[1].U.fix(1200)  # W/K-m^2
    m.fs.evaporator[1].area.fix(10)  # m^2
    m.fs.evaporator[1].delta_temperature_in.fix(2)  # K
    m.fs.evaporator[1].delta_temperature_out.fix(2.5)  # K

    # Inlet data for Condenser[2]
    m.fs.condenser[2].outlet.temperature[0].fix(53 + 273.15)  # K

    # Inlet data for Evaporator[2]
    m.fs.evaporator[2].U.fix(1000)  # W/K-m^2
    m.fs.evaporator[2].area.fix(30)  # m^2
    m.fs.evaporator[2].outlet_brine.temperature[0].fix(55 + 273.15)  # K
    m.fs.evaporator[2].delta_temperature_in.fix(8)  # K
    m.fs.evaporator[2].delta_temperature_out.fix(2.5)  # K

    # Inlet data for Condenser[3]
    m.fs.condenser[3].outlet.temperature[0].fix(58 + 273.15)  # K

    # Inlet data for Evaporator[3]
    m.fs.evaporator[3].U.fix(1000)  # W/K-m^2
    m.fs.evaporator[3].area.fix(20)  # m^2
    m.fs.evaporator[3].outlet_brine.temperature[0].fix(65 + 273.15)  # K
    m.fs.evaporator[3].delta_temperature_in.fix(10)  # K
    m.fs.evaporator[3].delta_temperature_out.fix(2.5)  # K

    # Inlet data for separator
    split_frac_a = 0.50
    m.fs.separator.split_fraction[0, "outlet_1"].fix(split_frac_a)
    m.fs.separator.outlet_1_state[0.0].flow_mass_phase_comp["Liq", "H2O"].fix(0)
    m.fs.separator.outlet_2_state[0.0].flow_mass_phase_comp["Liq", "H2O"].fix(0)

    # Inlet data for Condenser[4]
    m.fs.condenser[4].outlet.temperature[0].fix(68 + 273.15)  # K

    # Inlet data for Translator block
    m.fs.tblock.properties_out[0].flow_mass_phase_comp["Liq", "TDS"].fix(0)


def initialize(m, solver=None, optarg=None, outlvl=idaeslog.NOTSET):
    # Initialize feed
    m.fs.feed_sw.properties[0].mass_frac_phase_comp["Liq", "TDS"]
    solver.solve(m.fs.feed_sw)
    m.fs.feed_sw.initialize(outlvl=outlvl)

    # Initialize pump
    m.fs.pump.initialize(outlvl=outlvl)

    # Initialize economizer
    propagate_state(m.fs.pump_to_economizer)
    m.fs.economizer.initialize(outlvl=outlvl)

    # Initialize generator
    propagate_state(m.fs.economizer_to_generator)
    m.fs.generator.initialize(outlvl=outlvl)

    m.fs.economizer.tube_inlet.flow_mass_phase_comp[0, "Liq", "H2O"].unfix()
    m.fs.economizer.tube_inlet.flow_mass_phase_comp[0, "Liq", "TDS"].unfix()
    m.fs.economizer.tube_inlet.temperature.unfix()
    m.fs.economizer.tube_inlet.pressure.unfix()

    # Initialize economizer again to close the loop between generator and economizer
    propagate_state(m.fs.generator_to_economizer)
    m.fs.economizer.initialize(outlvl=outlvl)

    # Initialize expansion valve
    propagate_state(m.fs.economizer_to_valve)
    m.fs.expansion_valve.initialize(outlvl=outlvl)

    # Initailze mixer
    propagate_state(m.fs.valve_to_mixer)
    m.fs.mixer.initialize(optarg=optarg, outlvl=outlvl)

    # Initialize absorber
    propagate_state(m.fs.feed_to_absorber)
    propagate_state(m.fs.mixer_to_absorber)
    m.fs.absorber.initialize(outlvl=outlvl)

    # Initialize condenser [1]
    propagate_state(m.fs.generator_to_condenser)
    m.fs.condenser[1].initialize_build()

    # Initialize evaporator [1]
    propagate_state(m.fs.absorber_to_evaporator_feed)
    m.fs.evaporator[1].initialize(outlvl=outlvl)

    # Initialize condenser [2] with arbitrary heat_transfer value
    propagate_state(m.fs.evap1vapor_to_cond2)
    m.fs.condenser[2].initialize_build(heat=-m.fs.evaporator[1].heat_transfer.value)

    # Initialize evaporator [2]
    propagate_state(m.fs.evap1brine_to_evap2feed)
    m.fs.evaporator[2].initialize(outlvl=outlvl)

    # Initialize condenser [3] with arbitrary heat_transfer value
    propagate_state(m.fs.evap2vapor_to_cond3)
    m.fs.condenser[3].initialize_build(heat=-m.fs.evaporator[2].heat_transfer.value)

    # Initialize evaporator [3]
    propagate_state(m.fs.evap2brine_to_evap3feed)
    m.fs.evaporator[3].initialize(outlvl=outlvl)

    # Initialize separator
    propagate_state(m.fs.evap3vapor_to_separator)
    m.fs.separator.initialize(optarg=optarg, outlvl=outlvl)

    # Initialize condenser [4]
    propagate_state(m.fs.separator_to_condenser)
    m.fs.condenser[4].initialize(outlvl=outlvl)

    # Initialize translator block
    propagate_state(m.fs.separator_to_tblock)
    m.fs.tblock.initialize(optarg=optarg, outlvl=outlvl)

    m.fs.mixer.inlet_1.flow_mass_phase_comp[0, "Liq", "H2O"].unfix()
    m.fs.mixer.inlet_1.flow_mass_phase_comp[0, "Liq", "TDS"].unfix()
    m.fs.mixer.inlet_1.pressure.unfix()
    m.fs.mixer.inlet_1.temperature.unfix()

    # Initailize mixer
    propagate_state(m.fs.tblock_to_mixer)
    m.fs.mixer.initialize(optarg=optarg, outlvl=outlvl)

    print()
    print("****** Start initialization")

    if not degrees_of_freedom(m) == 0:
        raise ConfigurationError(
            "The degrees of freedom after building the model are not 0. "
            "You have {} degrees of freedom. "
            "Please check your inputs to ensure a square problem "
            "before initializing the model.".format(degrees_of_freedom(m))
        )
    init_results = solver.solve(m, tee=False)
    print(" Initialization solver status:", init_results.solver.termination_condition)
    print("****** End initialization")
    print()


def add_bounds(m):
    for i in m.fs.set_evaporators:
        m.fs.evaporator[i].area.setlb(10)
        m.fs.evaporator[i].area.setub(None)
        m.fs.evaporator[i].outlet_brine.temperature[0].setub(
            73 + 273.15
        )  # in K to protect pipes


def print_results(m):
    sw_blk_gen = m.fs.generator.properties_feed[0]
    brine_blk_gen = m.fs.generator.properties_brine[0]
    vapor_blk_gen = m.fs.generator.properties_vapor[0]
    print()
    print()
    print(
        "===================================================================================="
    )
    print("Unit : m.fs.generator".format())
    print(
        "------------------------------------------------------------------------------------"
    )
    print("    Unit performance")
    print()
    print("    Variables:")
    print()
    print("    Key   Value")
    print(
        "      delta temperature_in  : {:>4.3f}".format(
            value(m.fs.generator.delta_temperature_in)
        )
    )
    print(
        "      delta temperature_out : {:>4.3f}".format(
            value(m.fs.generator.delta_temperature_out)
        )
    )
    print("                      Area  : {:>4.3f}".format(value(m.fs.generator.area)))
    print("                         U  : {:>4.3f}".format(value(m.fs.generator.U)))
    print(
        "                         Qin: {:>4.3f}".format(
            value(m.fs.generator.heat_transfer)
        )
    )

    print(
        "------------------------------------------------------------------------------------"
    )
    print("   Stream Table")
    print(
        "                                      inlet_feed    outlet_brine   outlet_vapor"
    )
    print(
        "   flow_mass_phase_comp (kg/s) {:>15.4f} {:>14.4f} {:>14.4f}".format(
            value(
                sw_blk_gen.flow_mass_phase_comp["Liq", "H2O"]
                + sw_blk_gen.flow_mass_phase_comp["Liq", "TDS"]
            ),
            value(
                brine_blk_gen.flow_mass_phase_comp["Liq", "H2O"]
                + brine_blk_gen.flow_mass_phase_comp["Liq", "TDS"]
            ),
            value(vapor_blk_gen.flow_mass_phase_comp["Vap", "H2O"]),
        )
    )
    print(
        "   mass_frac_phase_comp (Liq, H2O){:>12.4f} {:>14.4f}            -".format(
            value(sw_blk_gen.mass_frac_phase_comp["Liq", "H2O"]),
            value(brine_blk_gen.mass_frac_phase_comp["Liq", "H2O"]),
        )
    )
    print(
        "   mass_frac_phase_comp (Liq, TDS){:>12.4f} {:>14.4f}            -".format(
            value(sw_blk_gen.mass_frac_phase_comp["Liq", "TDS"]),
            value(brine_blk_gen.mass_frac_phase_comp["Liq", "TDS"]),
        )
    )
    print(
        "   mole_frac_phase_comp (Liq, H2O){:>12.4f} {:>14.4f} {:>14.4f}".format(
            value(sw_blk_gen.mole_frac_phase_comp["Liq", "H2O"]),
            value(brine_blk_gen.mole_frac_phase_comp["Liq", "H2O"]),
            value(vapor_blk_gen.mole_frac_phase_comp["Liq", "H2O"]),
        )
    )
    print(
        "   mole_frac_phase_comp (Liq, TDS){:>12.4f} {:>14.4f}            -".format(
            value(sw_blk_gen.mole_frac_phase_comp["Liq", "TDS"]),
            value(brine_blk_gen.mole_frac_phase_comp["Liq", "TDS"]),
        )
    )
    print(
        "   temperature (K) {:>27.4f} {:>14.4f} {:>14.4f}".format(
            value(sw_blk_gen.temperature),
            value(brine_blk_gen.temperature),
            value(vapor_blk_gen.temperature),
        )
    )
    print(
        "   pressure (Pa) {:>29.4f} {:>14.4f} {:>14.4f}".format(
            value(sw_blk_gen.pressure),
            value(brine_blk_gen.pressure),
            value(vapor_blk_gen.pressure),
        )
    )
    print()

    for i in m.fs.set_condensers:
        m.fs.condenser[i].report()

    m.fs.molal_conc_solute_feed_AHP = (
        value(m.fs.generator.inlet_feed.flow_mass_phase_comp[0, "Liq", "TDS"])
        / value(m.fs.properties_feed.mw_comp["TDS"])
    ) / value(m.fs.generator.inlet_feed.flow_mass_phase_comp[0, "Liq", "H2O"])

    for i in m.fs.set_evaporators:
        m.fs.molal_conc_solute_feed = (
            value(m.fs.evaporator[i].inlet_feed.flow_mass_phase_comp[0, "Liq", "TDS"])
            / value(m.fs.properties_feed_sw.mw_comp["TDS"])
        ) / value(m.fs.evaporator[i].inlet_feed.flow_mass_phase_comp[0, "Liq", "H2O"])

        sw_blk = m.fs.evaporator[i].properties_feed[0]
        brine_blk = m.fs.evaporator[i].properties_brine[0]
        vapor_blk = m.fs.evaporator[i].properties_vapor[0]
        print()
        print()
        print(
            "===================================================================================="
        )
        if solve_nonideal:
            print("Unit : m.fs.evaporator[{}] (non-ideal)".format(i))
        else:
            print("Unit : m.fs.evaporator[{}] (ideal)".format(i))
        print(
            "------------------------------------------------------------------------------------"
        )
        print("    Unit performance")
        print()
        print("    Variables:")
        print()
        print("    Key   Value")
        print(
            "      delta temperature_in  : {:>4.3f}".format(
                value(m.fs.evaporator[i].delta_temperature_in)
            )
        )
        print(
            "      delta temperature_out : {:>4.3f}".format(
                value(m.fs.evaporator[i].delta_temperature_out)
            )
        )
        print(
            "                      Area  : {:>4.3f}".format(
                value(m.fs.evaporator[i].area)
            )
        )
        print(
            "                         U  : {:>4.3f}".format(value(m.fs.evaporator[i].U))
        )
        print("                   UA_term  : {:>4.3f}".format(value(m.fs.UA_term[i])))
        if solve_nonideal:
            print(
                "             act_coeff* H2O : {:>4.4f} (log:{:>4.4f})".format(
                    value(
                        m.fs.enrtl_state[i]
                        .properties[0]
                        .act_coeff_phase_comp["Liq", "H2O"]
                    ),
                    value(
                        log(
                            m.fs.enrtl_state[i]
                            .properties[0]
                            .act_coeff_phase_comp["Liq", "H2O"]
                        )
                    ),
                )
            )
            if run_multi:
                for j in m.fs.set_ions_multi:
                    print(
                        "             act_coeff* {} : {:>4.4f} (log:{:>4.4f})".format(
                            j,
                            value(
                                m.fs.enrtl_state[i]
                                .properties[0]
                                .act_coeff_phase_comp["Liq", j]
                            ),
                            value(
                                log(
                                    m.fs.enrtl_state[i]
                                    .properties[0]
                                    .act_coeff_phase_comp["Liq", j]
                                )
                            ),
                        )
                    )
                    print(
                        "        mean_ion_actv_coeff : {:>4.4f} (log: {:>4.4f})".format(
                            exp(value(m.fs.enrtl_state[i].mean_act_coeff)),
                            value(m.fs.enrtl_state[i].mean_act_coeff),
                        )
                    )
                    print(
                        " molal mean_ionic_actv_coeff: {:>4.4f} (log: {:>4.4f})".format(
                            exp(value(m.fs.enrtl_state[i].molal_mean_act_coeff)),
                            value(m.fs.enrtl_state[i].molal_mean_act_coeff),
                        )
                    )
                    print("    *calculated with eNRTL")
            else:
                for j in m.fs.set_ions_single:
                    print(
                        "             act_coeff* {} : {:>4.4f} (log:{:>4.4f})".format(
                            j,
                            value(
                                m.fs.enrtl_state[i]
                                .properties[0]
                                .act_coeff_phase_comp["Liq", j]
                            ),
                            value(
                                log(
                                    m.fs.enrtl_state[i]
                                    .properties[0]
                                    .act_coeff_phase_comp["Liq", j]
                                )
                            ),
                        )
                    )
                    print(
                        "        mean_ion_actv_coeff : {:>4.4f} (log: {:>4.4f})".format(
                            exp(value(m.fs.enrtl_state[i].mean_act_coeff)),
                            value(m.fs.enrtl_state[i].mean_act_coeff),
                        )
                    )
                    print(
                        " molal mean_ionic_actv_coeff: {:>4.4f} (log: {:>4.4f})".format(
                            exp(value(m.fs.enrtl_state[i].molal_mean_act_coeff)),
                            value(m.fs.enrtl_state[i].molal_mean_act_coeff),
                        )
                    )
                    print("    *calculated with eNRTL")

        else:
            print(
                "              act_coeff H2O : {:>4.4f}".format(
                    value(m.fs.act_coeff[i])
                )
            )
        print(
            "------------------------------------------------------------------------------------"
        )
        print("   Stream Table")
        print(
            "                                      inlet_feed    outlet_brine   outlet_vapor"
        )
        print(
            "   flow_mass_phase_comp (kg/s) {:>15.4f} {:>14.4f} {:>14.4f}".format(
                value(
                    sw_blk.flow_mass_phase_comp["Liq", "H2O"]
                    + sw_blk.flow_mass_phase_comp["Liq", "TDS"]
                ),
                value(
                    brine_blk.flow_mass_phase_comp["Liq", "H2O"]
                    + brine_blk.flow_mass_phase_comp["Liq", "TDS"]
                ),
                value(vapor_blk.flow_mass_phase_comp["Vap", "H2O"]),
            )
        )
        print(
            "   mass_frac_phase_comp (Liq, H2O){:>12.4f} {:>14.4f}            -".format(
                value(sw_blk.mass_frac_phase_comp["Liq", "H2O"]),
                value(brine_blk.mass_frac_phase_comp["Liq", "H2O"]),
            )
        )
        print(
            "   mass_frac_phase_comp (Liq, TDS){:>12.4f} {:>14.4f}            -".format(
                value(sw_blk.mass_frac_phase_comp["Liq", "TDS"]),
                value(brine_blk.mass_frac_phase_comp["Liq", "TDS"]),
            )
        )
        print(
            "   mole_frac_phase_comp (Liq, H2O){:>12.4f} {:>14.4f} {:>14.4f}".format(
                value(sw_blk.mole_frac_phase_comp["Liq", "H2O"]),
                value(brine_blk.mole_frac_phase_comp["Liq", "H2O"]),
                value(vapor_blk.mole_frac_phase_comp["Liq", "H2O"]),
            )
        )
        print(
            "   mole_frac_phase_comp (Liq, TDS){:>12.4f} {:>14.4f}            -".format(
                value(sw_blk.mole_frac_phase_comp["Liq", "TDS"]),
                value(brine_blk.mole_frac_phase_comp["Liq", "TDS"]),
            )
        )
        print(
            "   molal_conc_solute (mol TDS/kg H2O) {:>8.4f} {:>14.4f}            -".format(
                m.fs.molal_conc_solute_feed, value(m.fs.molal_conc_solute[i])
            )
        )
        print(
            "   temperature (K) {:>27.4f} {:>14.4f} {:>14.4f}".format(
                value(sw_blk.temperature),
                value(brine_blk.temperature),
                value(vapor_blk.temperature),
            )
        )
        print(
            "   pressure (Pa) {:>29.4f} {:>14.4f} {:>14.4f}".format(
                value(sw_blk.pressure),
                value(brine_blk.pressure),
                value(vapor_blk.pressure),
            )
        )
        print(
            "   saturation pressure (Pa) {:>18.4f} {:>14.4f} {:>14.4f}".format(
                value(sw_blk.pressure_sat),
                value(brine_blk.pressure_sat),
                value(vapor_blk.pressure_sat),
            )
        )
        print()
        if solve_nonideal:
            print("   eNRTL state block")
            print(
                "   flow_mass_phase_comp (Liq, H2O) {:>11.4f}".format(
                    value(
                        m.fs.enrtl_state[i]
                        .properties[0]
                        .flow_mass_phase_comp["Liq", "H2O"]
                    )
                )
            )
            if run_multi:
                for j in m.fs.set_ions_multi:
                    print(
                        "   flow_mass_phase_comp (Liq, {}) {:>11.4f}".format(
                            j,
                            value(
                                m.fs.enrtl_state[i]
                                .properties[0]
                                .flow_mass_phase_comp["Liq", j]
                            ),
                        )
                    )
                    sum_tds_brine_out = sum(
                        value(
                            m.fs.enrtl_state[i]
                            .properties[0]
                            .flow_mass_phase_comp["Liq", j]
                        )
                        for j in m.fs.set_ions_multi
                    )
                    print(
                        "    >>flow_mass_phase_comp (Liq, TDS) {:>8.4f}".format(
                            sum_tds_brine_out
                        )
                    )
                    if (
                        sum_tds_brine_out
                        - value(brine_blk.flow_mass_phase_comp["Liq", "TDS"])
                        >= 1e-1
                    ):
                        print(
                            "     **ERROR: Flow mass of TDS ({:>2.4f} kg/s) not equivalent"
                            " to sum of ions mass ({:>2.4f} kg/s)".format(
                                sum_tds_brine_out,
                                value(brine_blk.flow_mass_phase_comp["Liq", "TDS"]),
                            )
                        )
                        print("     Check balances!")
                    print(
                        "   temperature (K) {:>27.4f}".format(
                            value(m.fs.enrtl_state[i].properties[0].temperature)
                        )
                    )
                    print(
                        "   pressure (Pa) {:>29.4f}".format(
                            value(m.fs.enrtl_state[i].properties[0].pressure)
                        )
                    )
                    print()
            else:
                for j in m.fs.set_ions_single:
                    print(
                        "   flow_mass_phase_comp (Liq, {}) {:>11.4f}".format(
                            j,
                            value(
                                m.fs.enrtl_state[i]
                                .properties[0]
                                .flow_mass_phase_comp["Liq", j]
                            ),
                        )
                    )
                    sum_tds_brine_out = sum(
                        value(
                            m.fs.enrtl_state[i]
                            .properties[0]
                            .flow_mass_phase_comp["Liq", j]
                        )
                        for j in m.fs.set_ions_single
                    )
                    print(
                        "    >>flow_mass_phase_comp (Liq, TDS) {:>8.4f}".format(
                            sum_tds_brine_out
                        )
                    )
                    if (
                        sum_tds_brine_out
                        - value(brine_blk.flow_mass_phase_comp["Liq", "TDS"])
                        >= 1e-1
                    ):
                        print(
                            "     **ERROR: Flow mass of TDS ({:>2.4f} kg/s) not equivalent"
                            " to sum of ions mass ({:>2.4f} kg/s)".format(
                                sum_tds_brine_out,
                                value(brine_blk.flow_mass_phase_comp["Liq", "TDS"]),
                            )
                        )
                        print("     Check balances!")
                    print(
                        "   temperature (K) {:>27.4f}".format(
                            value(m.fs.enrtl_state[i].properties[0].temperature)
                        )
                    )
                    print(
                        "   pressure (Pa) {:>29.4f}".format(
                            value(m.fs.enrtl_state[i].properties[0].pressure)
                        )
                    )
                    print()
        print()
        print(
            "===================================================================================="
        )
        print()

    m.fs.separator.report()
    m.fs.tblock.report()
    m.fs.mixer.report()
    m.fs.absorber.report()
    m.fs.pump.report()
    m.fs.economizer.report()
    m.fs.expansion_valve.report()

    print("Variable                                 Value")
    print(
        " Total water produced (gal/min) {:>18.4f}".format(
            value(m.fs.total_water_produced_gpm)
        )
    )
    print(" Performance Ratio {:>31.4f}".format(value(m.fs.performance_ratio)))
    print(
        " Specific energy consumption (SC, kWh/m3) {:>8.4f}".format(
            value(m.fs.specific_energy_consumption)
        )
    )
    print(" Water recovery (%) {:>30.4f}".format(value(m.fs.water_recovery) * 100))
    for i in m.fs.set_evaporators:
        print(
            " Molal conc solute evap {} (mol/kg) {:>15.4f}".format(
                i, value(m.fs.molal_conc_solute[i])
            )
        )
    print()
    print()


def model_analysis(m, water_rec=None):
    # Unfix for optimization of variables
    # Economizer
    m.fs.economizer.area.unfix()
    m.fs.economizer.delta_temperature_in.unfix()
    m.fs.economizer.delta_temperature_out.unfix()

    # Generator
    m.fs.generator.area.unfix()
    m.fs.generator.outlet_brine.temperature[0].unfix()
    m.fs.generator.delta_temperature_in.unfix()
    m.fs.generator.delta_temperature_out.unfix()
    m.fs.generator.heat_transfer.unfix()

    # Absorber
    m.fs.absorber.area.unfix()

    # Condenser[1]
    m.fs.condenser[1].control_volume.heat[0].unfix()

    # Evaporator[1]
    m.fs.evaporator[1].outlet_brine.temperature[0].unfix()
    m.fs.evaporator[1].area.unfix()
    m.fs.evaporator[1].delta_temperature_in.unfix()

    # Condenser[2]
    m.fs.condenser[2].control_volume.heat[0].unfix()

    # Evaporator[2]
    m.fs.evaporator[2].area.unfix()
    m.fs.evaporator[2].outlet_brine.temperature[0].unfix()
    m.fs.evaporator[2].delta_temperature_in.unfix()

    # Condenser[3]
    m.fs.condenser[3].control_volume.heat[0].unfix()

    # Evaporator[3]
    m.fs.evaporator[3].area.unfix()
    m.fs.evaporator[3].outlet_brine.temperature[0].unfix()
    m.fs.evaporator[3].delta_temperature_in.unfix()

    # Separarator
    m.fs.separator.split_fraction[0, "outlet_1"].unfix()

    # Condenser[4]
    m.fs.condenser[4].control_volume.heat[0].unfix()

    # delta_temperature_in = condenser inlet temp - evaporator brine temp
    # delta_temperature_out = condenser outlet temp - evaporator brine temp
    @m.fs.Constraint(m.fs.set_evaporators)
    def eq_upper_bound_evaporators_delta_temprature_in(b, e):
        return b.evaporator[e].delta_temperature_in <= 10 * pyunits.K

    @m.fs.Constraint()
    def eq_upper_bound_generator_delta_temperature_in(b):
        return b.generator.delta_temperature_in <= 35 * pyunits.K

    @m.fs.Constraint()
    def eq_upper_bound_generator_delta_temprature_out(b):
        return b.generator.delta_temperature_out <= 35 * pyunits.K

    # Add constraint to make sure the pressure in the evaporators 2 and 3 is smaller than the pressure in evaporator 1.
    # Included for debugging purposes
    m.fs.set2_evaporators = RangeSet(m.fs.num_evaporators - 1)

    @m.fs.Constraint(m.fs.set2_evaporators)
    def eq_upper_bound_evaporators_pressure(b, e):
        return (
            b.evaporator[e + 1].outlet_brine.pressure[0]
            <= b.evaporator[e].outlet_brine.pressure[0]
        )

    # Add expression to calculate the UA term
    @m.fs.Expression(
        m.fs.set_evaporators, doc="Overall heat trasfer coefficient and area term"
    )
    def UA_term(b, e):
        return b.evaporator[e].area * b.evaporator[e].U

    @m.fs.Expression(doc="Overall heat trasfer coefficient and area term")
    def UA_term_gen(b):
        return b.generator.area * b.generator.U

    @m.fs.Constraint(doc="Generator area upper bound")
    def gen_area_upper_bound(b):
        return b.generator.area <= 500

    # Add constraints to prevent area from going to unreasonably high values
    @m.fs.Constraint(doc="Economizer area upper bound")
    def econ_area_upper_bound(b):
        return b.economizer.area <= 500

    @m.fs.Constraint(doc="Absorber area upper bound")
    def abs_area_upper_bound(b):
        return b.absorber.area <= 500

    m.fs.water_density = pyo.Param(initialize=1000, units=pyunits.kg / pyunits.m**3)

    # Calculate total water produced
    @m.fs.Expression()
    def total_water_produced_gpm(b):
        return pyo.units.convert(
            sum(
                b.condenser[e]
                .control_volume.properties_out[0]
                .flow_mass_phase_comp["Liq", "H2O"]
                for e in m.fs.set_condensers
            )
            / m.fs.water_density,
            to_units=pyunits.gallon / pyunits.minute,
        )

    # Backcalculation from [2] produced a latent heat of vaporization at T_ref of 73degC is 2,319.05 kJ/kg
    # Calculate performance ratio
    @m.fs.Expression()
    def performance_ratio(b):
        return (
            (
                b.condenser[1]
                .control_volume.properties_out[0]
                .flow_mass_phase_comp["Liq", "H2O"]
                + b.condenser[2]
                .control_volume.properties_out[0]
                .flow_mass_phase_comp["Liq", "H2O"]
                + b.condenser[3]
                .control_volume.properties_out[0]
                .flow_mass_phase_comp["Liq", "H2O"]
                + b.condenser[4]
                .control_volume.properties_out[0]
                .flow_mass_phase_comp["Liq", "H2O"]
            )
            * 2319.05
        ) / (b.generator.heat_transfer / 1000)

    # Calculate specific energy consumption
    m.fs.specific_energy_consumption = pyo.Var(
        initialize=11, units=pyunits.kW * pyunits.hour / pyunits.m**3, bounds=(0, 1e3)
    )

    @m.fs.Constraint(doc="Specific energy consumption [kWh/m^3]")
    def eq_specific_energy_consumption(b):
        return b.specific_energy_consumption == (
            pyo.units.convert(
                b.generator.heat_transfer, to_units=pyunits.kW  # in Watts
            )
            / pyo.units.convert(
                m.fs.total_water_produced_gpm, to_units=pyunits.m**3 / pyunits.hour
            )
        )

    # Add water recovery equation as a constraint
    m.fs.water_recovery = pyo.Var(
        initialize=0.2, bounds=(0, 1), units=pyunits.dimensionless, doc="Water recovery"
    )

    @m.fs.Constraint()
    def rule_water_recovery(b):
        return m.fs.water_recovery == (
            b.condenser[1]
            .control_volume.properties_out[0]
            .flow_mass_phase_comp["Liq", "H2O"]
            + b.condenser[2]
            .control_volume.properties_out[0]
            .flow_mass_phase_comp["Liq", "H2O"]
            + b.condenser[3]
            .control_volume.properties_out[0]
            .flow_mass_phase_comp["Liq", "H2O"]
            + b.condenser[4]
            .control_volume.properties_out[0]
            .flow_mass_phase_comp["Liq", "H2O"]
        ) / (
            m.fs.evaporator[1].inlet_feed.flow_mass_phase_comp[0, "Liq", "H2O"]
            + m.fs.evaporator[1].inlet_feed.flow_mass_phase_comp[0, "Liq", "TDS"]
        )

    @m.fs.Constraint()
    def water_recovery_ub(b):
        return b.water_recovery >= water_rec

    @m.fs.Constraint()
    def water_recovery_lb(b):
        return b.water_recovery <= water_rec


if __name__ == "__main__":
    optarg = {"max_iter": 500, "tol": 1e-8}
    solver = get_solver("ipopt", optarg)

    water_recovery_data = [0.7]  # adjust value for specific % of water recovery
    for c in range(len(water_recovery_data)):
        m = create_model()

        set_scaling(m)

        set_model_inputs(m)

        initialize(m, solver=solver)

        add_bounds(m)

        model_analysis(m, water_rec=water_recovery_data[c])

        results = solver.solve(m, tee=True)

        print_results(m)
