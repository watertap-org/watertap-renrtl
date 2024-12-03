###############################################################################
# WaterTAP Copyright (c) 2020-2024, The Regents of the University of California,
# through Lawrence Berkeley National Laboratory, Oak Ridge National Laboratory,
# National Renewable Energy Laboratory, and National Energy Technology
# Laboratory (subject to receipt of any required approvals from the U.S. Dept.
# of Energy). All rights reserved.
#
# Copyright 2023-2024, National Technology & Engineering Solutions of Sandia,
# LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the
# U.S. Government retains certain rights in this software
#
# Copyright 2023-2024, Pengfei Xu and Matthew D. Stuber and the University
# of Connecticut.
#
# Please see the files COPYRIGHT.md and LICENSE.md for full copyright and license
# information, respectively. These files are also available online at the URL
# "https://github.com/watertap-org/watertap/"
#
###############################################################################

import logging
import matplotlib.pyplot as plt

# Import Pyomo components
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
from pyomo.util.infeasible import log_infeasible_constraints, log_close_to_bounds

# Import IDAES components
import idaes.core.util.scaling as iscale
import idaes.logger as idaeslog
from idaes.core import FlowsheetBlock
from idaes.models.properties.modular_properties.base.generic_property import (
    GenericParameterBlock,
)
from idaes.core.solvers.get_solver import get_solver
from idaes.core.util.model_statistics import degrees_of_freedom
from idaes.core.util.initialization import propagate_state
from idaes.core import MaterialBalanceType
from idaes.models.unit_models import PressureChanger
from idaes.models.unit_models.pressure_changer import ThermodynamicAssumption
from idaes.core.util.exceptions import ConfigurationError

# Import property packages from WaterTAP
import watertap.property_models.seawater_prop_pack as props_sw
import watertap.property_models.water_prop_pack as props_w
from watertap.unit_models.mvc.components import Evaporator, Compressor, Condenser

# Import refined eNRTL configuration dictionary
import entrl_config_FpcTP

# Set logging options
logging.basicConfig(level=logging.INFO)
logging.getLogger("pyomo.repn.plugins.nl_writer").setLevel(logging.ERROR)


""" References:

[1] Matthew D. Stuber, Christopher Sullivan, Spencer A. Kirk, Jennifer
A. Farrand, Philip V. Schillai, Brian D. Fojtasek, and Aaron
H. Mandell. Pilot demonstration of concentrated solar-poweered
desalination of subsurface agricultural drainage water and other
brackish groundwater sources. Desalination, 355 (2015), 186-196.

"""


def populate_enrtl_state_vars(blk, base="FpcTP"):
    """Initialize state variables"""
    blk.temperature = 298.15
    blk.pressure = 101325

    if base == "FpcTP":
        feed_flow_mass = 10  # in kg/s
        feed_mass_frac_comp = {"Na+": 0.013768116, "Cl-": 0.021231884}
        feed_mass_frac_comp["H2O"] = 1 - sum(x for x in feed_mass_frac_comp.values())
        mw_comp = {"H2O": 18.015e-3, "Na+": 22.990e-3, "Cl-": 35.453e-3}

        for j in feed_mass_frac_comp:
            blk.flow_mol_phase_comp["Liq", j] = (
                feed_flow_mass * feed_mass_frac_comp[j] / mw_comp[j]
            )
            if j == "H2O":
                blk.flow_mol_phase_comp["Liq", j] /= 2


def create_model():
    m = ConcreteModel("Three-effect Mechanical Vapor Compression Model")

    m.fs = FlowsheetBlock(dynamic=False)

    # Declare a set for the ions in the electrolyte solution and the
    # stoichiometric coefficient of the ions in the solute molecule
    # and calculate the total mass of the solute molecule and the mass
    # ratio of each ion in the solute molecule.
    m.fs.set_ions = Set(initialize=["Na+", "Cl-"])
    m.fs.ion_coeff = {"Na+": 1, "Cl-": 1}

    # Add property packages for water and seawater.
    m.fs.properties_vapor = props_w.WaterParameterBlock()
    m.fs.properties_feed = props_sw.SeawaterParameterBlock()

    # Declare the evaporator and compressor units. Note: the
    # evaporator unit is a customized unit that includes a complete
    # condenser
    m.fs.num_evaporators = 3
    m.fs.set_evaporators = RangeSet(m.fs.num_evaporators)
    m.fs.evaporator = Evaporator(
        m.fs.set_evaporators,
        property_package_feed=m.fs.properties_feed,
        property_package_vapor=m.fs.properties_vapor,
    )
    m.fs.compressor = Compressor(property_package=m.fs.properties_vapor)
    m.fs.condenser = Condenser(
        m.fs.set_evaporators, property_package=m.fs.properties_vapor
    )

    # Add variable to calculate molal concentration of solute. The
    # upper bound is included to ensure that the molality of the
    # electrolyte solution is within saturation and the concentration
    # limits of eNRTL tau and alpha parameters.
    m.fs.molal_conc_solute = pyo.Var(
        m.fs.set_evaporators,
        initialize=2,
        bounds=(0, 6),
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

    # Add activity coefficient as a global variable for each
    # evaporator.
    m.fs.act_coeff = pyo.Var(
        m.fs.set_evaporators, initialize=1, units=pyunits.dimensionless, bounds=(0, 20)
    )

    # Add eNRTL method to calculate the activity coefficients for the
    # electrolyte solution. To do this, first declare a block to
    # include the generic properties needed by eNRTL as a state block
    # followed by a Generic Parameter Block that calls a configuration
    # file that includes eNRTL as the equation of state method.
    m.fs.enrtl_state = Block(m.fs.set_evaporators)

    m.fs.prop_enrtl = GenericParameterBlock(**entrl_config_FpcTP.configuration)

    for e in m.fs.set_evaporators:
        m.fs.enrtl_state[e].properties = m.fs.prop_enrtl.build_state_block([0])

        add_enrtl_method(m, n_evap=e)

    # Save the calculated activity coefficient in the global
    # activity coefficient variable.
    @m.fs.Constraint(m.fs.set_evaporators, doc="eNRTL activity coefficient for water")
    def eq_nonideal_activity_coefficient(b, e):
        return (
            b.act_coeff[e]
            == m.fs.enrtl_state[e].properties[0].act_coeff_phase_comp["Liq", "H2O"]
        )

    # Deactivate equilibrium equation from evaporator and include a
    # new equilibrium equation as a Constraint that includes the
    # activity coefficient. Note that since water is the only solvent
    # participating in the vapor-liquid equilibrium, the activity
    # coefficient and vapor pressure are of water as the
    # solvent.
    for e in m.fs.set_evaporators:
        m.fs.evaporator[e].eq_brine_pressure.deactivate()

    @m.fs.Constraint(m.fs.set_evaporators, doc="Vapor-liquid equilibrium equation")
    def _eq_phase_equilibrium(b, e):
        return b.evaporator[e].properties_brine[0].pressure == (
            m.fs.act_coeff[e]
            * b.evaporator[e].properties_brine[0].mole_frac_phase_comp["Liq", "H2O"]
            * b.evaporator[e].properties_vapor[0].pressure_sat
        )

    # Create and expand Arcs to create the model constraints necessary
    # to connect the units in the flowsheet.
    create_arcs(m)

    TransformationFactory("network.expand_arcs").apply_to(m)

    return m


def create_arcs(m):
    """Create arcs to connect the units in the flowsheet"""

    m.fs.evap1brine_to_evap2feed = Arc(
        source=m.fs.evaporator[1].outlet_brine,
        destination=m.fs.evaporator[2].inlet_feed,
        doc="Connect evaporator 1 brine outlet to evaporator 1 inlet",
    )
    m.fs.evap1vapor_to_cond2 = Arc(
        source=m.fs.evaporator[1].outlet_vapor,
        destination=m.fs.condenser[2].inlet,
        doc="Connect vapor outlet of evaporator 1 to condenser 2",
    )

    m.fs.evap2vapor_to_cond3 = Arc(
        source=m.fs.evaporator[2].outlet_vapor,
        destination=m.fs.condenser[3].inlet,
        doc="Connect vapor outlet of evaporator 2 to condenser 3",
    )
    m.fs.evap2brine_to_evap3feed = Arc(
        source=m.fs.evaporator[2].outlet_brine,
        destination=m.fs.evaporator[3].inlet_feed,
        doc="Connect evaporator 2 brine outlet to evaporator 3 inlet",
    )

    m.fs.evap3vapor_to_compressor = Arc(
        source=m.fs.evaporator[3].outlet_vapor,
        destination=m.fs.compressor.inlet,
        doc="Connect vapor outlet of evaporator 3 to compressor",
    )
    m.fs.compressor_to_cond1 = Arc(
        source=m.fs.compressor.outlet,
        destination=m.fs.condenser[1].inlet,
        doc="Connect compressor outlet to condenser 1",
    )


def add_enrtl_method(m, n_evap=None):
    """This function includes the refined eNRTL method using a Generic
    Parameter Block and connects the state variables to each
    Evaporator unit in the multi-effect MVC

    """

    sb_enrtl = m.fs.enrtl_state[n_evap].properties[0]  # just renaming the block

    # Populate eNRTL state block
    populate_enrtl_state_vars(sb_enrtl, base="FpcTP")

    # Calculate the total mass of the solute molecule and the mass
    # ratio of each ion in the solute molecule.
    m.fs.enrtl_state[n_evap].mol_mass_ion_molecule = sum(
        m.fs.ion_coeff[j] * sb_enrtl.mw_comp[j] for j in m.fs.set_ions
    )
    m.fs.enrtl_state[n_evap].mass_ratio_ion = {
        "Na+": sb_enrtl.mw_comp["Na+"] / m.fs.enrtl_state[n_evap].mol_mass_ion_molecule,
        "Cl-": sb_enrtl.mw_comp["Cl-"] / m.fs.enrtl_state[n_evap].mol_mass_ion_molecule,
    }

    # Add constraints to link the outlet temperature, pressure, and
    # mass flowrate of the evaporator brine with the eNRTL properties
    # block. Note that, since the flow from the seawater property
    # package is in terms of total TDS, we convert the flow from TDS
    # to the respective ions in the seawater.
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
        m.fs.set_ions, rule=enrtl_flow_mass_ion_comp
    )


def set_scaling(m):
    """This function sets the scaling factors for relevant variables in
    all units in the flowsheet

    """

    m.fs.properties_feed.set_default_scaling(
        "flow_mass_phase_comp", 1, index=("Liq", "H2O")
    )
    m.fs.properties_feed.set_default_scaling(
        "flow_mass_phase_comp", 1e2, index=("Liq", "TDS")
    )
    m.fs.properties_vapor.set_default_scaling(
        "flow_mass_phase_comp", 1, index=("Vap", "H2O")
    )
    m.fs.properties_vapor.set_default_scaling(
        "flow_mass_phase_comp", 1, index=("Liq", "H2O")
    )

    # Evaporators
    for e in m.fs.set_evaporators:
        iscale.set_scaling_factor(m.fs.evaporator[e].area, 1e-3)
        iscale.set_scaling_factor(m.fs.evaporator[e].U, 1e-3)
        iscale.set_scaling_factor(m.fs.evaporator[e].delta_temperature_in, 1e-1)
        iscale.set_scaling_factor(m.fs.evaporator[e].delta_temperature_out, 1e-1)
        iscale.set_scaling_factor(m.fs.evaporator[e].lmtd, 1e-1)
        iscale.set_scaling_factor(m.fs.evaporator[e].heat_transfer, 1e-6)
        # Condenser
        iscale.set_scaling_factor(m.fs.condenser[e].control_volume.heat, 1e-6)

    # Compressor
    iscale.set_scaling_factor(m.fs.compressor.control_volume.work, 1e-6)

    # Calculate scaling factors
    iscale.calculate_scaling_factors(m)


def set_model_inputs(m):
    """This function sets model inputs"""

    # Fix inlet data for evaporator 1
    m.fs.evaporator[1].inlet_feed.flow_mass_phase_comp[0, "Liq", "H2O"].fix(
        9.65
    )  # kg/s
    m.fs.evaporator[1].inlet_feed.flow_mass_phase_comp[0, "Liq", "TDS"].fix(
        0.35
    )  # kg/s
    m.fs.evaporator[1].inlet_feed.temperature[0].fix(50.52 + 273.15)  # K
    m.fs.evaporator[1].inlet_feed.pressure[0].fix(101325)  # Pa
    m.fs.evaporator[1].outlet_brine.temperature[0].fix(54 + 273.15)  # K
    m.fs.evaporator[1].U.fix(1e3)  # W/K-m^2
    m.fs.evaporator[1].area.fix(800)  # m^2
    m.fs.evaporator[1].delta_temperature_in.fix(10)
    m.fs.evaporator[1].delta_temperature_out.fix(5)

    # Fix design data for evaporator 2
    m.fs.evaporator[2].U.fix(1e3)
    m.fs.evaporator[2].area.fix(500)
    m.fs.evaporator[2].outlet_brine.temperature[0].fix(55 + 273.15)  # K
    m.fs.evaporator[2].delta_temperature_in.fix(10)
    m.fs.evaporator[2].delta_temperature_out.fix(5)

    # Fix design data for evaporator 3
    m.fs.evaporator[3].U.fix(1e3)
    m.fs.evaporator[3].area.fix(500)
    m.fs.evaporator[3].outlet_brine.temperature[0].fix(60 + 273.15)  # K
    m.fs.evaporator[3].delta_temperature_in.fix(10)
    m.fs.evaporator[3].delta_temperature_out.fix(5)

    # Fix compressor data
    m.fs.compressor.pressure_ratio.fix(1.8)
    m.fs.compressor.efficiency.fix(0.8)

    # Fix condenser data
    m.fs.condenser[1].control_volume.heat[0].fix(-3605623)
    m.fs.condenser[2].control_volume.heat[0].fix(-5768998)
    m.fs.condenser[3].control_volume.heat[0].fix(-3605623)


def initialize(m, solver=None, outlvl=idaeslog.NOTSET):
    """This function initializes the flowsheet as a square problem"""

    # Initialize evaporator 1
    m.fs.evaporator[1].initialize(outlvl=outlvl)

    # Initialize condenser 2
    propagate_state(m.fs.evap1vapor_to_cond2)
    m.fs.condenser[2].initialize_build(heat=-m.fs.evaporator[1].heat_transfer.value)

    # Initialize evaporator 2
    propagate_state(m.fs.evap1brine_to_evap2feed)
    m.fs.evaporator[2].initialize(outlvl=outlvl)

    # Initialize condenser 3
    propagate_state(m.fs.evap2vapor_to_cond3)
    m.fs.condenser[3].initialize_build(heat=-m.fs.evaporator[2].heat_transfer.value)

    # Initialize evaporator 3
    propagate_state(m.fs.evap2brine_to_evap3feed)
    m.fs.evaporator[3].initialize(outlvl=outlvl)

    # Initialize compressor
    propagate_state(m.fs.evap3vapor_to_compressor)
    m.fs.compressor.initialize(outlvl=outlvl)

    # Initialize condenser 1
    propagate_state(m.fs.compressor_to_cond1)
    m.fs.condenser[1].initialize_build(heat=-m.fs.evaporator[3].heat_transfer.value)

    print()
    print("****** Start initialization")
    if not degrees_of_freedom(m) == 0:
        raise ConfigurationError(
            f"You have {degrees_of_freedom(m)} degree(s) of freedom. "
            "Please, check your inputs to ensure a square problem  "
            "(or zero degrees of freedom) to initialize the model."
        )
    init_results = solver.solve(m, tee=False)
    assert (
        init_results.solver.termination_condition == "optimal"
    ), f"The solver termination is not optimal during initialization. Please, check the model inputs."
    print("****** End initialization")
    print()


def add_bounds(m):
    """This function adds bounds to variables in the flowsheet"""

    for i in m.fs.set_evaporators:
        m.fs.evaporator[i].area.setlb(10)
        m.fs.evaporator[i].area.setub(None)
        m.fs.evaporator[i].outlet_brine.temperature[0].setub(73 + 273.15)
        # Bounds from ref[1]
        m.fs.evaporator[i].U.setlb(500)
        m.fs.evaporator[i].U.setub(2500)


def print_results(m):
    m.fs.compressor.report()

    for i in m.fs.set_evaporators:
        m.fs.condenser[i].report()

    for i in m.fs.set_evaporators:
        # Material properties of feed, brine outlet, and vapor outlet
        sw_blk = m.fs.evaporator[i].properties_feed[0]
        brine_blk = m.fs.evaporator[i].properties_brine[0]
        vapor_blk = m.fs.evaporator[i].properties_vapor[0]
        print()
        print()
        print(
            "===================================================================================="
        )
        print("Unit : m.fs.evaporator[{}] (non-ideal)".format(i))
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
        print(
            "             act_coeff* H2O : {:>4.4f} (log:{:>4.4f})".format(
                value(
                    m.fs.enrtl_state[i].properties[0].act_coeff_phase_comp["Liq", "H2O"]
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
        for j in m.fs.set_ions:
            print(
                "            act_coeff* {} : {:>4.4f} (log:{:>4.4f})".format(
                    j,
                    value(
                        m.fs.enrtl_state[i].properties[0].act_coeff_phase_comp["Liq", j]
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
            "       mean_ion_actv_coeff* : {:>4.4f} (log: {:>4.4f})".format(
                exp(
                    value(m.fs.enrtl_state[i].properties[0].Liq_log_gamma_appr["NaCl"])
                ),
                value(m.fs.enrtl_state[i].properties[0].Liq_log_gamma_appr["NaCl"]),
            )
        )
        print(
            "molal mean_ionic_actv_coeff*: {:>4.4f} (log: {:>4.4f})".format(
                exp(value(m.fs.enrtl_state[i].properties[0].Liq_log_gamma_molal)),
                value(m.fs.enrtl_state[i].properties[0].Liq_log_gamma_molal),
            )
        )
        print("    *calculated with refined-eNRTL")
        print()
    print(
        "===================================================================================="
    )
    print("Other variables:")
    print(
        "------------------------------------------------------------------------------------"
    )
    print()
    print(
        " Energy input (kW): {:0.4f}".format(
            value(m.fs.compressor.control_volume.work[0] * (1e-3))
        )
    )
    print(
        " Total water produced (gal/min): {:>0.4f}".format(
            value(m.fs.total_water_produced_gpm)
        )
    )
    print(
        " Specific energy consumption (SC, kWh/m3): {:>0.4f}".format(
            value(m.fs.specific_energy_consumption)
        )
    )
    print(" Water recovery (%): {:>0.4f}".format(value(m.fs.water_recovery) * 100))
    for i in m.fs.set_evaporators:
        print(
            " Molal conc solute evap {} (mol/kg): {:>0.4f}".format(
                i, value(m.fs.molal_conc_solute[i])
            )
        )
    print()
    print()


def unfix_dofs(m):
    """This function unfixes the degrees of freedom that were fixed during
    initialization

    """

    for i in m.fs.set_evaporators:
        m.fs.evaporator[i].area.unfix()
        m.fs.evaporator[i].U.unfix()
        m.fs.evaporator[i].outlet_brine.temperature[0].unfix()
        m.fs.evaporator[i].delta_temperature_in.unfix()
        m.fs.evaporator[i].delta_temperature_out.unfix()
        m.fs.condenser[i].control_volume.heat[0].unfix()

    m.fs.compressor.control_volume.work.unfix()
    m.fs.compressor.pressure_ratio.unfix()


def model_analysis(m, water_rec=None):
    # Unfix degrees of freedom that were fixed during initialization
    unfix_dofs(m)

    # Add upper bound to the compressor pressure ratio. Included for
    # now during debugging stage to check behavior of compressor
    @m.fs.Constraint()
    def eq_upper_bound_compression_ratio(b):
        return b.compressor.pressure_ratio <= 2 * pyunits.dimensionless

    # Add constraints to add an upper bound on the evaporator delta temperatures
    # delta_temperature_in = condenser inlet temp - evaporator brine temp
    # delta_temperature_out = condenser outlet temp - evaporator brine temp
    @m.fs.Constraint(m.fs.set_evaporators)
    def eq_upper_bound_evaporators_delta_temperature_in(b, e):
        return b.evaporator[e].delta_temperature_in == 3 * pyunits.K

    @m.fs.Constraint(m.fs.set_evaporators)
    def eq_upper_bound_evaporators_delta_temprature_out(b, e):
        return b.evaporator[e].delta_temperature_out <= 10 * pyunits.K

    # Add constraint to make sure the pressure in the evaporators 2
    # and 3 is smaller than the pressure in evaporator 1. Included for
    # debugging purposes
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

        # Calculate total water produced and total specific energy

    # consumption.
    m.fs.water_density = pyo.Param(initialize=1000, units=pyunits.kg / pyunits.m**3)

    @m.fs.Expression()
    def total_water_produced_gpm(b):
        return pyo.units.convert(
            sum(
                b.condenser[e]
                .control_volume.properties_out[0]
                .flow_mass_phase_comp["Liq", "H2O"]
                for e in m.fs.set_evaporators
            )
            / m.fs.water_density,
            to_units=pyunits.gallon / pyunits.minute,
        )

    m.fs.specific_energy_consumption = pyo.Var(
        initialize=11, units=pyunits.kW * pyunits.hour / pyunits.m**3, bounds=(0, 1e3)
    )

    @m.fs.Constraint(doc="Specific energy consumption [kWh/m^3]")
    def eq_specific_energy_consumption(b):
        return b.specific_energy_consumption == (
            pyo.units.convert(b.compressor.control_volume.work[0], to_units=pyunits.kW)
            / pyo.units.convert(
                m.fs.total_water_produced_gpm, to_units=pyunits.m**3 / pyunits.hour
            )
        )

    m.fs.water_recovery = pyo.Var(
        initialize=0.2, bounds=(0, 1), units=pyunits.dimensionless, doc="Water recovery"
    )

    @m.fs.Constraint()
    def rule_water_recovery(b):
        return m.fs.water_recovery == (
            1
            - (
                m.fs.evaporator[1].inlet_feed.flow_mass_phase_comp[0, "Liq", "H2O"]
                - sum(
                    b.condenser[e]
                    .control_volume.properties_out[0]
                    .flow_mass_phase_comp["Liq", "H2O"]
                    for e in m.fs.set_evaporators
                )
            )
            / m.fs.evaporator[1].inlet_feed.flow_mass_phase_comp[0, "Liq", "H2O"]
        )

    @m.fs.Constraint()
    def fix_water_recovery(b):
        return b.water_recovery == water_rec

    m.obj = pyo.Objective(expr=1, sense=minimize)


def plot_results(m):
    plt.figure(1)
    plt.grid(linestyle=":", which="both", color="gray", alpha=0.50)
    plt.plot(
        water_recovery_perc,
        sc,
        marker="o",
        ms=8,
        color="tab:red",
        alpha=0.7,
        ls="-",
        lw=4,
        label="r-eNRTL",
    )
    plt.xlabel("Water Recovery (%)")
    plt.ylabel("Specific Energy Consumption (kW/m3/h)")
    plt.legend(frameon=False)
    plt.show()


if __name__ == "__main__":
    optarg = {"max_iter": 300, "tol": 1e-8}
    solver = get_solver("ipopt", optarg)

    # Add lists to plot results
    water_recovery_perc = []
    sc = []
    area_evap1 = []
    area_evap2 = []
    area_evap3 = []

    # Add water recovery data
    water_recovery_data = [0.5, 0.6, 0.7]
    for c in range(len(water_recovery_data)):
        m = create_model()

        set_scaling(m)

        set_model_inputs(m)

        initialize(m, solver=solver)

        add_bounds(m)

        model_analysis(m, water_rec=water_recovery_data[c])

        results = solver.solve(m, tee=True)

        print_results(m)

        # Save results in previously declared lists
        water_recovery_perc.append(water_recovery_data[c] * 100)
        sc.append(value(m.fs.specific_energy_consumption))
        area_evap1.append(value(m.fs.evaporator[1].area))
        area_evap2.append(value(m.fs.evaporator[2].area))
        area_evap3.append(value(m.fs.evaporator[3].area))

    # Plot sc vs water recovery
    if len(water_recovery_data) >= 2:
        plot_results(m)
