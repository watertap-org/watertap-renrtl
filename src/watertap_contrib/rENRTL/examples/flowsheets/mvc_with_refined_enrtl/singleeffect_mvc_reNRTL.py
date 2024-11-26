###############################################################################
# WaterTAP Copyright (c) 2020-2023, The Regents of the University of California,
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

# Import Pyomo components
import pyomo.environ as pyo
from pyomo.network import Arc
from pyomo.environ import units as pyunits
from pyomo.environ import (
    ConcreteModel,
    TransformationFactory,
    Block,
    Constraint,
    Objective,
    minimize,
    Param,
    value,
    Set,
    log,
    exp,
)

# Import IDAES libraries
import idaes.core.util.scaling as iscale
import idaes.logger as idaeslog
from idaes.core import FlowsheetBlock
from idaes.models.properties.modular_properties.base.generic_property import (
    GenericParameterBlock,
)
from idaes.core.solvers.get_solver import get_solver
from idaes.core.util.model_statistics import degrees_of_freedom
from idaes.core.util.initialization import propagate_state
from idaes.core.util.exceptions import ConfigurationError

# Import WaterTAP libraries
from watertap.unit_models.mvc.components import Evaporator, Compressor, Condenser

# Import property packages from WaterTAP
import watertap.property_models.seawater_prop_pack as props_sw
import watertap.property_models.water_prop_pack as props_w

# Import refined eNRTL configuration dictionary
import entrl_config_FpcTP

# Set logging options
logging.basicConfig(level=logging.INFO)
logging.getLogger("pyomo.repn.plugins.nl_writer").setLevel(logging.ERROR)


"""

References:

[1] Matthew D. Stuber, Christopher Sullivan, Spencer A. Kirk, Jennifer
A. Farrand, Philip V. Schillai, Brian D. Fojtasek, and Aaron
H. Mandell. Pilot demonstration of concentrated solar-poweered
desalination of subsurface agricultural drainage water and other
brackish groundwater sources. Desalination, 355 (2015), 186-196.

"""


def populate_enrtl_state_vars(blk, base="FpcTP"):
    blk.temperature = 298.15
    blk.pressure = 101325

    if base == "FpcTP":
        feed_flow_mass = 10  # kg/s
        feed_mass_frac_comp = {
            "Na+": 0.013768116,
            "Cl-": 0.021231884,
        }
        feed_mass_frac_comp["H2O"] = 1 - sum(x for x in feed_mass_frac_comp.values())

        mw_comp = {
            "H2O": 18.015e-3,
            "Na+": 22.990e-3,
            "Cl-": 35.453e-3,
        }

        for j in feed_mass_frac_comp:
            blk.flow_mol_phase_comp["Liq", j] = (
                feed_flow_mass * feed_mass_frac_comp[j] / mw_comp[j]
            )
            if j == "H2O":
                blk.flow_mol_phase_comp["Liq", j] /= 2


def create_model():
    m = ConcreteModel("Single Effect Mechanical Vapor Compression Model")

    m.fs = FlowsheetBlock(dynamic=False)

    # Add property packages for water and seawater.
    m.fs.properties_feed = props_sw.SeawaterParameterBlock()
    m.fs.properties_vapor = props_w.WaterParameterBlock()

    # Declare the evaporator, and compressor, and condenser units
    m.fs.evaporator = Evaporator(
        property_package_feed=m.fs.properties_feed,
        property_package_vapor=m.fs.properties_vapor,
    )
    m.fs.compressor = Compressor(property_package=m.fs.properties_vapor)
    m.fs.condenser = Condenser(property_package=m.fs.properties_vapor)

    # Add variable to calculate molal concentration of solute. The
    # upper bound is included to ensure that the molality of the
    # electrolyte solution is within saturation and the concentration
    # limits of eNRTL tau and alpha parameters.
    m.fs.molal_conc_solute = pyo.Var(
        initialize=2,
        bounds=(0, 6),
        units=pyunits.mol / pyunits.kg,
        doc="Molal concentration of solute",
    )

    @m.fs.Constraint(
        doc="Molal concentration of solute in solvent [mol of TDS/kg of H2O]"
    )
    def rule_molal_conc_solute(b):
        return m.fs.molal_conc_solute == (
            (
                b.evaporator.properties_brine[0].flow_mass_phase_comp["Liq", "TDS"]
                / b.properties_feed.mw_comp["TDS"]  # to convert it to mol/s
            )
            / b.evaporator.properties_brine[0].flow_mass_phase_comp["Liq", "H2O"]
        )

    # Add variable for activity coefficient
    m.fs.act_coeff = pyo.Var(initialize=1, bounds=(1e-3, 10))

    # Add eNRTL method that calculates the activity coefficients for
    # the electrolyte solution
    add_enrtl_method(m)

    # Add constraint to save non-ideal activity coefficient for water
    # since it will be used in the equilibrium equation
    @m.fs.Constraint(doc="Nonideal activity coefficient of water")
    def _eq_nonideal_act_coefficient(b):
        return (
            b.act_coeff
            == b.enrtl_state.properties[0].act_coeff_phase_comp["Liq", "H2O"]
        )

    # Deactivate equilibrium equation from evaporator and include a
    # new equilibrium equation as a Constraint that includes the
    # activity coefficient. Note that since water is the only solvent
    # participating in the vapor-liquid equilibrium, the activity
    # coefficient and vapor pressure are of water as the
    # solvent.
    m.fs.evaporator.eq_brine_pressure.deactivate()

    @m.fs.Constraint(doc="Vapor-liquid equilibrium equation")
    def _eq_phase_equilibrium(b):
        return b.evaporator.properties_brine[0].pressure == (
            m.fs.act_coeff
            * b.evaporator.properties_brine[0].mole_frac_phase_comp["Liq", "H2O"]
            * b.evaporator.properties_vapor[0].pressure_sat
        )

    # Create and expand Arcs to create the model constraints necessary to connect
    # the units in the flowsheet
    create_arcs(m)

    TransformationFactory("network.expand_arcs").apply_to(m)
    m.fs.evaporator.connect_to_condenser(m.fs.condenser)

    return m


def create_arcs(m):
    """This method creates arcs that connect all units in the flowsheet"""

    # Declare Arcs to connect the units in the flowsheet.
    m.fs.s01 = Arc(
        source=m.fs.evaporator.outlet_vapor, destination=m.fs.compressor.inlet
    )
    m.fs.s02 = Arc(source=m.fs.compressor.outlet, destination=m.fs.condenser.inlet)


def add_enrtl_method(m):
    """This function includes the refined eNRTL method in a Generic
    Parameter Block and connects the state variables to the Evaporator
    unit in the MVC

    """

    # Declare a Generic Parameter Block that calls a configuration
    # file that includes eNRTL as the equation of state method
    m.fs.prop_enrtl = GenericParameterBlock(**entrl_config_FpcTP.configuration)

    # Declare a block to include the generic properties needed by
    # eNRTL as a state block
    m.fs.enrtl_state = Block()
    m.fs.enrtl_state.properties = m.fs.prop_enrtl.build_state_block([0])
    sb_enrtl = m.fs.enrtl_state.properties[0]  # just renaming the block

    # Populate eNRTL state block
    populate_enrtl_state_vars(sb_enrtl, base="FpcTP")

    # Declare a set for the ions in the electrolyte solution and the
    # stoichiometric coefficient of the ions in the solute molecule
    # and calculate the total mass of the solute molecule and the mass
    # ratio of each ion in the solute molecule
    set_ions = ["Na+", "Cl-"]
    m.fs.set_ions = Set(initialize=set_ions)
    m.fs.ion_coeff = {"Na+": 1, "Cl-": 1}
    m.fs.enrtl_state.mol_mass_ion_molecule = sum(
        m.fs.ion_coeff[j] * sb_enrtl.mw_comp[j] for j in m.fs.set_ions
    )
    m.fs.enrtl_state.mass_ratio_ion = {
        "Na+": sb_enrtl.mw_comp["Na+"] / m.fs.enrtl_state.mol_mass_ion_molecule,
        "Cl-": sb_enrtl.mw_comp["Cl-"] / m.fs.enrtl_state.mol_mass_ion_molecule,
    }

    # Add constraints to link the outlet temperature, pressure, and
    # mass flowrate of the evaporator brine with the eNRTL properties
    # block. Note that, since the flow from the seawater property
    # package is in terms of total TDS, we convert the flow from TDS
    # to the respective ions in the seawater
    @m.fs.enrtl_state.Constraint()
    def eq_enrtl_temperature(b):
        return (
            b.properties[0].temperature
            == m.fs.evaporator.properties_brine[0].temperature
        )

    @m.fs.enrtl_state.Constraint()
    def eq_enrtl_pressure(b):
        return b.properties[0].pressure == m.fs.evaporator.properties_brine[0].pressure

    @m.fs.enrtl_state.Constraint()
    def eq_enrtl_flow_mass_H2O(b):
        return (
            b.properties[0].flow_mass_phase_comp["Liq", "H2O"]
            == m.fs.evaporator.properties_brine[0].flow_mass_phase_comp["Liq", "H2O"]
        )

    @m.fs.enrtl_state.Constraint(m.fs.set_ions)
    def eq_enrtl_flow_mass_ion_comp(b, j):
        return b.properties[0].flow_mass_phase_comp["Liq", j] == (
            (
                m.fs.evaporator.properties_brine[0].flow_mass_phase_comp["Liq", "TDS"]
                * b.mass_ratio_ion[j]
            )
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

    # Evaporator
    iscale.set_scaling_factor(m.fs.evaporator.area, 1e-3)
    iscale.set_scaling_factor(m.fs.evaporator.U, 1e-3)
    iscale.set_scaling_factor(m.fs.evaporator.delta_temperature_in, 1e-1)
    iscale.set_scaling_factor(m.fs.evaporator.delta_temperature_out, 1e-1)
    iscale.set_scaling_factor(m.fs.evaporator.lmtd, 1e-1)

    # Compressor
    iscale.set_scaling_factor(m.fs.compressor.control_volume.work, 1e-6)

    # Condenser
    iscale.set_scaling_factor(m.fs.condenser.control_volume.heat, 1e-6)

    # Calculate scaling factors
    iscale.calculate_scaling_factors(m)


def set_model_inputs(m):
    """This function sets the model inputs"""

    # Inlet data
    m.fs.evaporator.inlet_feed.flow_mass_phase_comp[0, "Liq", "H2O"].fix(9.65)
    m.fs.evaporator.inlet_feed.flow_mass_phase_comp[0, "Liq", "TDS"].fix(0.35)
    m.fs.evaporator.inlet_feed.temperature[0].fix(50.52 + 273.15)  # K
    m.fs.evaporator.inlet_feed.pressure[0].fix(101325)  # Pa
    m.fs.evaporator.outlet_brine.temperature[0].fix(273.15 + 54)  # K
    m.fs.evaporator.U.fix(1e3)  # W/K-m^2
    m.fs.evaporator.area.fix(1000)  # m^2

    # Compressor
    m.fs.compressor.pressure_ratio = 2
    m.fs.compressor.control_volume.work.fix(2e5)
    m.fs.compressor.efficiency.fix(0.8)


def initialize(m, solver=None, outlvl=idaeslog.NOTSET):
    """This function initializes the flowsheet as a square problem"""

    if solver is None:
        solver = get_solver()
    optarg = solver.options

    # Initialize evaporator
    m.fs.evaporator.initialize_build(
        delta_temperature_in=30, delta_temperature_out=5, outlvl=outlvl
    )

    # Initialize compressor
    propagate_state(m.fs.s01)
    m.fs.compressor.initialize(outlvl=outlvl)

    # Initialize condenser
    propagate_state(m.fs.s02)
    m.fs.condenser.initialize_build(heat=-m.fs.evaporator.heat_transfer.value)

    print()
    print("****** Start initialization")
    if degrees_of_freedom(m) != 0:
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


def print_results(m):
    m.fs.molal_conc_solute_feed = (
        value(m.fs.evaporator.inlet_feed.flow_mass_phase_comp[0, "Liq", "TDS"])
        / value(m.fs.properties_feed.mw_comp["TDS"])
    ) / value(m.fs.evaporator.inlet_feed.flow_mass_phase_comp[0, "Liq", "H2O"])

    m.fs.compressor.report()
    m.fs.condenser.report()

    # Print table for evaporator when calling the function
    # print_evap_table below
    print_evap_table(m)


def print_evap_table(m):
    # Material properties of evaporator feed and brine and vapor
    # outlets
    sw_blk = m.fs.evaporator.properties_feed[0]
    brine_blk = m.fs.evaporator.properties_brine[0]
    vapor_blk = m.fs.evaporator.properties_vapor[0]

    print()
    print()
    print(
        "===================================================================================="
    )
    print("Unit : m.fs.evaporator (non-ideal)")
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
            value(m.fs.evaporator.delta_temperature_in)
        )
    )
    print("      delta temperature_out : {:>4.3f}".format(value(m.fs.evaporator.lmtd)))
    print("                       lmtd : {:>4.3f}".format(value(m.fs.evaporator.lmtd)))
    print("                      Area  : {:>4.3f}".format(value(m.fs.evaporator.area)))
    print("                         U  : {:>4.3f}".format(value(m.fs.evaporator.U)))
    print("                   UA_term  : {:>4.3f}".format(value(m.fs.UA_term)))
    print(
        "             heat_transfer  : {:>4.3f}".format(
            value(m.fs.evaporator.heat_transfer)
        )
    )  # in W
    enrtl_prop = m.fs.enrtl_state.properties[0]
    print(
        "             act_coeff* H2O : {:>4.4f} (log:{:>4.4f})".format(
            value(enrtl_prop.act_coeff_phase_comp["Liq", "H2O"]),
            value(log(enrtl_prop.act_coeff_phase_comp["Liq", "H2O"])),
        )
    )
    for j in m.fs.set_ions:
        print(
            "             act_coeff* {} : {:>4.4f} (log:{:>4.4f})".format(
                j,
                value(enrtl_prop.act_coeff_phase_comp["Liq", j]),
                value(log(enrtl_prop.act_coeff_phase_comp["Liq", j])),
            )
        )
    print(
        "       mean_ion_actv_coeff* : {:>4.4f} (log: {:>4.4f})".format(
            exp(value(enrtl_prop.Liq_log_gamma_appr["NaCl"])),
            value(enrtl_prop.Liq_log_gamma_appr["NaCl"]),
        )
    )
    print(
        "molal mean_ionic_actv_coeff*: {:>4.4f} (log: {:>4.4f})".format(
            exp(value(enrtl_prop.Liq_log_gamma_molal)),
            value(enrtl_prop.Liq_log_gamma_molal),
        )
    )
    print("       *calculated with reNRTL")
    print()
    print()
    print(
        "===================================================================================="
    )
    print("Other variables:")
    print(
        "------------------------------------------------------------------------------------"
    )

    print(
        " Energy input (kW): {:0.4f}".format(
            value(m.fs.compressor.control_volume.work[0]) * (1e-3)
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
    print(" Molal conc solute (mol/kg): {:>0.4f}".format(value(m.fs.molal_conc_solute)))


def model_analysis(m):
    # Unfix data that was fixed during initialization
    m.fs.evaporator.area.unfix()
    m.fs.evaporator.U.unfix()
    m.fs.compressor.control_volume.work.unfix()
    m.fs.evaporator.outlet_brine.temperature[0].unfix()

    # Add upper bound to the compressor pressure ratio. Included for
    # now during debugging stage to check behavior of compressor
    @m.fs.Constraint()
    def eq_upper_bound_compression_ratio(b):
        return b.compressor.pressure_ratio <= 2 * pyunits.dimensionless

    # Add constraints to add an upper bound on the evaporator delta temperatures
    # delta_temperature_in = condenser inlet temp - evaporator brine temp
    # delta_temperature_out = condenser outlet temp - evaporator brine temp
    @m.fs.Constraint()
    def eq_upper_bound_evaporators_delta_temperature_in(b):
        return b.evaporator.delta_temperature_in == 3 * pyunits.K

    @m.fs.Constraint()
    def eq_upper_bound_evaporators_delta_temprature_out(b):
        return b.evaporator.delta_temperature_out <= 10 * pyunits.K

    # Add expression to calculate the UA term
    @m.fs.Expression(doc="Overall heat trasfer coefficient and area term")
    def UA_term(b):
        return b.evaporator.area * b.evaporator.U

    # Add constraints for lower and upper bound of overall heat
    # transfer coefficient. Bounds based on figure 7 in ref[1]
    @m.fs.Constraint(doc="Overall heat trasfer coefficient lower bound")
    def U_lower_bound(b):
        return b.evaporator.U >= 500 * (
            pyunits.joule / pyunits.K / pyunits.m**2 / pyunits.second
        )

    @m.fs.Constraint(doc="Overall heat trasfer coefficient upper bound")
    def U_upper_bound(b):
        return b.evaporator.U <= 2500 * (
            pyunits.joule / pyunits.K / pyunits.m**2 / pyunits.second
        )

    # The temperature of the brine should not be higher than 73C to
    # avoid damaging the equipment
    @m.fs.Constraint(doc="Upper bound of evaporator temperature")
    def temperature_upper_bound(b):
        return (
            b.evaporator.outlet_brine.temperature[0] <= (73 + 273.15) * pyunits.K
        )  # in K

    # Calculate total water produced and total specific energy
    # consumption.
    m.fs.water_density = pyo.Param(initialize=1000, units=pyunits.kg / pyunits.m**3)

    @m.fs.Expression()
    def total_water_produced_gpm(b):
        return pyo.units.convert(
            b.condenser.control_volume.properties_out[0].flow_mass_phase_comp[
                "Liq", "H2O"
            ]
            / b.water_density,
            to_units=pyunits.gallon / pyunits.minute,
        )

    m.fs.specific_energy_consumption = pyo.Var(
        initialize=2, units=pyunits.kW * pyunits.hour / pyunits.m**3, bounds=(0, 1e3)
    )

    @m.fs.Constraint(doc="Specific energy consumption in kWh/m^3")
    def rule_specific_energy_consumption(b):
        return b.specific_energy_consumption == (
            pyo.units.convert(b.compressor.control_volume.work[0], to_units=pyunits.kW)
            / pyo.units.convert(
                b.total_water_produced_gpm, to_units=pyunits.m**3 / pyunits.hour
            )
        )

    m.fs.water_recovery = pyo.Var(
        initialize=0.2, bounds=(0, 1), units=pyunits.dimensionless, doc="Water recovery"
    )

    @m.fs.Constraint(doc="Water recovery")
    def rule_water_recovery(b):
        return m.fs.water_recovery == (
            1
            - (
                m.fs.evaporator.inlet_feed.flow_mass_phase_comp[0, "Liq", "H2O"]
                - b.condenser.control_volume.properties_out[0].flow_mass_phase_comp[
                    "Liq", "H2O"
                ]
            )
            / m.fs.evaporator.inlet_feed.flow_mass_phase_comp[0, "Liq", "H2O"]
        )

    @m.fs.Constraint(doc="Set water recovery")
    def rule_fix_water_recovery(b):
        return m.fs.water_recovery == 0.30 * pyunits.dimensionless

    m.obj = pyo.Objective(expr=1, sense=minimize)


if __name__ == "__main__":
    optarg = {"max_iter": 300, "tol": 1e-8}
    solver = get_solver("ipopt", optarg)

    m = create_model()

    set_scaling(m)

    set_model_inputs(m)

    initialize(m, solver=solver)

    model_analysis(m)

    results = solver.solve(m, tee=True)

    print_results(m)
