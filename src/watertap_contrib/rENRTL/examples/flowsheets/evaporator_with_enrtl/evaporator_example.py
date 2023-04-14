###############################################################################
# WaterTAP Copyright (c) 2020-2023, The Regents of the University of California,
# through Lawrence Berkeley National Laboratory, Oak Ridge National Laboratory,
# National Renewable Energy Laboratory, and National Energy Technology
# Laboratory (subject to receipt of any required approvals from the U.S. Dept.
# of Energy). All rights reserved.
#
# Copyright 2023, National Technology & Engineering Solutions of Sandia,
# LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the
# U.S. Government retains certain rights in this software
#
# Copyright 2023, Pengfei Xu and Matthew D. Stuber and the University
# of Connecticut.
#
# Please see the files COPYRIGHT.md and LICENSE.md for full copyright and license
# information, respectively. These files are also available online at the URL
# "https://github.com/watertap-org/watertap/"
#
###############################################################################

"""This is an example for the separation of freshwater from seawater
using the Evaporator unit model from WaterTAP. This unit allows the
use of two property packages within the same unit: one for the feed
and bottom product (as liquid phase) and one for the top product (as
vapor phase). The top product is considered to be pure water while the
bottom product is a concentrated brine.

Author: Soraya Rawlings

"""
import logging

# Import Pyomo components
import pyomo.environ as pyo
from pyomo.environ import units as pyunits
from pyomo.environ import (
    ConcreteModel,
    TransformationFactory,
    Block,
    Constraint,
    minimize,
    Param,
    value,
    Set,
)

# Import IDAES libraries
import idaes.core.util.scaling as iscale
import idaes.logger as idaeslog
from idaes.core import FlowsheetBlock
from idaes.models.properties.modular_properties.base.generic_property import (
    GenericParameterBlock,
)
from idaes.core.solvers import get_solver
from idaes.core.util.model_statistics import degrees_of_freedom
from idaes.core.util.exceptions import ConfigurationError

# Import evaporator model and property packages from WaterTap
from watertap.unit_models.mvc.components import Evaporator
import watertap.property_models.seawater_prop_pack as props_sw
import watertap.property_models.water_prop_pack as props_w

# Import eNRTL configuration dictionary
import entrl_config_FpcTP

logging.getLogger("pyomo.repn.plugins.nl_writer").setLevel(logging.ERROR)

""" References:

[1] Matthew D. Stuber, Christopher Sullivan, Spencer A. Kirk, Jennifer
A. Farrand, Philip V. Schillai, Brian D. Fojtasek, and Aaron
H. Mandell. Pilot demonstration of concentrated solar-poweered
desalination of subsurface agricultural drainage water and other
brackish groundwater sources. Desalination, 355 (2015), 186-196.

"""


def create_model():
    m = ConcreteModel("Evaporator Model")

    m.fs = FlowsheetBlock(dynamic=False)

    # Add property packages for water and seawater
    m.fs.properties_vapor = props_w.WaterParameterBlock()
    m.fs.properties_feed = props_sw.SeawaterParameterBlock()

    # Declare evaporator unit and the property packages for each
    # outlet
    m.fs.evaporator = Evaporator(
        property_package_feed=m.fs.properties_feed,
        property_package_vapor=m.fs.properties_vapor,
    )

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

    @m.fs.Constraint(doc="Molal concentration of solute in mol of TDS/kg of H2O")
    def rule_molal_conc_solute(b):
        return b.molal_conc_solute == (
            (
                b.evaporator.properties_brine[0].flow_mass_phase_comp["Liq", "TDS"]
                / b.properties_feed.mw_comp["TDS"]  # to convert it to mol/s
            )
            / b.evaporator.properties_brine[0].flow_mass_phase_comp["Liq", "H2O"]
        )

    return m


def add_equilibrium_equation(m, solve_ideal=None):

    if solve_ideal:
        # For an ideal system, add the activity coefficient as a
        # parameter with a value of 1
        m.fs.act_coeff = pyo.Param(
            initialize=1,
            units=pyunits.dimensionless,
            doc="Ideal activity coefficient for water",
        )
    else:
        # For a non-ideal system, add activity coefficient as a
        # variable
        m.fs.act_coeff = pyo.Var(
            initialize=1,
            bounds=(1e-3, 10),
            units=pyunits.dimensionless,
            doc="Ideal activity coefficient for water",
        )

        # For non-ideal system, add eNRTL method to calculate water
        # activity coefficient
        add_enrtl_method(m)

        # Add equality constraint to save non-ideal activity
        # coefficient value calculated using the eNRTL method
        @m.fs.Constraint(doc="Nonideal activity coefficient of water")
        def _eq_nonideal_act_coefficient(b):
            return (
                b.act_coeff
                == b.enrtl_state.properties[0].act_coeff_phase_comp["Liq", "H2O"]
            )

    # Deactivate equilibrium equation from evaporator unit since we
    # include an equilibrium constraint that includes nonideality
    # (activity coefficients)
    m.fs.evaporator.eq_brine_pressure.deactivate()

    # Add new equilibrium equation
    @m.fs.Constraint(doc="Vapor-liquid equilibrium equation")
    def _eq_phase_equilibrium(b):
        return b.evaporator.properties_brine[0].pressure == (
            m.fs.act_coeff
            * b.evaporator.properties_brine[0].mole_frac_phase_comp["Liq", "H2O"]
            * b.evaporator.properties_vapor[0].pressure_sat
        )


def populate_enrtl_state_vars(blk, base="FpcTP"):
    blk.temperature = 298.15
    blk.pressure = 101325

    if base == "FpcTP":
        feed_flow_mass = 10  # kg/s
        feed_mass_frac_comp = {"Na+": 0.013768116, "Cl-": 0.021231884}
        feed_mass_frac_comp["H2O"] = 1 - sum(x for x in feed_mass_frac_comp.values())
        mw_comp = {"H2O": 18.015e-3, "Na+": 22.990e-3, "Cl-": 35.453e-3}

        for j in feed_mass_frac_comp:
            blk.flow_mol_phase_comp["Liq", j] = (
                feed_flow_mass * feed_mass_frac_comp[j] / mw_comp[j]
            )
            if j == "H2O":
                blk.flow_mol_phase_comp["Liq", j] /= 2


def add_enrtl_method(m):

    # Create an instance of the Generic Parameter Block component and
    # provide the eNRTL configuration dictionary
    m.fs.prop_enrtl = GenericParameterBlock(**entrl_config_FpcTP.configuration)

    # Declare a new block to include the generic properties needed by
    # eNRTL as a state block
    m.fs.enrtl_state = Block()
    m.fs.enrtl_state.properties = m.fs.prop_enrtl.build_state_block([0])
    sb_enrtl = m.fs.enrtl_state.properties[0]

    # Populate eNRTL state block
    populate_enrtl_state_vars(sb_enrtl, base="FpcTP")

    # Declare the set of ions in the electrolyte solution and their
    # stoichiometric coefficient in the solute molecule to be used to
    # calculate the total mass of the solute molecule and the mass
    # ratio of each ion in the solute molecule
    m.fs.set_ions = Set(initialize=["Na+", "Cl-"])
    m.fs.ion_coeff = {"Na+": 1, "Cl-": 1}
    m.fs.enrtl_state.mol_mass_ion_molecule = sum(
        m.fs.ion_coeff[j] * sb_enrtl.mw_comp[j] for j in m.fs.set_ions
    )
    m.fs.enrtl_state.mass_ratio_ion = {
        "Na+": sb_enrtl.mw_comp["Na+"] / m.fs.enrtl_state.mol_mass_ion_molecule,
        "Cl-": sb_enrtl.mw_comp["Cl-"] / m.fs.enrtl_state.mol_mass_ion_molecule,
    }

    # Add constraints to link the temperature, pressure, and mass
    # flowrate of the brine in the evaporator with the eNRTL state
    # block. Note that, since the seawater property package is in
    # terms of total dissolved solids (TDS), not individual solute
    # ions as in the eNRTL method, we convert the flow from TDS to the
    # respective ions of the solute in seawater using the mass ratio
    # calculated above
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
        flow_mass_in = b.properties[0].flow_mass_phase_comp["Liq", j]
        return flow_mass_in == (
            (
                m.fs.evaporator.properties_brine[0].flow_mass_phase_comp["Liq", "TDS"]
                * b.mass_ratio_ion[j]
            )
        )


def set_scaling(m):
    """Set scaling factors for variables in property packages and
    evaporator unit

    """

    # Properties in property packages
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
    iscale.set_scaling_factor(m.fs.evaporator.heat_transfer, 1e-6)

    # Calculate scaling factors
    iscale.calculate_scaling_factors(m)


def set_model_inputs(m):
    """Set inputs and design parameters for the evaporator"""

    m.fs.evaporator.inlet_feed.flow_mass_phase_comp[0, "Liq", "H2O"].fix(9.65)  # kg/s
    m.fs.evaporator.inlet_feed.flow_mass_phase_comp[0, "Liq", "TDS"].fix(0.35)  # kg/s
    m.fs.evaporator.inlet_feed.temperature[0].fix(273.15 + 50.52)  # K
    m.fs.evaporator.inlet_feed.pressure[0].fix(101325)  # Pa
    m.fs.evaporator.outlet_brine.temperature[0].fix(273.15 + 60)  # K
    m.fs.evaporator.U.fix(1e3)  # W/K-m^2
    m.fs.evaporator.area.fix(400)  # m^2
    m.fs.evaporator.delta_temperature_in.fix(30)  # K
    m.fs.evaporator.delta_temperature_out.fix(5)  # K


def initialize(m, solver=None, outlvl=idaeslog.WARNING):
    """Initialize the evaporator unit"""

    m.fs.evaporator.initialize(outlvl=outlvl)

    # Raise an error if the degrees of freedom are not 0
    if degrees_of_freedom(m) != 0:
        raise ConfigurationError(
            f"You have {degrees_of_freedom(m)} degree(s) of freedom. "
            "Please, check your inputs to ensure a square problem  "
            "(or zero degrees of freedom) to initialize the model."
        )

    init_results = solver.solve(m, tee=False)
    print()
    print(
        "**Initialization solver termination:",
        init_results.solver.termination_condition,
    )
    print()


def model_analysis(m):

    # Unfix data that was fixed during initialization
    m.fs.evaporator.area.unfix()
    m.fs.evaporator.U.unfix()
    m.fs.evaporator.outlet_brine.temperature[0].unfix()

    # Fix the pressure in the evaporator chamber
    m.fs.evaporator.outlet_brine.pressure[0].fix(30e3)

    # Add expression to calculate a new term UA that represents the
    # product of the evaporator overall heat transfer coefficient and
    # its area
    @m.fs.Expression(doc="Overall heat trasfer coefficient and area term")
    def UA_term(b):
        return b.evaporator.area * b.evaporator.U

    # Add constraints for lower and upper bound of overall heat
    # transfer coefficient. Bounds obtained from ref[1]
    @m.fs.Constraint(doc="Overall heat transfer coefficient lower bound")
    def U_lower_bound(b):
        return b.evaporator.U >= 500

    @m.fs.Constraint(doc="Overall heat transfer coefficient upper bound")
    def U_upper_bound(b):
        return b.evaporator.U <= 2500

    # Add constraint to ensure that the temperature in the evaporator
    # chamber is not higher than 73C so to avoid damages in the
    # equipment
    @m.fs.Constraint(doc="Evaporator temperature upper bound")
    def temperature_upper_bound(b):
        return b.evaporator.outlet_brine.temperature[0] <= (73 + 273.15)


def print_results(m):
    """Display results for evaporator unit and relevant variables"""

    print()
    print("================================================================")
    print("Molal concentration of solute in brine:", value(m.fs.molal_conc_solute))
    print("Water activity coefficient:", value(m.fs.act_coeff))
    print()
    m.fs.evaporator.display()


def build_evaporator_model(solve_ideal=None):
    """Build the evaporator model"""

    # Create a flowsheet, add properties, and evaporator unit model
    m = create_model()

    # Set all required inputs to the model
    set_model_inputs(m)

    # Set scaling factors
    set_scaling(m)

    return m


if __name__ == "__main__":

    # Declare NLP solver and its options
    optarg = {"max_iter": 150}
    solver = get_solver("ipopt", optarg)

    # Declare the thermodynamic assumption to use in the model. If set
    # to True, the model is solved assuming an ideal system and if set
    # to False, the eNRTL method is used to calculate the water
    # activity coefficient
    solve_ideal = False

    # Create evaporator model
    m = build_evaporator_model()

    # Initialize evaporator
    initialize(m, solver=solver)

    # Add new vapor-liquid equilibrium equation
    add_equilibrium_equation(m, solve_ideal=solve_ideal)

    # Fix and unfix variables and add constraints to the evaporator
    # model
    model_analysis(m)

    # Solve simulation example
    results = solver.solve(m, tee=True)

    # Display results
    print_results(m)
