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

'''
References:
[1] Robinson, R. A., & Stokes, R. H. Electrolyte Solutions. Academic Press, New York, 1959. xv + 559 pp.

[2] Stuber, M. D., Sullivan, C., Kirk, S. A., Farrand, J. A., Schillaci, P. V., Fojtasek, B. D., & Mandell, A. H. (2015). 
Pilot demonstration of concentrated solar-powered desalination of subsurface agricultural drainage water and 
other brackish groundwater sources. Desalination, 355, 186-196. https://doi.org/10.1016/j.desal.2014.10.037

This is a close loop 3MED-only model configuration. 
The model uses exprimental conditions from [2] and validates well at a water recovery of 60%.

The following changes need to be made to run specific conditions for 60% water recovery:
1. Ideal
def set_scaling(m):
    if "temperature" in var.name:
                    iscale.set_scaling_factor(var, 1e4) 
                    
2. r-eNRTL(constant)
def set_scaling(m):
    if "temperature" in var.name:
                    iscale.set_scaling_factor(var, 1e-4) 
                    
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
                    iscale.set_scaling_factor(var, 1e-2)                 
'''
import logging

# Import Pyomo components
import pyomo.environ as pyo
from pyomo.environ import (ConcreteModel, TransformationFactory,
                           Block, Constraint, Expression,
                           Objective, minimize, Param,
                           value, Set, RangeSet,
                           log, exp, Var)
from pyomo.network import Arc
from pyomo.environ import units as pyunits

# Import IDAES components
import idaes.core.util.scaling as iscale
import idaes.logger as idaeslog
from idaes.core import FlowsheetBlock
from idaes.models.properties.modular_properties.base.generic_property import (
    GenericParameterBlock
    )
from idaes.models.unit_models import Feed

from idaes.core.solvers.get_solver import get_solver
from idaes.core.util.model_statistics import degrees_of_freedom
from idaes.core.util.initialization import propagate_state
from idaes.models.unit_models import Pump, Heater

# Import property packages and WaterTAP components
import watertap.property_models.seawater_prop_pack as props_sw
import watertap.property_models.water_prop_pack as props_w    

from watertap.unit_models.mvc.components import (Evaporator, Condenser)

logging.basicConfig(level=logging.INFO)
logging.getLogger('pyomo.repn.plugins.nl_writer').setLevel(logging.ERROR)

# solve_nonideal gives the option to solve an ideal and nonideal system
# If solve_nonideal is set to true, eNRTL is used to calculate the activity coefficients of solvent and solutes; 
# when set to False, the model is solved assuming an ideal system with an activity coefficient of 1 for the solvent
solve_nonideal = True

# run_multi toggles between single-component and multi-component electrolyte systems. To use the multi-refined eNRTL, run_multi=True. To use the single refined eNRTL, run_multi=False.
run_multi = True 

if run_multi:
    import renrtl_multi_config #multi electrolytes
else: 
    import enrtl_config_FpcTP #single electrolyte
    

def populate_enrtl_state_vars(blk, base="FpcTP"): 
    """ Initialize state variables
    """
    blk.temperature = 27 + 273.15
    blk.pressure = 101325

    if base == "FpcTP":
        feed_flow_mass = 0.15  # kg/s
        feed_mass_frac_comp = {"Na+": 0.009179, "Cl-": 0.014154} 
        feed_mass_frac_comp["H2O"] = (1 - sum(x for x in feed_mass_frac_comp.values()))
        mw_comp = {"H2O": 18.015e-3, "Na+": 22.990e-3, "Cl-": 35.453e-3}

        for j in feed_mass_frac_comp:
            blk.flow_mol_phase_comp["Liq", j] = (feed_flow_mass*feed_mass_frac_comp[j] /
                                                 mw_comp[j])
            if j == "H2O":
                blk.flow_mol_phase_comp["Liq", j] /= 2
                
def populate_enrtl_state_vars_multi(blk, base="FpcTP"): 
    """ Initialize state variables
    """
    blk.temperature = 27 + 273.15
    blk.pressure = 101325

    if base == "FpcTP":
        feed_flow_mass = 0.15  # kg/s
        feed_mass_frac_comp = {"Na+": 0.003022, "Cl-": 0.004601, "SO4_2-": 0.01263} 
        feed_mass_frac_comp["H2O"] = (1 - sum(x for x in feed_mass_frac_comp.values()))
        mw_comp = {"H2O": 18.015e-3, "Na+": 22.990e-3, "Cl-": 35.453e-3,"SO4_2-": 96.064e-3}

        for j in feed_mass_frac_comp:
            blk.flow_mol_phase_comp["Liq", j] = (feed_flow_mass*feed_mass_frac_comp[j] /
                                                 mw_comp[j])
            if j == "H2O":
                blk.flow_mol_phase_comp["Liq", j] /= 2

def create_model():
    m = ConcreteModel("Three-effect MED")
    m.fs = FlowsheetBlock(dynamic=False)
    
    # Add property packages for water and seawater
    m.fs.properties_vapor = props_w.WaterParameterBlock()
    m.fs.properties_feed = props_sw.SeawaterParameterBlock()

    m.fs.feed = Feed(property_package=m.fs.properties_feed)

    # Declare unit models 
    # Note: the evaporator unit is a customized unit that includes a complete condenser
    m.fs.num_evaporators = 3
    m.fs.set_evaporators = RangeSet(m.fs.num_evaporators)
    m.fs.set_condensers = RangeSet(m.fs.num_evaporators + 1)

    m.fs.evaporator = Evaporator(m.fs.set_evaporators,
                                 property_package_feed=m.fs.properties_feed,
                                 property_package_vapor=m.fs.properties_vapor)
    m.fs.condenser = Condenser(m.fs.set_condensers,
                               property_package=m.fs.properties_vapor)
    m.fs.pump = Pump (property_package=m.fs.properties_vapor)
    m.fs.steam_generator = Heater(property_package=m.fs.properties_vapor)
    
    # Add variable to calculate molal concentration of solute
    # The upper bound is included to ensure that the molality of the electrolyte solution is within saturation and the concentration limits of eNRTL tau and alpha parameters
    m.fs.molal_conc_solute = pyo.Var(m.fs.set_evaporators,
                                     initialize=2,
                                     bounds=(0, 6),
                                     units=pyunits.mol/pyunits.kg,
                                     doc="Molal concentration of solute")
    @m.fs.Constraint(m.fs.set_evaporators,
                     doc="Molal concentration of solute in solvent in mol of TDS/kg of H2O")
    def rule_molal_conc_solute(b, e):
        return  m.fs.molal_conc_solute[e] == (
            (
                b.evaporator[e].properties_brine[0].flow_mass_phase_comp["Liq", "TDS"]/
                b.properties_feed.mw_comp["TDS"] # to convert it to mol/s
            )/b.evaporator[e].properties_brine[0].flow_mass_phase_comp["Liq", "H2O"]
        )

    # Add eNRTL method to calculate the activity coefficients for the electrolyte solution 
    # Note that since water is the only solvent participating in the vapor-liquid equilibrium,the activity coefficient is of water
    if solve_nonideal:

        # Add activity coefficient as a global variable in each evaporator
        m.fs.act_coeff = pyo.Var(m.fs.set_evaporators,
                                 initialize=1,
                                 units=pyunits.dimensionless,
                                 bounds=(0, 20))

        # Declare a block to include the generic properties needed by eNRTL as a state block
        m.fs.enrtl_state = Block(m.fs.set_evaporators)

        if run_multi:
            # Declare a Generic Parameter Block that calls a configuration file that includes eNRTL as the equation of state method.
            # Use the multi-component configuration
            m.fs.prop_enrtl = GenericParameterBlock(**renrtl_multi_config.configuration)

            # Declare a set for the ions in the electrolyte solution and the stoichiometric coefficient of the ions in the solute molecule
            # Calculate the total mass of the solute molecule and the mass ratio of each ion in the solute molecule
            m.fs.set_ions_multi = Set(initialize=["Na+", "Cl-", "SO4_2-"])
            m.fs.ion_coeff_multi = {"Na+": 3, "Cl-": 1, "SO4_2-": 1 } 
            
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
        @m.fs.Constraint(m.fs.set_evaporators,doc="eNRTL activity coefficient for water")
        def eNRTL_activity_coefficient(b, e):
            return (
                b.act_coeff[e] ==
                m.fs.enrtl_state[e].properties[0].act_coeff_phase_comp["Liq", "H2O"]
            )
    else:
        # Add the activity coefficient as a parameter with a value of 1
        m.fs.act_coeff = pyo.Param(m.fs.set_evaporators,
                                   initialize=1,
                                   units=pyunits.dimensionless)

    # Deactivate equilibrium equation from evaporators. 
    # Note that when deactivated, one DOF appears for each evaporator.
    for e in m.fs.set_evaporators:
        m.fs.evaporator[e].eq_brine_pressure.deactivate()

    # Add vapor-liquid equilibrium equation.
    @m.fs.Constraint(m.fs.set_evaporators,
                     doc="Vapor-liquid equilibrium equation")
    def _eq_phase_equilibrium(b, e):
        return (
            1* # mole fraction of water in vapor phase
            b.evaporator[e].properties_brine[0].pressure
        ) == (
            m.fs.act_coeff[e]*
            b.evaporator[e].properties_brine[0].mole_frac_phase_comp["Liq", "H2O"]*
            b.evaporator[e].properties_vapor[0].pressure_sat
        )
    create_arcs(m)

    TransformationFactory("network.expand_arcs").apply_to(m)

    return m


def create_arcs(m):
    # Create arcs to connect units in the flowsheet

    m.fs.evap1brine_to_evap2feed = Arc(
        source=m.fs.evaporator[1].outlet_brine,
        destination=m.fs.evaporator[2].inlet_feed,
        doc="Connect evaporator 1 brine outlet to evaporator 2 inlet"
    )
    
    m.fs.evap1vapor_to_cond2 = Arc(
        source=m.fs.evaporator[1].outlet_vapor,
        destination=m.fs.condenser[2].inlet,
        doc="Connect vapor outlet of evaporator 1 to condenser 2"
    )

    m.fs.evap2vapor_to_cond3 = Arc(
        source=m.fs.evaporator[2].outlet_vapor,
        destination=m.fs.condenser[3].inlet,
        doc="Connect vapor outlet of evaporator 2 to condenser 3"
    )

    m.fs.evap2brine_to_evap3feed = Arc(
        source=m.fs.evaporator[2].outlet_brine,
        destination=m.fs.evaporator[3].inlet_feed,
        doc="Connect evaporator 2 brine outlet to evaporator 3 inlet"
    )

    m.fs.evap3vapor_to_condenser = Arc(
        source=m.fs.evaporator[3].outlet_vapor,
        destination=m.fs.condenser[4].inlet,
        doc="Connect vapor outlet of evaporator 3 to condenser 4"
    )

    m.fs.condenser_to_pump = Arc(
         source=m.fs.condenser[1].outlet,
         destination=m.fs.pump.inlet,
         doc="Connect condenser outlet to pump"
    )

    m.fs.pump_to_generator = Arc(
        source=m.fs.pump.outlet,
        destination=m.fs.steam_generator.inlet,
        doc="Connect pump outlet to generator"
    )
    
    m.fs.generator_to_condenser = Arc(
    source=m.fs.steam_generator.outlet,
    destination=m.fs.condenser[1].inlet,
    doc=" Connect steam generator outlet to condenser"
    )
    
def add_enrtl_method_single(m, n_evap=None):
    
    sb_enrtl = m.fs.enrtl_state[n_evap].properties[0] #renaming the block

    # Populate eNRTL state block
    populate_enrtl_state_vars(sb_enrtl, base="FpcTP")

    # Calculate the total mass of the solute molecule and the mass ratio of each ion in the solute molecule.
    m.fs.enrtl_state[n_evap].mol_mass_ion_molecule = sum(m.fs.ion_coeff_single[j]*sb_enrtl.mw_comp[j]
                                                         for j in m.fs.set_ions_single)
    m.fs.enrtl_state[n_evap].mass_ratio_ion = {
        "Na+": sb_enrtl.mw_comp["Na+"]/m.fs.enrtl_state[n_evap].mol_mass_ion_molecule,
        "Cl-": sb_enrtl.mw_comp["Cl-"]/m.fs.enrtl_state[n_evap].mol_mass_ion_molecule
    }

    # Add constraints to link the outlet temperature, pressure, and mass flowrate of the evaporator brine with the eNRTL properties block. 
    # Note that, since the flow from the seawater property package is in terms of total TDS, we convert the flow from TDS to the respective ions in the seawater.
    m.fs.enrtl_state[n_evap].eq_enrtl_temperature = Constraint(
        expr=(
            sb_enrtl.temperature == m.fs.evaporator[n_evap].properties_brine[0].temperature
        )
    )

    m.fs.enrtl_state[n_evap].eq_enrtl_pressure = Constraint(
        expr=(
            sb_enrtl.pressure == m.fs.evaporator[n_evap].properties_brine[0].pressure
        )
    )

    m.fs.enrtl_state[n_evap].eq_enrtl_flow_mass_H2O = Constraint(
        expr=(
            sb_enrtl.flow_mass_phase_comp["Liq", "H2O"] ==
            m.fs.evaporator[n_evap].properties_brine[0].flow_mass_phase_comp["Liq", "H2O"]
        )
    )

    def enrtl_flow_mass_ion_comp(b, j):
        return (
            sb_enrtl.flow_mass_phase_comp["Liq", j] == (
                (m.fs.evaporator[n_evap].properties_brine[0].flow_mass_phase_comp["Liq", "TDS"]*
                 b.mass_ratio_ion[j])
            )
        )
    m.fs.enrtl_state[n_evap].enrtl_flow_mass_ion_comp_eq = Constraint(m.fs.set_ions_single,
                                                                      rule=enrtl_flow_mass_ion_comp)

    # Add expression to calculate the mean ionic activity coefficient. 
    # It can be commented when the variable is not used. 
    # The equation is taken from reference [1], page 28. 
    # Note that gamma value is used (not log gamma), so to convert it to molal basis we return the log(act_coeff).
    m.fs.enrtl_state[n_evap].mean_act_coeff = Expression(
        expr=log(
            (sb_enrtl.act_coeff_phase_comp["Liq", "Na+"] ** m.fs.ion_coeff_single["Na+"]*
             sb_enrtl.act_coeff_phase_comp["Liq", "Cl-"] ** m.fs.ion_coeff_single["Cl-"])**
            (1/sum(m.fs.ion_coeff_single[j] for j in m.fs.set_ions_single))
        )
    )

    # Add expressions to convert mean ionic activity coefficient to molal basis.
    m.fs.enrtl_state[n_evap].conv_mole_frac_to_molal = Expression(
        expr=log(
            1 +
            (sb_enrtl.mw_comp["H2O"]*2*m.fs.molal_conc_solute[n_evap])/1 # 1 kg of solvent
        )
    )
    m.fs.enrtl_state[n_evap].molal_mean_act_coeff = Expression(
        expr=(m.fs.enrtl_state[n_evap].mean_act_coeff -
              m.fs.enrtl_state[n_evap].conv_mole_frac_to_molal)
    )

def add_enrtl_method_multi(m, n_evap=None):

    sb_enrtl = m.fs.enrtl_state[n_evap].properties[0] #renaming the block

    # Populate eNRTL state block
    populate_enrtl_state_vars_multi(sb_enrtl, base="FpcTP")

    # Calculate the total mass of the solute molecule and the mass ratio of each ion in each solute molecule.
    m.fs.enrtl_state[n_evap].mol_mass_ion_molecule_nacl = sum(m.fs.ion_coeff_nacl[j]*sb_enrtl.mw_comp[j]
                                                 for j in m.fs.set_ions_nacl)
    
    m.fs.enrtl_state[n_evap].mol_mass_ion_molecule_na2so4 = sum(m.fs.ion_coeff_na2so4[j]*sb_enrtl.mw_comp[j]
                                                 for j in m.fs.set_ions_na2so4)
    
    m.fs.enrtl_state[n_evap].mass_ratio_ion = {
        "Na+": sb_enrtl.mw_comp["Na+"]/(m.fs.enrtl_state[n_evap].mol_mass_ion_molecule_nacl + m.fs.enrtl_state[n_evap].mol_mass_ion_molecule_na2so4),
        "Cl-": sb_enrtl.mw_comp["Cl-"]/m.fs.enrtl_state[n_evap].mol_mass_ion_molecule_nacl,
        "SO4_2-": sb_enrtl.mw_comp["SO4_2-"]/m.fs.enrtl_state[n_evap].mol_mass_ion_molecule_na2so4
    }

    # Add constraints to link the outlet temperature, pressure, and mass flowrate of the evaporator brine with the eNRTL properties block. 
    # Note that, since the flow from the seawater property package is in terms of total TDS, we convert the flow from TDS to the respective ions in the seawater.
    m.fs.enrtl_state[n_evap].eq_enrtl_temperature = Constraint(
        expr=(
            sb_enrtl.temperature == m.fs.evaporator[n_evap].properties_brine[0].temperature
        )
    )

    m.fs.enrtl_state[n_evap].eq_enrtl_pressure = Constraint(
        expr=(
            sb_enrtl.pressure == m.fs.evaporator[n_evap].properties_brine[0].pressure
        )
    )

    m.fs.enrtl_state[n_evap].eq_enrtl_flow_mass_H2O = Constraint(
        expr=(
            sb_enrtl.flow_mass_phase_comp["Liq", "H2O"] ==
            m.fs.evaporator[n_evap].properties_brine[0].flow_mass_phase_comp["Liq", "H2O"]
        )
    )

    def enrtl_flow_mass_ion_comp(b, j):
        return (
            sb_enrtl.flow_mass_phase_comp["Liq", j] == (
                (m.fs.evaporator[n_evap].properties_brine[0].flow_mass_phase_comp["Liq", "TDS"]*
                 b.mass_ratio_ion[j])
            )
        )
    m.fs.enrtl_state[n_evap].enrtl_flow_mass_ion_comp_eq = Constraint(m.fs.set_ions_multi,
                                                                      rule=enrtl_flow_mass_ion_comp)

    # Add expression to calculate the mean ionic activity coefficient. 
    # It can be commented when the variable is not used. 
    # The equation is taken from reference [1], page 28. 
    # Note that gamma value is used (not log gamma), so to convert it to molal basis we return the log(act_coeff).
    m.fs.enrtl_state[n_evap].mean_act_coeff = Expression(
        expr=log(
            (sb_enrtl.act_coeff_phase_comp["Liq", "Na+"] ** m.fs.ion_coeff_multi["Na+"]*
             sb_enrtl.act_coeff_phase_comp["Liq", "Cl-"] ** m.fs.ion_coeff_multi["Cl-"]*
             sb_enrtl.act_coeff_phase_comp["Liq", "SO4_2-"] ** m.fs.ion_coeff_multi["SO4_2-"])**
            (1/sum(m.fs.ion_coeff_multi[j] for j in m.fs.set_ions_multi))
        )
    )

    # Add expressions to convert mean ionic activity coefficient to molal basis.
    m.fs.enrtl_state[n_evap].conv_mole_frac_to_molal = Expression(
        expr=log(
            1 +
            (sb_enrtl.mw_comp["H2O"]*2*m.fs.molal_conc_solute[n_evap])/1 # 1 kg of solvent
        )
    )
    m.fs.enrtl_state[n_evap].molal_mean_act_coeff = Expression(
        expr=(m.fs.enrtl_state[n_evap].mean_act_coeff -
              m.fs.enrtl_state[n_evap].conv_mole_frac_to_molal)
    )

def set_scaling(m):
    # Scaling factors are added for all the variables
    for var in m.fs.component_data_objects(pyo.Var, descend_into=True):
            if "temperature" in var.name:
                iscale.set_scaling_factor(var, 1e-2) 
            if "lmtd" in var.name:
                iscale.set_scaling_factor(var, 1e-1)
            if "delta_temperature_in" in var.name:
                iscale.set_scaling_factor(var, 1e-1)
            if "delta_temperature_out" in var.name:
                iscale.set_scaling_factor(var, 1e-1)
            if "pressure" in var.name:
                iscale.set_scaling_factor(var, 1e-6)
            if "dens_mass_" in var.name:
                iscale.set_scaling_factor(var, 1e3)
            if "flow_mass_phase_comp" in var.name:
                iscale.set_scaling_factor(var, 1e3)
            if "flow_mol_phase_comp" in var.name:
                 iscale.set_scaling_factor(var, 1e2)
            if "area" in var.name:
                iscale.set_scaling_factor(var, 1e-1)
            if "heat_transfer" in var.name:
                iscale.set_scaling_factor(var, 1e-4)
            if "heat" in var.name:
                iscale.set_scaling_factor(var, 1e-4)
            if "U" in var.name:
                iscale.set_scaling_factor(var, 1e-2)
            if "work" in var.name:
                iscale.set_scaling_factor(var, 1e-5)

    # Done to overide certain scaling factors
    m.fs.properties_feed.set_default_scaling("flow_mass_phase_comp", 1e1, index=("Liq", "H2O"))
    m.fs.properties_feed.set_default_scaling("flow_mass_phase_comp", 1e2, index=("Liq", "TDS"))
    m.fs.properties_vapor.set_default_scaling("flow_mass_phase_comp", 1e1, index=("Vap", "H2O"))
    m.fs.properties_vapor.set_default_scaling("flow_mass_phase_comp", 1e1, index=("Liq", "H2O"))

    # Calculate scaling factors
    iscale.calculate_scaling_factors(m)

def set_model_inputs(m):

    # Feed
    # Seawater feed flowrate was determined from backcalculation from provided experimental SEC,Qin value, and water recovery % from [2]
    # TDS flowrate = [(TDS concentration=23,170ppm * Seawater feed flowrate=0.15)]/1.0e6
    m.fs.evaporator[1].inlet_feed.flow_mass_phase_comp[0, "Liq", "H2O"].fix(0.15) # kg/s
    m.fs.evaporator[1].inlet_feed.flow_mass_phase_comp[0, "Liq", "TDS"].fix(0.0035) # kg/s
    m.fs.evaporator[1].inlet_feed.temperature[0].fix(27 + 273.15) # K
    m.fs.evaporator[1].inlet_feed.pressure[0].fix(101325) # Pa

    # Condenser[1]
    m.fs.condenser[1].outlet.temperature[0].fix(69 + 273.15) # K
    m.fs.condenser[1].inlet.flow_mass_phase_comp[0, 'Liq', 'H2O'].fix(0.00) # kg/s
    
    # Pressure changer
    m.fs.pump.outlet.pressure.fix(30000) # Pa
    m.fs.pump.efficiency_pump.fix(0.8) # in fraction
    
    # Steam generator
    m.fs.steam_generator.outlet.temperature.fix(69.1 + 273.15) # K
    m.fs.steam_generator.control_volume.heat[0].fix(96370) # W

    # Evaporator[1]
    m.fs.evaporator[1].outlet_brine.temperature[0].fix(65 + 273.15) # K
    m.fs.evaporator[1].U.fix(500) # W/K-m^2
    m.fs.evaporator[1].area.fix(10) # m^2
    m.fs.evaporator[1].delta_temperature_in.fix(10) # K
    m.fs.evaporator[1].delta_temperature_out.fix(8) # K

    # Condenser[2]
    m.fs.condenser[2].outlet.temperature[0].fix(64 + 273.15) # K

    # Evaporator[2]
    m.fs.evaporator[2].U.fix(500) # W/K-m^2
    m.fs.evaporator[2].area.fix(10) # m^2
    m.fs.evaporator[2].outlet_brine.temperature[0].fix(66 + 273.15) # K
    m.fs.evaporator[2].delta_temperature_in.fix(10) # K
    m.fs.evaporator[2].delta_temperature_out.fix(8) # K

    # Condenser[3]
    m.fs.condenser[3].outlet.temperature[0].fix(60 + 273.15) # K

    # Evaporator[3]
    m.fs.evaporator[3].U.fix(500) # W/K-m^2
    m.fs.evaporator[3].area.fix(10) # m^2
    m.fs.evaporator[3].outlet_brine.temperature[0].fix(70 + 273.15) # K
    m.fs.evaporator[3].delta_temperature_in.fix(10) # K
    m.fs.evaporator[3].delta_temperature_out.fix(8) # K

    # Condenser[4]
    m.fs.condenser[4].outlet.temperature[0].fix(55 + 273.15) # K

def initialize(m, solver=None, outlvl=idaeslog.NOTSET):

    # Initialize condenser [1]
    m.fs.condenser[1].initialize_build(heat=-m.fs.evaporator[3].heat_transfer.value)

    # Initialize pump
    propagate_state(m.fs.condenser_to_pump)
    m.fs.pump.initialize(outlvl=outlvl)

    # Initialize steam generator
    propagate_state(m.fs.pump_to_generator)
    m.fs.steam_generator.initialize(outlvl=outlvl)
    
    # Initialize evaporator [1]
    m.fs.evaporator[1].initialize(outlvl=outlvl)
    
    # Initialize condenser [2]
    propagate_state(m.fs.evap1vapor_to_cond2)
    m.fs.condenser[2].initialize_build(heat=-m.fs.evaporator[1].heat_transfer.value)

    # Initialize evaporator [2]
    propagate_state(m.fs.evap1brine_to_evap2feed)
    m.fs.evaporator[2].initialize(outlvl=outlvl)

    # Initialize condenser [3]
    propagate_state(m.fs.evap2vapor_to_cond3)
    m.fs.condenser[3].initialize_build(heat=-m.fs.evaporator[2].heat_transfer.value)
    
    # Initialize evaporator [3]
    propagate_state(m.fs.evap2brine_to_evap3feed)
    m.fs.evaporator[3].initialize(outlvl=outlvl)

    #  Initialize condenser [4]
    propagate_state(m.fs.evap3vapor_to_condenser)
    m.fs.condenser[4].initialize(outlvl=outlvl)

    print()
    print('****** Start initialization')

    if not degrees_of_freedom(m) == 0:
       raise ConfigurationError(
           "The degrees of freedom after building the model are not 0. "
           "You have {} degrees of freedom. "
           "Please check your inputs to ensure a square problem "
           "before initializing the model.".format(degrees_of_freedom(m))
       )
    init_results = solver.solve(m, tee=False)
    
    print(' Initialization solver status:', init_results.solver.termination_condition)
    print('****** End initialization')
    print()

def add_bounds(m):

    for i in m.fs.set_evaporators:
        m.fs.evaporator[i].area.setlb(10)
        m.fs.evaporator[i].area.setub(None)
        m.fs.evaporator[i].outlet_brine.temperature[0].setub(73 + 273.15) # in K

def print_results(m):
    m.fs.steam_generator.report()
    m.fs.pump.report()

    for i in m.fs.set_condensers:
        m.fs.condenser[i].report()

    for i in m.fs.set_evaporators:
        m.fs.molal_conc_solute_feed = (
            (
                value(m.fs.evaporator[i].inlet_feed.flow_mass_phase_comp[0, "Liq", "TDS"])/
                value(m.fs.properties_feed.mw_comp["TDS"])
            )/value(m.fs.evaporator[i].inlet_feed.flow_mass_phase_comp[0, "Liq", "H2O"])
        )

        # Material properties of feed, brine outlet, and vapor outlet
        sw_blk = m.fs.evaporator[i].properties_feed[0]
        brine_blk = m.fs.evaporator[i].properties_brine[0]
        vapor_blk = m.fs.evaporator[i].properties_vapor[0]
        print()
        print()
        print('====================================================================================')
        if solve_nonideal:
            print('Unit : m.fs.evaporator[{}] (non-ideal)'.format(i))
        else:
            print('Unit : m.fs.evaporator[{}] (ideal)'.format(i))
        print('------------------------------------------------------------------------------------')
        print('    Unit performance')
        print()
        print('    Variables:')
        print()
        print('    Key   Value')
        print('      delta temperature_in  : {:>4.3f}'.format(
            value(m.fs.evaporator[i].delta_temperature_in)))
        print('      delta temperature_out : {:>4.3f}'.format(
            value(m.fs.evaporator[i].delta_temperature_out)))
        print('                      Area  : {:>4.3f}'.format(
            value(m.fs.evaporator[i].area)))
        print('                         U  : {:>4.3f}'.format(
            value(m.fs.evaporator[i].U)))
        print('                   UA_term  : {:>4.3f}'.format(
            value(m.fs.UA_term[i])))
        if solve_nonideal:
            print('             act_coeff* H2O : {:>4.4f} (log:{:>4.4f})'.format(
                value(m.fs.enrtl_state[i].properties[0].act_coeff_phase_comp["Liq", "H2O"]),
                value(log(m.fs.enrtl_state[i].properties[0].act_coeff_phase_comp["Liq", "H2O"]))))
            if run_multi:
                for j in m.fs.set_ions_multi:
                    print('             act_coeff* {} : {:>4.4f} (log:{:>4.4f})'.format(
                        j, value(m.fs.enrtl_state[i].properties[0].act_coeff_phase_comp["Liq", j]),
                        value(log(m.fs.enrtl_state[i].properties[0].act_coeff_phase_comp["Liq", j]))))
                    print('        mean_ion_actv_coeff : {:>4.4f} (log: {:>4.4f})'.format(
                        exp(value(m.fs.enrtl_state[i].mean_act_coeff)),
                        value(m.fs.enrtl_state[i].mean_act_coeff)))
                    print(' molal mean_ionic_actv_coeff: {:>4.4f} (log: {:>4.4f})'.format(
                        exp(value(m.fs.enrtl_state[i].molal_mean_act_coeff)),
                        value(m.fs.enrtl_state[i].molal_mean_act_coeff)))
                    print('    *calculated with eNRTL')
            else:
                for j in m.fs.set_ions_single:
                    print('             act_coeff* {} : {:>4.4f} (log:{:>4.4f})'.format(
                        j, value(m.fs.enrtl_state[i].properties[0].act_coeff_phase_comp["Liq", j]),
                        value(log(m.fs.enrtl_state[i].properties[0].act_coeff_phase_comp["Liq", j]))))
                    print('        mean_ion_actv_coeff : {:>4.4f} (log: {:>4.4f})'.format(
                        exp(value(m.fs.enrtl_state[i].mean_act_coeff)),
                        value(m.fs.enrtl_state[i].mean_act_coeff)))
                    print(' molal mean_ionic_actv_coeff: {:>4.4f} (log: {:>4.4f})'.format(
                        exp(value(m.fs.enrtl_state[i].molal_mean_act_coeff)),
                        value(m.fs.enrtl_state[i].molal_mean_act_coeff)))
                    print('    *calculated with eNRTL')
                
        else:
            print('              act_coeff H2O : {:>4.4f}'.format(
                value(m.fs.act_coeff[i])))
        print('------------------------------------------------------------------------------------')
        print('   Stream Table')
        print('                                      inlet_feed    outlet_brine   outlet_vapor')
        print('   flow_mass_phase_comp (kg/s) {:>15.4f} {:>14.4f} {:>14.4f}'.format(
            value(sw_blk.flow_mass_phase_comp["Liq", "H2O"]
                  + sw_blk.flow_mass_phase_comp["Liq", "TDS"]),
            value(brine_blk.flow_mass_phase_comp["Liq", "H2O"]
                  + brine_blk.flow_mass_phase_comp["Liq", "TDS"]),
            value(vapor_blk.flow_mass_phase_comp["Vap", "H2O"])))
        print('   mass_frac_phase_comp (Liq, H2O){:>12.4f} {:>14.4f}            -'.format(
            value(sw_blk.mass_frac_phase_comp["Liq", "H2O"]),
            value(brine_blk.mass_frac_phase_comp["Liq", "H2O"])))
        print('   mass_frac_phase_comp (Liq, TDS){:>12.4f} {:>14.4f}            -'.format(
            value(sw_blk.mass_frac_phase_comp["Liq", "TDS"]),
            value(brine_blk.mass_frac_phase_comp["Liq", "TDS"])))
        print('   mole_frac_phase_comp (Liq, H2O){:>12.4f} {:>14.4f} {:>14.4f}'.format(
            value(sw_blk.mole_frac_phase_comp["Liq", "H2O"]),
            value(brine_blk.mole_frac_phase_comp["Liq", "H2O"]),
            value(vapor_blk.mole_frac_phase_comp["Liq", "H2O"])))
        print('   mole_frac_phase_comp (Liq, TDS){:>12.4f} {:>14.4f}            -'.format(
            value(sw_blk.mole_frac_phase_comp["Liq", "TDS"]),
            value(brine_blk.mole_frac_phase_comp["Liq", "TDS"])))
        print('   molal_conc_solute (mol TDS/kg H2O) {:>8.4f} {:>14.4f}            -'.format(
            m.fs.molal_conc_solute_feed,
            value(m.fs.molal_conc_solute[i])))
        print('   temperature (K) {:>27.4f} {:>14.4f} {:>14.4f}'.format(
            value(sw_blk.temperature),
            value(brine_blk.temperature),
            value(vapor_blk.temperature)))
        print('   pressure (Pa) {:>29.4f} {:>14.4f} {:>14.4f}'.format(
            value(sw_blk.pressure),
            value(brine_blk.pressure),
            value(vapor_blk.pressure)))
        print('   saturation pressure (Pa) {:>18.4f} {:>14.4f} {:>14.4f}'.format(
            value(sw_blk.pressure_sat),
            value(brine_blk.pressure_sat),
            value(vapor_blk.pressure_sat)))
        print()
        if solve_nonideal:
            print('   eNRTL state block')
            print('   flow_mass_phase_comp (Liq, H2O) {:>11.4f}'.format(
                value(m.fs.enrtl_state[i].properties[0].flow_mass_phase_comp["Liq", "H2O"])))
            if run_multi:
                for j in m.fs.set_ions_multi:
                    print('   flow_mass_phase_comp (Liq, {}) {:>11.4f}'.format(
                        j, value(m.fs.enrtl_state[i].properties[0].flow_mass_phase_comp["Liq", j])))
                sum_tds_brine_out = sum(value(m.fs.enrtl_state[i].properties[0].flow_mass_phase_comp["Liq", j])
                                        for j in m.fs.set_ions_multi)
                print('    >>flow_mass_phase_comp (Liq, TDS) {:>8.4f}'.format(
                    sum_tds_brine_out))
                if sum_tds_brine_out - value(brine_blk.flow_mass_phase_comp["Liq", "TDS"]) >= 1e-1:
                    print("     **ERROR: Flow mass of TDS ({:>2.4f} kg/s) not equivalent"
                        " to sum of ions mass ({:>2.4f} kg/s)".format(
                            sum_tds_brine_out,
                            value(brine_blk.flow_mass_phase_comp["Liq", "TDS"])))
                    print("     Check balances!")
                    print('   temperature (K) {:>27.4f}'.format(
                        value(m.fs.enrtl_state[i].properties[0].temperature)))
                    print('   pressure (Pa) {:>29.4f}'.format(
                        value(m.fs.enrtl_state[i].properties[0].pressure)))
                    print()
            else:
                for j in m.fs.set_ions_single:
                    print('   flow_mass_phase_comp (Liq, {}) {:>11.4f}'.format(
                        j, value(m.fs.enrtl_state[i].properties[0].flow_mass_phase_comp["Liq", j])))
                sum_tds_brine_out = sum(value(m.fs.enrtl_state[i].properties[0].flow_mass_phase_comp["Liq", j])
                                        for j in m.fs.set_ions_single)
                print('    >>flow_mass_phase_comp (Liq, TDS) {:>8.4f}'.format(
                    sum_tds_brine_out))
                if sum_tds_brine_out - value(brine_blk.flow_mass_phase_comp["Liq", "TDS"]) >= 1e-1:
                    print("     **ERROR: Flow mass of TDS ({:>2.4f} kg/s) not equivalent"
                        " to sum of ions mass ({:>2.4f} kg/s)".format(
                            sum_tds_brine_out,
                            value(brine_blk.flow_mass_phase_comp["Liq", "TDS"])))
                    print("     Check balances!")
                    print('   temperature (K) {:>27.4f}'.format(
                        value(m.fs.enrtl_state[i].properties[0].temperature)))
                    print('   pressure (Pa) {:>29.4f}'.format(
                        value(m.fs.enrtl_state[i].properties[0].pressure)))
                    print()
        print()
        print('====================================================================================')
        print()
    

    print('Variable                                 Value')
    print(' Total water produced (gal/min) {:>18.4f}'.format(
        value(m.fs.total_water_produced_gpm)))
    print(' Specific energy consumption (SC, kWh/m3) {:>8.4f}'.format(
        value(m.fs.specific_energy_consumption)))
    print(' Performance Ratio {:>31.4f}'.format(
        value(m.fs.performance_ratio)))
    print(' Water recovery (%) {:>30.4f}'.format(value(m.fs.water_recovery)*100))
    for i in m.fs.set_evaporators:
        print(' Molal conc solute evap {} (mol/kg) {:>15.4f}'.format(i, value(m.fs.molal_conc_solute[i])))
    print()
    print()
        
def model_analysis(m, water_rec=None):
    # Unfix for optimization of variable
    # Condenser[1]
    m.fs.condenser[1].control_volume.heat[0].unfix()
    
    # Evaporator[1]
    m.fs.evaporator[1].area.unfix()
    m.fs.evaporator[1].outlet_brine.temperature[0].unfix()
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

    # Condenser[4]
    m.fs.condenser[4].control_volume.heat[0].unfix()
    
    # Steam generator
    m.fs.steam_generator.control_volume.properties_in[0.0].temperature.unfix()
    m.fs.steam_generator.control_volume.heat[0].unfix()

    # delta_temperature_in = condenser inlet temp - evaporator brine temp
    # delta_temperature_out = condenser outlet temp - evaporator brine temp
    for e in m.fs.set_evaporators:
        m.fs.evaporator[e].delta_temperature_in.fix(3)

    @m.fs.Constraint(doc="Generator area upper bound")
    def gen_heat_bound(b):
        return b.steam_generator.control_volume.heat[0] <= 110000
    

    # Add constraint to make sure the pressure in the evaporators 2 and 3 is smaller than the pressure in evaporator 1
    m.fs.set2_evaporators = RangeSet(m.fs.num_evaporators - 1)
    @m.fs.Constraint(m.fs.set2_evaporators)
    def eq_upper_bound_evaporators_pressure(b, e):
        return (
            b.evaporator[e + 1].outlet_brine.pressure[0] <=
            b.evaporator[e].outlet_brine.pressure[0]
        )

    # Add expression to calculate the UA term
    @m.fs.Expression(m.fs.set_evaporators,
                     doc="Overall heat trasfer coefficient and area term")
    def UA_term(b, e):
        return b.evaporator[e].area*b.evaporator[e].U
    
    # Calculate total water produced and total specific energy consumption.
    m.fs.water_density = pyo.Param(initialize=1000,
                                   units=pyunits.kg/pyunits.m**3)
    
    @m.fs.Expression()
    def total_water_produced_gpm(b):
         return pyo.units.convert(
             (
              b.condenser[2].control_volume.properties_out[0].flow_mass_phase_comp["Liq", "H2O"] +
              b.condenser[3].control_volume.properties_out[0].flow_mass_phase_comp["Liq", "H2O"] +
              b.condenser[4].control_volume.properties_out[0].flow_mass_phase_comp["Liq", "H2O"])/m.fs.water_density,
             to_units=pyunits.gallon/pyunits.minute
         )
    
    #Backcalculation from [2] produced a latent heat of vaporization at Trefof 73degC is 2,319.05 kJ/kg
    @m.fs.Expression()
    def performance_ratio(b):
        return (
              (b.condenser[2].control_volume.properties_out[0].flow_mass_phase_comp["Liq", "H2O"] +
              b.condenser[3].control_volume.properties_out[0].flow_mass_phase_comp["Liq", "H2O"] +
              b.condenser[4].control_volume.properties_out[0].flow_mass_phase_comp["Liq", "H2O"]) * 2319.05)/(b.steam_generator.heat_duty[0]/1000)
    
    m.fs.specific_energy_consumption = pyo.Var(initialize=11,
                                               units=pyunits.kW*pyunits.hour/pyunits.m**3,
                                               bounds=(0, 1e3))
    
    @m.fs.Constraint(doc="Specific energy consumption [kWh/m^3]")
    def eq_specific_energy_consumption(b):
         return b.specific_energy_consumption == (
             pyo.units.convert(b.steam_generator.heat_duty[0], #in Watts
                             to_units=pyunits.kW)/
             pyo.units.convert(m.fs.total_water_produced_gpm, to_units=pyunits.m**3/pyunits.hour)
         )
         
    m.fs.water_recovery = pyo.Var(initialize=0.2,
                                  bounds=(0, 1),
                                  units=pyunits.dimensionless,
                                  doc="Water recovery")
    
    # Water recovery equation used in [2]
    @m.fs.Constraint()
    def rule_water_recovery(b):
        return m.fs.water_recovery == (
            b.condenser[2].control_volume.properties_out[0].flow_mass_phase_comp["Liq", "H2O"] +
            b.condenser[3].control_volume.properties_out[0].flow_mass_phase_comp["Liq", "H2O"] +
            b.condenser[4].control_volume.properties_out[0].flow_mass_phase_comp["Liq", "H2O"]
        ) / (
            m.fs.evaporator[1].inlet_feed.flow_mass_phase_comp[0, "Liq", "H2O"] + 
            m.fs.evaporator[1].inlet_feed.flow_mass_phase_comp[0, "Liq", "TDS"]
        )


    @m.fs.Constraint()
    def water_recovery_ub(b):
        return b.water_recovery >= water_rec
    @m.fs.Constraint()
    def water_recovery_lb(b):
        return b.water_recovery <= water_rec

if __name__ == "__main__":

    optarg = {
        "max_iter": 500,
        "tol": 1e-8
    }
    solver = get_solver('ipopt', optarg)
    water_recovery_data = [0.6]
    for c in range(len(water_recovery_data)):
        m = create_model()

        set_scaling(m)

        set_model_inputs(m)
   
        initialize(m, solver=solver)
        
        add_bounds(m)

        model_analysis(m, water_rec=water_recovery_data[c])

        results = solver.solve(m, tee=True)
        
        print_results(m)
