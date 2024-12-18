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

This is a close loop 3MED-only model configuration. 
The model uses experimental conditions from [2] and validates well at a water recovery of 60%.
"""
import logging
import pytest

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
    assert_optimal_termination,
)
from pyomo.network import Arc
from pyomo.environ import units as pyunits
from pyomo.util.check_units import assert_units_consistent

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
from idaes.models.unit_models import Pump, Heater

# Import property packages and WaterTAP components
import watertap.property_models.seawater_prop_pack as props_sw
import watertap.property_models.water_prop_pack as props_w

from watertap.unit_models.mvc.components import Evaporator, Condenser

# Import configuration dictionary
import enrtl_config_FpcTP  # single electrolyte
import renrtl_multi_config  # multi electrolytes

module = __import__("MED_eNRTL")

# Access the functions from the module
populate_enrtl_state_vars = module.populate_enrtl_state_vars
populate_enrtl_state_vars_multi = module.populate_enrtl_state_vars_multi
create_model = module.create_model
create_arcs = module.create_arcs
add_enrtl_method_single = module.add_enrtl_method_single
add_enrtl_method_multi = module.add_enrtl_method_multi
set_scaling = module.set_scaling
set_model_inputs = module.set_model_inputs
initialize = module.initialize
add_bounds = module.add_bounds
model_analysis = module.model_analysis

logging.basicConfig(level=logging.INFO)
logging.getLogger("pyomo.repn.plugins.nl_writer").setLevel(logging.ERROR)

# solve_nonideal gives the option to solve an ideal and nonideal system
# If solve_nonideal is set to true, eNRTL is used to calculate the activity coefficients of solvent and solutes;
# when set to False, the model is solved assuming an ideal system with an activity coefficient of 1 for the solvent
solve_nonideal = True
run_multi = False

class TestMED:
    @pytest.fixture(scope="class")
    def MED_eNRTL (self):
    m = create_model()
    return m
    
    @pytest.mark.unit
    def test_build_model(self, MED_eNRTL):
        m = MED_eNRTL

        # test model set up
        assert isinstance(m, ConcreteModel)
        assert isinstance(m.fs, FlowsheetBlock)
        assert isinstance(m.fs.properties_vapor, props_w.WaterParameterBlock)
        assert isinstance(m.fs.properties_feed, props_sw.SeawaterParameterBlock)

        # test unit models
        assert isinstance(m.fs.feed, Feed)
        assert isinstance(m.fs.evaporator, Evaporator)
        assert isinstance(m.fs.condenser, Condenser)
        assert isinstance(m.fs.pump, Pump)
        assert isinstance(m.fs.steam_generator, Heater)

    @pytest.mark.unit
    def test_create_arcs(self, MED_eNRTL):
        m = MED_eNRTL
        create_arcs(m)

        arc_dict = {
            m.fs.evap1brine_to_evap2feed: (
                m.fs.evaporator[1].outlet_brine,
                m.fs.evaporator[2].inlet_feed,
            ),
            m.fs.evap1vapor_to_cond2: (
                m.fs.evaporator[1].outlet_vapor,
                m.fs.condenser[2].inlet,
            ),
            m.fs.evap2vapor_to_cond3: (
                m.fs.evaporator[2].outlet_vapor,
                m.fs.condenser[3].inlet,
            ),
            m.fs.evap2brine_to_evap3feed: (
                m.fs.evaporator[2].outlet_brine,
                m.fs.evaporator[3].inlet_feed,
            ),
            m.fs.evap3vapor_to_condenser: (
                m.fs.evaporator[3].outlet_vapor,
                m.fs.condenser[4].inlet,
            ),
            m.fs.condenser_to_pump: (m.fs.condenser[1].outlet, m.fs.pump.inlet),
            m.fs.pump_to_generator: (m.fs.pump.outlet, m.fs.steam_generator.inlet),
            m.fs.generator_to_condenser: (
                m.fs.steam_generator.outlet,
                m.fs.condenser[1].inlet,
            ),
        }
        for arc, port_tpl in arc_dict.items():
            assert arc.source is port_tpl[0]
            assert arc.destination is port_tpl[1]

        # units
        assert_units_consistent(m.fs)

    @pytest.mark.component
    def test_set_model_inputs(self, MED_eNRTL):
        m = MED_eNRTL
        set_model_inputs(m)

        # check fixed variables
        # Feed
        assert (
            m.fs.evaporator[1]
            .inlet_feed.flow_mass_phase_comp[0, "Liq", "H2O"]
            .is_fixed()
        )
        assert (
            value(m.fs.evaporator[1].inlet_feed.flow_mass_phase_comp[0, "Liq", "H2O"])
            == 0.15
        )
        assert (
            m.fs.evaporator[1]
            .inlet_feed.flow_mass_phase_comp[0, "Liq", "TDS"]
            .is_fixed()
        )
        assert (
            value(m.fs.evaporator[1].inlet_feed.flow_mass_phase_comp[0, "Liq", "TDS"])
            == 0.0035
        )
        assert m.fs.evaporator[1].inlet_feed.temperature[0].is_fixed()
        assert value(m.fs.evaporator[1].inlet_feed.temperature[0]) == 27 + 273.15
        assert m.fs.evaporator[1].inlet_feed.pressure[0].is_fixed()
        assert value(m.fs.evaporator[1].inlet_feed.pressure[0]) == 101325

        # Condenser[1]
        assert m.fs.condenser[1].outlet.temperature[0].is_fixed()
        assert value(m.fs.condenser[1].outlet.temperature[0]) == 69 + 273.15
        assert m.fs.condenser[1].inlet.flow_mass_phase_comp[0, "Liq", "H2O"].is_fixed()
        assert (
            value(m.fs.condenser[1].inlet.flow_mass_phase_comp[0, "Liq", "H2O"]) == 0.00
        )

        # Pressure changer
        assert m.fs.pump.outlet.pressure.is_fixed()
        assert value(m.fs.pump.outlet.pressure) == 30000
        assert m.fs.pump.efficiency_pump.is_fixed()
        assert value(m.fs.pump.efficiency_pump) == 0.8

        # Steam generator
        assert m.fs.steam_generator.outlet.temperature.is_fixed()
        assert value(m.fs.steam_generator.outlet.temperature) == 69.1 + 273.15
        assert m.fs.steam_generator.control_volume.heat[0].is_fixed()
        assert value(m.fs.steam_generator.control_volume.heat[0]) == 96370

        # Evaporator[1]
        assert m.fs.evaporator[1].outlet_brine.temperature[0].is_fixed()
        assert value(m.fs.evaporator[1].outlet_brine.temperature[0]) == 65 + 273.15
        assert m.fs.evaporator[1].U.is_fixed()
        assert value(m.fs.evaporator[1].U) == 500
        assert m.fs.evaporator[1].area.is_fixed()
        assert value(m.fs.evaporator[1].area) == 10
        assert m.fs.evaporator[1].delta_temperature_in.is_fixed()
        assert value(m.fs.evaporator[1].delta_temperature_in) == 10
        assert m.fs.evaporator[1].delta_temperature_out.is_fixed()
        assert value(m.fs.evaporator[1].delta_temperature_out) == 8

        # Condenser[2]
        assert m.fs.condenser[2].outlet.temperature[0].is_fixed()
        assert value(m.fs.condenser[2].outlet.temperature[0]) == 64 + 273.15

        # Evaporator[2]
        assert m.fs.evaporator[2].U.is_fixed()
        assert value(m.fs.evaporator[2].U) == 500
        assert m.fs.evaporator[2].area.is_fixed()
        assert value(m.fs.evaporator[2].area) == 10
        assert m.fs.evaporator[2].outlet_brine.temperature[0].is_fixed()
        assert value(m.fs.evaporator[2].outlet_brine.temperature[0]) == 66 + 273.15
        assert m.fs.evaporator[2].delta_temperature_in.is_fixed()
        assert value(m.fs.evaporator[2].delta_temperature_in) == 10
        assert m.fs.evaporator[2].delta_temperature_out.is_fixed()
        assert value(m.fs.evaporator[2].delta_temperature_out) == 8

        # Condenser[3]
        assert m.fs.condenser[3].outlet.temperature[0].is_fixed()
        assert value(m.fs.condenser[3].outlet.temperature[0]) == 60 + 273.15

        # Evaporator[3]
        assert m.fs.evaporator[3].U.is_fixed()
        assert value(m.fs.evaporator[3].U) == 500
        assert m.fs.evaporator[3].area.is_fixed()
        assert value(m.fs.evaporator[3].area) == 10
        assert m.fs.evaporator[3].outlet_brine.temperature[0].is_fixed
        assert value(m.fs.evaporator[3].outlet_brine.temperature[0]) == 70 + 273.15
        assert m.fs.evaporator[3].delta_temperature_in.is_fixed()
        assert value(m.fs.evaporator[3].delta_temperature_in) == 10
        assert m.fs.evaporator[3].delta_temperature_out.is_fixed()
        assert value(m.fs.evaporator[3].delta_temperature_out) == 8

        # Condenser[4]
        assert m.fs.condenser[4].outlet.temperature[0].is_fixed()
        assert value(m.fs.condenser[4].outlet.temperature[0]) == 55 + 273.15

    @pytest.mark.component
    @pytest.mark.requires_idaes_solver
    def test_initialize(self, MED_eNRTL):
        m = MED_eNRTL
        initialize(m)

        assert value(m.fs.evaporator[1].U) == pytest.approx(500, rel=1e-3)
        assert value(m.fs.evaporator[1].delta_temperature_out) == pytest.approx(
            8, rel=1e-3
        )
        assert value(m.fs.evaporator[2].U) == pytest.approx(500, rel=1e-3)
        assert value(m.fs.evaporator[2].delta_temperature_out) == pytest.approx(
            8, rel=1e-3
        )
        assert value(m.fs.evaporator[3].U) == pytest.approx(500, rel=1e-3)
        assert value(m.fs.evaporator[3].delta_temperature_out) == pytest.approx(
            8, rel=1e-3
        )
        assert value(m.fs.pump.outlet.pressure) == pytest.approx(30000, rel=1e-3)
        assert value(m.fs.steam_generator.outlet.temperature) == pytest.approx(
            69.1 + 273.15, rel=1e3
        )
        assert value(m.fs.steam_generator.control_volume.heat[0]) == pytest.approx(
            96370, rel=1e-3
        )

        assert degrees_of_freedom(m) == 0

    @pytest.mark.component
    @pytest.mark.requires_idaes_solver
    def test_model_analysis(self, MED_eNRTL):
        m = MED_eNRTL
        model_analysis(m)

        solver = get_solver()
        initialize(m, solver=solver)

        results = solver.solve(m, tee=False)
        assert_optimal_termination(results)

        # additional constraints, variables, and expressions
        assert isinstance(m.fs.gen_heat_bound, Constraint)
        for e in m.fs.set2_evaporators:
            assert isinstance(m.fs.eq_upper_bound_evaporators_pressure[e], Constraint)
        assert isinstance(m.fs.eq_specific_energy_consumption, Constraint)
        assert isinstance(m.fs.rule_water_recovery, Constraint)
        assert isinstance(m.fs.water_recovery_ub, Constraint)
        assert isinstance(m.fs.water_recovery_lb, Constraint)
        for e in m.fs.set_evaporators:
            assert isinstance(m.fs.UA_term[e], Expression)
        assert isinstance(m.fs.total_water_produced_gpm, Expression)
        assert isinstance(m.fs.performance_ratio, Expression)

        # based on values at 60% water recovery from [2]
        assert m.fs.steam_generator.outlet.temperature.value == pytest.approx(
            69.1 + 273.15, rel=1e-3
        )
        assert m.fs.steam_generator.outlet.pressure.value == pytest.approx(
            30000, rel=1e-3
        )
        assert m.fs.total_water_produced_gpm == pytest.approx(1.489, rel=1e-3)
        assert m.fs.specific_energy_consumption.value == pytest.approx(297.84, rel=1e-3)
        assert m.fs.performance_ratio.value == pytest.approx(2.262, rel=1e-3)
