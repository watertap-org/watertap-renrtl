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

This is a closed loop 3MED-AHP model configuration. A break-point is placed between the absorber tube outlet and the pump inlet to 
account for the necessary concentration increase following the absorber, as no specific absorber unit model is available in the model libraries
The model uses experimental conditions from [2] and validates well at the listed water recoveries below for single electrolyte systems. 
"""
import logging
import pytest

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

module = __import__("3MED_AHP_eNRTL")

# Access the functions from the module
populate_enrtl_state_vars_single = module.populate_enrtl_state_vars_single
populate_enrtl_state_vars_multi = module.populate_enrtl_state_vars_multi
populate_enrtl_state_vars_gen = module.populate_enrtl_state_vars_gen
create_model = module.create_model
create_arcs = module.create_arcs
add_enrtl_method_single = module.add_enrtl_method_single
add_enrtl_method_multi = module.add_enrtl_method_multi
add_enrtl_method_AHP = module.add_enrtl_method_AHP
set_scaling = module.set_scaling
set_model_inputs = module.set_model_inputs
initialize = module.initialize
add_bounds = module.add_bounds
model_analysis = module.model_analysis

logging.basicConfig(level=logging.INFO)
logging.getLogger("pyomo.repn.plugins.nl_writer").setLevel(logging.ERROR)

# solve_nonideal gives the option to solve an ideal and nonideal case for the MED loop of the system
# solve_nonideal_AHP gives the option to solve an ideal and nonideal case for the AHP loop of the system
# If solve_nonideal is set to true, eNRTL is used to calculate the activity coefficients of solvent and solutes;
# when set to False, the model is solved assuming an ideal system with an activity coefficient of 1 for the solvent.

solve_nonideal = True  # 3MED loop
solve_nonideal_AHP = True  # AHP loop


class TestMED:
    @pytest.fixture(scope="class")
    def MED_AHP_eNRTL (self):
        m = create_model()
        return m
    @pytest.mark.unit
    def test_create_model(self, MED_AHP_eNRTL):
        m = MED_AHP_eNRTL
        # create_model(m)

        # test model set up
        assert isinstance(m, ConcreteModel)
        assert isinstance(m.fs, FlowsheetBlock)
        assert isinstance(m.fs.properties_vapor, props_w.WaterParameterBlock())
        assert isinstance(m.fs.properties_feed_sw, props_sw.SeawaterParameterBlock())
        assert isinstance(m.fs.properties_feed, props_libr.LiBrParameterBlock())

        # test unit models
        assert isinstance(m.fs.feed_sw, Feed)
        assert isinstance(m.fs.feed, Feed)
        assert isinstance(m.fs.generator, Evaporator)
        assert isinstance(m.fs.economizer, HeatExchanger)
        assert isinstance(m.fs.expansion_valve, PressureChanger)
        assert isinstance(m.fs.mixer, Mixer)
        assert isinstance(m.fs.absorber, HeatExchanger)
        assert isinstance(m.fs.pump, PressureChanger)
        assert isinstance(m.fs.evaporator, Evaporator)
        assert isinstance(m.fs.condenser, Condenser)
        assert isinstance(m.fs.separator, Separator)
        assert isinstance(m.fs.tblock, Translator)

        # additional constraints, variables, and expressions
        assert isinstance(m.fs.eq_flow_mass_comp_H2O, Constraint)
        assert isinstance(m.fs.eq_temperature, Constraint)
        assert isinstance(m.fs.eq_pressure, Constraint)

    @pytest.mark.unit
    def test_create_arcs(self, MED_AHP_eNRTL):
        m = MED_AHP_eNRTL
        create_arcs(m)

        arc_dict = {
            m.fs.pump_to_economizer: (m.fs.pump.outlet, m.fs.economizer.shell_inlet),
            m.fs.economizer_to_generator: (
                m.fs.economizer.shell_outlet,
                m.fs.generator.inlet_feed,
            ),
            m.fs.generator_to_economizer: (
                m.fs.generator.outlet_brine,
                m.fs.economizer.tube_inlet,
            ),
            m.fs.economizer_to_valve: (
                m.fs.economizer.tube_outlet,
                m.fs.expansion_valve.inlet,
            ),
            m.fs.valve_to_mixer: (m.fs.expansion_valve.outlet, m.fs.mixer.inlet_2),
            m.fs.mixer_to_absorber: (m.fs.mixer.outlet, m.fs.absorber.tube_inlet),
            m.fs.feed_to_absorber: (m.fs.feed_sw.outlet, m.fs.absorber.shell_inlet),
            m.fs.generator_to_condenser: (
                m.fs.generator.outlet_vapor,
                m.fs.condenser[1].inlet,
            ),
            m.fs.absorber_to_evaporator_feed: Arc(
                m.fs.absorber.shell_outlet, m.fs.evaporator[1].inlet_feed
            ),
            m.fs.evap1brine_to_evap2feed: Arc(
                m.fs.evaporator[1].outlet_brine, m.fs.evaporator[2].inlet_feed
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
            m.fs.evap3vapor_to_separator: (
                m.fs.evaporator[3].outlet_vapor,
                m.fs.separator.inlet,
            ),
            m.fs.separator_to_condenser: (
                m.fs.separator.outlet_2,
                m.fs.condenser[4].inlet,
            ),
            m.fs.separator_to_tblock: (m.fs.separator.outlet_1, m.fs.tblock.inlet),
            m.fs.tblock_to_mixer: (m.fs.tblock.outlet, m.fs.mixer.inlet_1),
        }
        for arc, port_tpl in arc_dict.items():
            assert arc.source is port_tpl[0]
            assert arc.destination is port_tpl[1]

        # units
        assert_units_consistent(m.fs)

    @pytest.mark.component
    def test_set_model_inputs(self, MED_AHP_eNRTL):
        m = MED_AHP_eNRTL
        set_model_inputs(m)

        # check fixed variables
        # Feed
        assert m.fs.feed_sw.properties[0].flow_mass_phase_comp["Liq", "H2O"].is_fixed()
        assert (
            value(m.fs.feed_sw.properties[0].flow_mass_phase_comp["Liq", "H2O"]) == 0.24
        )
        assert m.fs.feed_sw.properties[0].flow_mass_phase_comp["Liq", "TDS"].is_fixed()
        assert (
            value(m.fs.feed_sw.properties[0].flow_mass_phase_comp["Liq", "TDS"])
            == 0.0058
        )
        assert m.fs.feed_sw.properties[0].temperature.is_fixed()
        assert value(m.fs.feed_sw.properties[0].temperature) == 27 + 273.15
        assert m.fs.feed_sw.properties[0].pressure.is_fixed()
        assert value(m.fs.feed_sw.properties[0].pressure) == 101325

        # Inlet data for pump
        assert m.fs.pump.inlet.flow_mass_phase_comp[0, "Liq", "H2O"].is_fixed
        assert value(m.fs.pump.inlet.flow_mass_phase_comp[0, "Liq", "H2O"]) == 0.45
        assert m.fs.pump.inlet.flow_mass_phase_comp[0, "Liq", "TDS"].is_fixed
        assert value(m.fs.pump.inlet.flow_mass_phase_comp[0, "Liq", "TDS"]) == 0.55
        assert m.fs.pump.inlet.temperature.is_fixed()
        assert value(m.fs.pump.inlet.temperature) == 150 + 273.15
        assert m.fs.pump.inlet.pressure.is_fixed()
        assert value(m.fs.pump.inlet.pressure) == 10000

        assert m.fs.pump.deltaP.is_fixed()
        assert value(m.fs.pump.deltaP) == 2e3
        assert m.fs.pump.efficiency_pump.is_fixed()
        assert value(m.fs.pump.efficiency_pump) == 0.7

        # Inlet data for economizer
        assert m.fs.economizer.tube_inlet.flow_mass_phase_comp[
            0, "Liq", "H2O"
        ].is_fixed()
        assert (
            value(m.fs.economizer.tube_inlet.flow_mass_phase_comp[0, "Liq", "H2O"])
            == 0.35
        )
        assert m.fs.economizer.tube_inlet.flow_mass_phase_comp[
            0, "Liq", "TDS"
        ].is_fixed()
        assert (
            value(m.fs.economizer.tube_inlet.flow_mass_phase_comp[0, "Liq", "TDS"])
            == 0.65
        )
        assert m.fs.economizer.tube_inlet.temperature.is_fixed()
        assert value(m.fs.economizer.tube_inlet.temperature) == 200 + 273.15
        assert m.fs.economizer.tube_inlet.pressure.is_fixed()
        assert value(m.fs.economizer.tube_inlet.pressure) == 30000

        assert m.fs.economizer.area.is_fixed()
        assert value(m.fs.economizer.area) == 40
        assert m.fs.economizer.overall_heat_transfer_coefficient.is_fixed()
        assert value(m.fs.economizer.overall_heat_transfer_coefficient) == 600
        assert m.fs.economizer.crossflow_factor.is_fixed()
        assert value(m.fs.economizer.crossflow_factor) == 0.5

        # Inlet data for generator
        assert m.fs.generator.outlet_vapor.pressure[0].is_fixed()
        assert value(m.fs.generator.outlet_vapor.pressure[0]) == 30e3
        assert m.fs.generator.U.is_fixed()
        assert value(m.fs.generator.U) == 500
        assert m.fs.generator.area.is_fixed()
        assert value(m.fs.generator.area) == 10
        assert m.fs.generator.heat_transfer.is_fixed()
        assert value(m.fs.generator.heat_transfer) == 111e3
        assert m.fs.generator.delta_temperature_in.is_fixed()
        assert value(m.fs.generator.delta_temperature_in) == 10

        # Inlet data for expansion Valve
        assert m.fs.expansion_valve.deltaP.is_fixed()
        assert value(m.fs.expansion_valve.deltaP) == -20e3
        assert m.fs.expansion_valve.efficiency_pump.is_fixed()
        assert value(m.fs.expansion_valve.efficiency_pump) == 0.7

        # Inlet data for mixer
        assert m.fs.mixer.inlet_1.flow_mass_phase_comp[0, "Liq", "H2O"].is_fixed()
        assert value(m.fs.mixer.inlet_1.flow_mass_phase_comp[0, "Liq", "H2O"]) == 0.15
        assert m.fs.mixer.inlet_1.flow_mass_phase_comp[0, "Liq", "TDS"].is_fixed()
        assert value(m.fs.mixer.inlet_1.flow_mass_phase_comp[0, "Liq", "TDS"]) == 0
        assert m.fs.mixer.inlet_1.pressure.is_fixed()
        assert value(m.fs.mixer.inlet_1.pressure) == 31000
        assert m.fs.mixer.inlet_1.temperature.is_fixed()
        assert value(m.fs.mixer.inlet_1.temperature) == 65 + 273.15

        # Inlet data for absorber
        assert m.fs.absorber.overall_heat_transfer_coefficient.is_fixed()
        assert value(m.fs.absorber.overall_heat_transfer_coefficient) == 500
        assert m.fs.absorber.shell_outlet.temperature.is_fixed()
        assert value(m.fs.absorber.shell_outlet.temperature) == 75 + 273.15
        assert m.fs.absorber.crossflow_factor.is_fixed()
        assert value(m.fs.absorber.crossflow_factor) == 0.5

        # Inlet data for condenser[1]
        assert m.fs.condenser[1].outlet.temperature[0].is_fixed()
        assert value(m.fs.condenser[1].outlet.temperature[0]) == 51 + 273.15

        # Inlet data for Evaporator[1]
        assert m.fs.evaporator[1].outlet_brine.temperature[0].is_fixed()
        assert value(m.fs.evaporator[1].outlet_brine.temperature[0]) == 52 + 273.15
        assert m.fs.evaporator[1].U.is_fixed()
        assert value(m.fs.evaporator[1].U) == 1200
        assert m.fs.evaporator[1].area.is_fixed()
        assert value(m.fs.evaporator[1].area) == 10
        assert m.fs.evaporator[1].delta_temperature_in.is_fxied()
        assert value(m.fs.evaporator[1].delta_temperature_in) == 2
        assert m.fs.evaporator[1].delta_temperature_out.is_fixed()
        assert value(m.fs.evaporator[1].delta_temperature_out) == 2.5

        # Inlet data for Condenser[2]
        assert m.fs.condenser[2].outlet.temperature[0].is_fixed()
        assert value(m.fs.condenser[2].outlet.temperature[0]) == 53 + 273.15

        # Inlet data for Evaporator[2]
        assert m.fs.evaporator[2].U.is_fixed()
        assert value(m.fs.evaporator[2].U) == 1000
        assert m.fs.evaporator[2].area.is_fixed()
        assert value(m.fs.evaporator[2].area) == 30
        assert m.fs.evaporator[2].outlet_brine.temperature[0].is_fixed()
        assert value(m.fs.evaporator[2].outlet_brine.temperature[0]) == 55 + 273.15
        assert m.fs.evaporator[2].delta_temperature_in.is_fixed()
        assert value(m.fs.evaporator[2].delta_temperature_in) == 8
        assert m.fs.evaporator[2].delta_temperature_out.is_fixed()
        assert value(m.fs.evaporator[2].delta_temperature_out) == 2.5

        # Inlet data for Condenser[3]
        assert m.fs.condenser[3].outlet.temperature[0].is_fixed()
        assert value(m.fs.condenser[3].outlet.temperature[0]) == 58 + 273.15

        # Inlet data for Evaporator[3]
        assert m.fs.evaporator[3].U.is_fixed()
        assert value(m.fs.evaporator[3].U) == 1000
        assert m.fs.evaporator[3].area.is_fixed()
        assert value(m.fs.evaporator[3].area) == 20
        assert m.fs.evaporator[3].outlet_brine.temperature[0].is_fixed()
        assert value(m.fs.evaporator[3].outlet_brine.temperature[0]) == 65 + 273.15
        assert m.fs.evaporator[3].delta_temperature_in.is_fixed()
        assert value(m.fs.evaporator[3].delta_temperature_in) == 10
        assert m.fs.evaporator[3].delta_temperature_out.is_fixed()
        assert value(m.fs.evaporator[3].delta_temperature_out) == 2.5

        # Inlet data for separator
        assert m.fs.separator.split_fraction[0, "outlet_1"].is_fixed()
        assert value(m.fs.separator.split_fraction[0, "outlet_1"]) == 0.5
        assert (
            m.fs.separator.outlet_1_state[0.0]
            .flow_mass_phase_comp["Liq", "H2O"]
            .is_fixed()
        )
        assert (
            value(m.fs.separator.outlet_1_state[0.0].flow_mass_phase_comp["Liq", "H2O"])
            == 0
        )
        assert (
            m.fs.separator.outlet_2_state[0.0]
            .flow_mass_phase_comp["Liq", "H2O"]
            .is_fixed()
        )
        assert (
            value(m.fs.separator.outlet_2_state[0.0].flow_mass_phase_comp["Liq", "H2O"])
            == 0
        )

        # Inlet data for Condenser[4]
        assert m.fs.condenser[4].outlet.temperature[0].is_fixed()
        assert value(m.fs.condenser[4].outlet.temperature[0]) == 68 + 273.15

        # Inlet data for Translator block
        assert (
            m.fs.tblock.properties_out[0].flow_mass_phase_comp["Liq", "TDS"].is_fixed()
        )
        assert (
            value(m.fs.tblock.properties_out[0].flow_mass_phase_comp["Liq", "TDS"]) == 0
        )

    @pytest.mark.component
    @pytest.mark.requires_idaes_solver
    def test_initialize(self, MED_AHP_eNRTL):
        m = MED_AHP_eNRTL
        initialize(m)

        assert value(m.fs.absorber.overall_heat_transfer_coefficient) == pytest.approx(
            500, rel=1e3
        )
        assert value(m.fs.absorber.shell_outlet.temperature) == pytest.approx(
            75 + 273.15, rel=1e3
        )
        assert value(
            m.fs.economizer.overall_heat_transfer_coefficient
        ) == pytest.approx(600, rel=1e3)
        assert value(m.fs.generator.outlet_vapor.pressure[0]) == pytest.approx(
            30000, rel=1e3
        )
        assert value(m.fs.generator.U) == pytest.approx(500, rel=1e3)
        assert value(m.fs.generator.heat_transfer) == pytest.approx(111e3, rel=1e3)
        assert value(m.fs.evaporator[1].U) == pytest.approx(1200, rel=1e3)
        assert value(m.fs.evaporator[1].delta_temperature_out) == pytest.approx(
            2.5, rel=1e3
        )
        assert value(m.fs.evaporator[2].U) == pytest.approx(1000, 1e3)
        assert value(m.fs.evaporator[2].delta_temperature_out) == pytest.approx(
            2.5, rel=1e3
        )
        assert value(m.fs.evaporator[3].U) == pytest.approx(1000, 1e3)
        assert value(m.fs.evaporator[3].delta_temperature_out) == pytest.approx(
            2.5, rel=1e3
        )

        assert degrees_of_freedom(m) == 0

    @pytest.mark.component
    @pytest.mark.requires_idaes_solver
    def test_model_analysis(self, MED_AHP_eNRTL):
        m = MED_AHP_eNRTL
        model_analysis(m)

        solver = get_solver()
        initialize(m, solver=solver)

        results = solver.solve(m, tee=False)
        assert_optimal_termination(results)

        # additional constraints, variables, and expressions
        for e in m.fs.set_evaporators:
            assert isinstance(
                m.fs.eq_upper_bound_evaporators_delta_temprature_in[e], Constraint
            )
        assert isinstance(
            m.fs.eq_upper_bound_generator_delta_temperature_in, Constraint
        )
        assert isinstance(
            m.fs.eq_upper_bound_generator_delta_temprature_out, Constraint
        )
        for e in m.fs.set2_evaporators:
            assert isinstance(m.fs.eq_upper_bound_evaporators_pressure[e], Constraint)
        assert isinstance(m.fs.gen_area_upper_bound, Constraint)
        assert isinstance(m.fs.econ_area_upper_bound, Constraint)
        assert isinstance(m.fs.abs_area_upper_bound, Constraint)
        assert isinstance(m.fs.eq_specific_energy_consumption, Constraint)
        assert isinstance(m.fs.rule_water_recovery, Constraint)
        assert isinstance(m.fs.water_recovery_ub, Constraint)
        assert isinstance(m.fs.water_recovery_lb, Constraint)
        for e in m.fs.set_evaporators:
            assert isinstance(m.fs.UA_term[e], Expression)
        assert isinstance(m.fs.UA_term_gen, Expression)
        assert isinstance(m.fs.total_water_produced_gpm, Expression)
        assert isinstance(m.fs.performance_ratio, Expression)

        # based on estimated values at 70% water recovery from [2]
        assert m.fs.generator.outlet.pressure.value == pytest.approx(30000, rel=1e-3)
        assert m.fs.specific_energy_consumption.value == pytest.approx(205.00, rel=1e-3)
        assert m.fs.performance_ratio.value == pytest.approx(3.4, rel=1e-3)