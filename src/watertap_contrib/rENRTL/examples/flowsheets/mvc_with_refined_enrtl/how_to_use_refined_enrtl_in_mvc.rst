.. _how_to_use_refined_enrtl_in_mvc:

How to use refined electrolyte NRTL in Mechanical Vapor Compression
===================================================================

To document how to use the refined electrolyte non-random two-liquid (r-eNRTL) method, a three-effect mechanical vapor compression (MVC) system is used. The single-effect is taken from WaterTAP unit model library. To demonstrate how to integrate this thermodynamic method, an example is presented where a flow of seawater is fed to the first evaporator unit in the series to be separated into fresh water and a concentrated brine solution. The evaporator units are equivalent to the evaporator example in :ref:`how_to_setup_evaporator_with_enrtl`. In this multi-effect system, fresh water is separated as vapor at the top outlet of the last evaporator, while a concentrated brine is separated as the bottom product as liquid. Since the `Seawater property package` is used to calculate the seawater properties, the seawater feed is assumed to contain a fixed amount of total dissolved solids (TDS) in the water.  

To consider the non-ideality equilibrium in the evaporator units, the equilibrium equation in each evaporator unit is replaced by the following expression:

.. math:: P = x_{\rm H_2O} \gamma_{\rm H_2O}P^{\rm sat}_{\rm H_2O}

where :math:`P` is the pressure of the system, :math:`x_{\rm H_2O}` is the mole fraction of water, :math:`P^{\rm sat}_{\rm H_2O}` is the saturation pressure of water, and :math:`\gamma_{\rm H_2O}` is the activity coefficient of water that is calculated using the r-eNRTL. In this example, we assume that the seawater has a single solvent, :math:`\rm H_2O`, and a single solute, :math:`\rm NaCl`. The solute dissociates in the water as given in the equation below:

.. math:: \rm NaCl \rightarrow \rm Na^{+} + \rm Cl^{-} 

To use the r-eNRTL, a configuration dictionary is needed and imported at the top of the model. Refer to :ref:`how_to_setup_refined_enrtl` to learn more details about how to set it up. To build a three-effect MVC system, we followed the steps given below:

1) Import necessary Pyomo, IDAES, and WaterTAP components, libraries, and property packages.

2) Import the r-eNRTL configuration dictionary. This configuration dictionary includes all the parameters and assumptions needed by the r-eNRTL method. More details about how to set it up, refer to :ref:`how_to_setup_refined_enrtl`.

3) Create function to declare initial data needed by the thermodynamic method.
   
4) Create main function to include the concrete model, flowsheet, and property parameter blocks. Declare the `Seawater property package` for the feed (and liquid outlet stream) and the `Water` property package for the top product (or vapor stream). Also, create a set to determine the number of evaporators in the flowsheet and add all the units involved in the multi-effect MVC system such as evaporators, condensers, and a compressor.

5) Declare a new block to include the generic properties needed by the r-eNRTL and construct a `StateBlock` associated with it. Once it is ready, create an instance of the `GenericParameterBlock` component and provide the r-eNRTL configuration dictionary imported in Step 2. For each evaporator unit, populate the state block by calling the function created in Step 3, which contains initial values for temperature, pressure, and mass of solvent and solute ions. Once the state block is ready, add constraints to link the temperature, pressure, and mass flowrate of the evaporator brine with the r-eNRTL state block. Note that, since the `Seawater property package` is in terms of TDS, not ions, the mass ratio of each ion in the solute is calculated and used to make the conversion from TDS to solute ions in the flow constraints. 

6) Include new equilibrium equation that includes the activity coefficient of water, :math:`\gamma_{\rm H_2O}`, calculated using the r-eNRTL method. To include this equation, the existing equilibrium equation from the evaporator unit is deactivated and replaced with the new equation. 

7) Create arcs to connect the units in the flowsheet. To generate the state variables constraints, expand these arcs using Pyomo `TransformationFactory`.
   
8) Set and calculate scaling factors for relevant variables in the flowsheet.

9) Fix model inputs to have a square problem (i.e., zero degrees of freedom) and initialize the multi-effect MVC system.

11) Unfix variables that were fixed during initialization and add bounds to relevant variables in the system.

12) Include new constraints and solve simulation example.

The example code that follows these steps is provided below:

Example code:
^^^^^^^^^^^^^

.. code-block::

   ##### STEP 1
   import logging
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
   import watertap.property_models.seawater_prop_pack as props_sw
   import watertap.property_models.water_prop_pack as props_w
   from watertap.unit_models.mvc.components import Evaporator, Compressor, Condenser
   logging.basicConfig(level=logging.INFO)
   logging.getLogger("pyomo.repn.plugins.nl_writer").setLevel(logging.ERROR)
   
   """ References:
   [1] Matthew D. Stuber, Christopher Sullivan, Spencer A. Kirk, Jennifer
   A. Farrand, Philip V. Schillai, Brian D. Fojtasek, and Aaron
   H. Mandell. Pilot demonstration of concentrated solar-poweered
   desalination of subsurface agricultural drainage water and other
   brackish groundwater sources. Desalination, 355 (2015), 186-196.
   """

   ##### STEP 2
   import entrl_config_FpcTP

   ##### STEP 3
   def populate_enrtl_state_vars(blk, base="FpcTP"):
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

   def main():

        ##### STEP 4
	m = ConcreteModel("Three-effect Mechanical Vapor Compression Model")
	m.fs = FlowsheetBlock(dynamic=False)
	m.fs.properties_vapor = props_w.WaterParameterBlock()
	m.fs.properties_feed = props_sw.SeawaterParameterBlock()
	m.fs.set_ions = Set(initialize=["Na+", "Cl-"])
	m.fs.ion_coeff = {"Na+": 1, "Cl-": 1}

	m.fs.num_evaporators = 3
	m.fs.set_evaporators = RangeSet(m.fs.num_evaporators)
	m.fs.evaporator = Evaporator(
           m.fs.set_evaporators,
           property_package_feed=m.fs.properties_feed,
           property_package_vapor=m.fs.properties_vapor,
	)
	m.fs.compressor = Compressor(property_package=m.fs.properties_vapor)
	m.fs.condenser = Condenser(
           m.fs.set_evaporators,
	   property_package=m.fs.properties_vapor
        )

        ##### STEP 5
	m.fs.enrtl_state = Block(m.fs.set_evaporators)
	m.fs.prop_enrtl = GenericParameterBlock(**entrl_config_FpcTP.configuration)
	for n_evap in m.fs.set_evaporators:
	   m.fs.enrtl_state[n_evap].properties = m.fs.prop_enrtl.build_state_block([0])
           sb_enrtl = m.fs.enrtl_state[n_evap].properties[0]  # just renaming the block
           populate_enrtl_state_vars(sb_enrtl, base="FpcTP")

	   m.fs.enrtl_state[n_evap].mol_mass_ion_molecule = sum(
               m.fs.ion_coeff[j] * sb_enrtl.mw_comp[j] for j in m.fs.set_ions
           )
           m.fs.enrtl_state[n_evap].mass_ratio_ion = {
               "Na+": sb_enrtl.mw_comp["Na+"] / m.fs.enrtl_state[n_evap].mol_mass_ion_molecule,
               "Cl-": sb_enrtl.mw_comp["Cl-"] / m.fs.enrtl_state[n_evap].mol_mass_ion_molecule,
           }
        
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
                   == m.fs.evaporator[n_evap].properties_brine[0].flow_mass_phase_comp["Liq", "H2O"]
               )
           )

           def enrtl_flow_mass_ion_comp(b, j):
               return sb_enrtl.flow_mass_phase_comp["Liq", j] == (
                   (
                       m.fs.evaporator[n_evap].properties_brine[0].flow_mass_phase_comp["Liq", "TDS"]
                       * b.mass_ratio_ion[j]
                   )
               )
           m.fs.enrtl_state[n_evap].enrtl_flow_mass_ion_comp_eq = Constraint(
               m.fs.set_ions,
               rule=enrtl_flow_mass_ion_comp
           )

        ##### STEP 6
	for e in m.fs.set_evaporators:
	   m.fs.evaporator[e].eq_brine_pressure.deactivate()
	@m.fs.Constraint(m.fs.set_evaporators, doc="Vapor-liquid equilibrium equation")
	def _eq_phase_equilibrium(b, e):
	   return b.evaporator[e].properties_brine[0].pressure == (
               m.fs.enrtl_state[e].properties[0].act_coeff_phase_comp["Liq", "H2O"]
               * b.evaporator[e].properties_brine[0].mole_frac_phase_comp["Liq", "H2O"]
               * b.evaporator[e].properties_vapor[0].pressure_sat
           )

        ##### STEP 7
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
       TransformationFactory("network.expand_arcs").apply_to(m)


       ##### STEP 8
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
       for e in m.fs.set_evaporators:
           iscale.set_scaling_factor(m.fs.evaporator[e].area, 1e-3)
           iscale.set_scaling_factor(m.fs.evaporator[e].U, 1e-3)
           iscale.set_scaling_factor(m.fs.evaporator[e].delta_temperature_in, 1e-1)
           iscale.set_scaling_factor(m.fs.evaporator[e].delta_temperature_out, 1e-1)
           iscale.set_scaling_factor(m.fs.evaporator[e].lmtd, 1e-1)
           iscale.set_scaling_factor(m.fs.evaporator[e].heat_transfer, 1e-6)
           iscale.set_scaling_factor(m.fs.condenser[e].control_volume.heat, 1e-6)
       iscale.set_scaling_factor(m.fs.compressor.control_volume.work, 1e-6)
       iscale.calculate_scaling_factors(m)

       ##### STEP 9
       m.fs.evaporator[1].inlet_feed.flow_mass_phase_comp[0, "Liq", "H2O"].fix(9.65)
       m.fs.evaporator[1].inlet_feed.flow_mass_phase_comp[0, "Liq", "TDS"].fix(0.35)
       m.fs.evaporator[1].inlet_feed.temperature[0].fix(50.52 + 273.15)
       m.fs.evaporator[1].inlet_feed.pressure[0].fix(101325)
       m.fs.evaporator[1].outlet_brine.temperature[0].fix(54 + 273.15)
       m.fs.evaporator[1].U.fix(1e3)
       m.fs.evaporator[1].area.fix(800)
       m.fs.evaporator[1].delta_temperature_in.fix(10)
       m.fs.evaporator[1].delta_temperature_out.fix(5)
       m.fs.evaporator[2].U.fix(1e3)
       m.fs.evaporator[2].area.fix(500)
       m.fs.evaporator[2].outlet_brine.temperature[0].fix(55 + 273.15)
       m.fs.evaporator[2].delta_temperature_in.fix(10)
       m.fs.evaporator[2].delta_temperature_out.fix(5)
       m.fs.evaporator[3].U.fix(1e3)
       m.fs.evaporator[3].area.fix(500)
       m.fs.evaporator[3].outlet_brine.temperature[0].fix(60 + 273.15)
       m.fs.evaporator[3].delta_temperature_in.fix(10)
       m.fs.evaporator[3].delta_temperature_out.fix(5)
       m.fs.compressor.pressure_ratio.fix(1.8)
       m.fs.compressor.efficiency.fix(0.8)
       m.fs.condenser[1].control_volume.heat[0].fix(-3605623)
       m.fs.condenser[2].control_volume.heat[0].fix(-5768998)
       m.fs.condenser[3].control_volume.heat[0].fix(-3605623)

       outlvl=idaeslog.WARNING
       m.fs.evaporator[1].initialize(outlvl=outlvl)
       propagate_state(m.fs.evap1vapor_to_cond2)
       m.fs.condenser[2].initialize_build(heat=-m.fs.evaporator[1].heat_transfer.value)
       propagate_state(m.fs.evap1brine_to_evap2feed)
       m.fs.evaporator[2].initialize(outlvl=outlvl)
       propagate_state(m.fs.evap2vapor_to_cond3)
       m.fs.condenser[3].initialize_build(heat=-m.fs.evaporator[2].heat_transfer.value)
       propagate_state(m.fs.evap2brine_to_evap3feed)
       m.fs.evaporator[3].initialize(outlvl=outlvl)
       propagate_state(m.fs.evap3vapor_to_compressor)
       m.fs.compressor.initialize(outlvl=outlvl)
       propagate_state(m.fs.compressor_to_cond1)
       m.fs.condenser[1].initialize_build(heat=-m.fs.evaporator[3].heat_transfer.value)

       solver = get_solver()
       init_results = solver.solve(m, tee=False)

       ##### STEP 11
       for i in m.fs.set_evaporators:
           m.fs.evaporator[i].area.unfix()
           m.fs.evaporator[i].U.unfix()
           m.fs.evaporator[i].outlet_brine.temperature[0].unfix()
           m.fs.evaporator[i].delta_temperature_in.unfix()
           m.fs.evaporator[i].delta_temperature_out.unfix()
           m.fs.condenser[i].control_volume.heat[0].unfix()
       m.fs.compressor.control_volume.work.unfix()
       m.fs.compressor.pressure_ratio.unfix()
       for i in m.fs.set_evaporators:
           m.fs.evaporator[i].area.setlb(10)
           m.fs.evaporator[i].area.setub(None)
           m.fs.evaporator[i].outlet_brine.temperature[0].setub(73 + 273.15)
           # Bounds from ref[1]
           m.fs.evaporator[i].U.setlb(500)
           m.fs.evaporator[i].U.setub(2500)

       ##### STEP 12
       @m.fs.Constraint()
       def eq_upper_bound_compression_ratio(b):
           return b.compressor.pressure_ratio <= 2 * pyunits.dimensionless
       @m.fs.Constraint(m.fs.set_evaporators)
       def eq_upper_bound_evaporators_delta_temperature_in(b, e):
           return b.evaporator[e].delta_temperature_in == 3 * pyunits.K
       @m.fs.Constraint(m.fs.set_evaporators)
       def eq_upper_bound_evaporators_delta_temprature_out(b, e):
           return b.evaporator[e].delta_temperature_out <= 10 * pyunits.K

       m.fs.set2_evaporators = RangeSet(m.fs.num_evaporators - 1)
       @m.fs.Constraint(m.fs.set2_evaporators)
       def eq_upper_bound_evaporators_pressure(b, e):
           return (
               b.evaporator[e + 1].outlet_brine.pressure[0]
               <= b.evaporator[e].outlet_brine.pressure[0]
           )
       @m.fs.Expression(
           m.fs.set_evaporators, doc="Overall heat trasfer coefficient and area term"
       )
       def UA_term(b, e):
           return b.evaporator[e].area * b.evaporator[e].U
       m.fs.water_density = pyo.Param(initialize=1000, units=pyunits.kg / pyunits.m**3)
       @m.fs.Expression()
       def total_water_produced_gpm(b):
           return pyo.units.convert(
               sum(
                   b.condenser[e].control_volume.properties_out[0].flow_mass_phase_comp["Liq", "H2O"]
                   for e in m.fs.set_evaporators
		   ) / m.fs.water_density,
	       to_units=pyunits.gallon / pyunits.minute,
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
                       b.condenser[e].control_volume.properties_out[0].flow_mass_phase_comp["Liq", "H2O"]
                       for e in m.fs.set_evaporators
                       )
		     )
		     / m.fs.evaporator[1].inlet_feed.flow_mass_phase_comp[0, "Liq", "H2O"]
		 )

       results = solver.solve(m, tee=False)

      print("Energy input (kW): {:0.4f}".format(
               value(m.fs.compressor.control_volume.work[0] * (1e-3)))
      )

    # Call main function
    if __name__ == "__main__":
        main()

