.. _how_to_setup_evaporator_with_enrtl:

How to setup an evaporator model using electrolyte NRTL
=======================================================

To document how to setup an evaporator model to use the electrolyte non-random two-liquid (eNRTL) method, the evaporator unit model from WaterTAP is used. To demonstrate how to integrate this thermodynamic method, we present an example where a flow of seawater is fed to the evaporator to be separated into fresh water and a concentrated brine solution. This separation is possible since the evaporator unit allows the use of two property packages: one for the feed (and liquid phase) and one for the vapor phase. Fresh water is separated as vapor at the top outlet of the evaporator, while a concentrated brine is separated at the bottom of the evaporator as liquid. Since the `Seawater property package` is used to calculate the seawater properties, the seawater feed is assumed to contain a fixed amount of total dissolved solids (TDS) in the water.  

To use the eNRTL method in the evaporator model, the equilibrium equation in the evaporator unit model is replaced by the following expression:

.. math:: P = x_{\rm H_2O} \gamma_{\rm H_2O}P^{\rm sat}_{\rm H_2O}

where :math:`P` is the pressure of the system, :math:`x_{\rm H_2O}` is the mole fraction of water, :math:`P^{\rm sat}_{\rm H_2O}` is the saturation pressure of water, and :math:`\gamma_{\rm H_2O}` is the activity coefficient of water that is calculated using eNRTL. In this example, we assume that the seawater has a single solvent, :math:`\rm H_2O`, and a single solute, :math:`\rm NaCl`. The solute is dissolved in the water as shown in the equation below:

.. math:: \rm NaCl \rightarrow \rm Na^{+} + \rm Cl^{-} 

To setup the evaporator unit model with eNRTL, we follow the steps below:

1) Import necessary Pyomo, IDAES, and WaterTAP components, libraries, and property packages.
   
2) Create a Pyomo concrete model, flowsheet, and property parameter blocks. Once created, add evaporator unit into the flowsheet and include the `Seawater property package` for the feed (and liquid outlet stream) and the `Water` property package for the vapor stream.
   
3) Declare a configuration dictionary that includes all the parameters, equations, and assumptions needed by the eNRTL method. For this example, a symmetric reference state is considered and the binary interaction energy parameters :math:`\tau` are given. Once it is ready, create an instance of the `GenericParameterBlock` component and provide the eNRTL configuration dictionary.

4) Declare a new block to include the generic properties needed by eNRTL and construct a `StateBlock` associated with it. Populate the state block with initial values for temperature, pressure, and mass of solvent and solute ions.

5) Link the temperature, pressure, and mass flowrate of the evaporator brine with the eNRTL state block. Since the `Seawater property package` is in terms of TDS, not ions, the mass ratio of each ion in the solute is calculated and used to make the conversion from TDS to solute ions in the flow constraints.  

6) Include new equilibrium equation that includes the activity coefficient of water, :math:`\gamma_{\rm H_2O}`, that is calculated using the eNRTL method. To include this equation, the existing equilibrium equation from the evaporator unit is deactivated and replaced with the new equation. 

7) Set and calculate the scaling factors for properties and all the variables in the evaporator.

8) Fix model inputs to have a square problem (i.e., zero degrees of freedom).

9) Initialize the evaporator unit.

10) Solve simulation model as a square problem.

The example code that follows these steps is provided below:

Example code:
^^^^^^^^^^^^^

.. code-block::

   ##### STEP 1
   import pyomo.environ as pyo
   from pyomo.environ import units as pyunits
   from pyomo.environ import (ConcreteModel, TransformationFactory, Block, Constraint, minimize, Param, value, Set)
   import idaes.core.util.scaling as iscale
   import idaes.logger as idaeslog
   from idaes.core import FlowsheetBlock
   from idaes.models.properties.modular_properties.base.generic_property import GenericParameterBlock
   from idaes.core.solvers import get_solver
   from idaes.core.util.model_statistics import degrees_of_freedom
   from idaes.core import AqueousPhase, Solvent, Apparent, Anion, Cation
   from idaes.models.properties.modular_properties.eos.enrtl import ENRTL
   from idaes.models.properties.modular_properties.eos.enrtl_reference_states import Symmetric
   from idaes.models.properties.modular_properties.base.generic_property import StateIndex
   from idaes.models.properties.modular_properties.state_definitions import FpcTP
   from idaes.models.properties.modular_properties.pure.electrolyte import relative_permittivity_constant
   from watertap.unit_models.mvc.components import Evaporator
   import watertap.property_models.seawater_prop_pack as props_sw
   import watertap.property_models.water_prop_pack as props_w

   # Reference [1] Local Composition Model for Excess Gibbs Energy of Electrolyte
   # Systems, Pt 1.  Chen, C.-C., Britt, H.I., Boston, J.F., Evans, L.B.,
   # AIChE Journal, 1982, Vol. 28(4), pgs. 588-596

   def main():

        ##### STEP 2
        m = ConcreteModel("Evaporator with eNRTL")   
        m.fs = FlowsheetBlock(dynamic=False)
        m.fs.properties_vapor = props_w.WaterParameterBlock()
        m.fs.properties_feed = props_sw.SeawaterParameterBlock()
        m.fs.evaporator = Evaporator(property_package_feed=m.fs.properties_feed,
                                     property_package_vapor=m.fs.properties_vapor)

        ##### STEP 3
        class ConstantVolMol:
            def build_parameters(b):
                b.vol_mol_pure = Param(initialize=18e-6,
                                       units=pyunits.m**3 / pyunits.mol,
                                       mutable=True)
            def return_expression(b, cobj, T):
                return cobj.vol_mol_pure
	configuration = {"components": {"H2O": {"type": Solvent,
                                                "vol_mol_liq_comp": ConstantVolMol,
                                                "relative_permittivity_liq_comp": relative_permittivity_constant,
                                                "parameter_data": {"mw": (18.01528e-3, pyunits.kg/pyunits.mol),
                                                                   "relative_permittivity_liq_comp": 78.54}},
                                        "NaCl": {"type": Apparent,
                                                 "dissociation_species": {"Na+": 1, "Cl-": 1}},
                                        "Na+": {"type": Cation,
                                                "charge": +1,
                                                "parameter_data": {"mw": (22.990e-3, pyunits.kg/pyunits.mol)}},
                                        "Cl-": {"type": Anion,
                                                "charge": -1,
                                                "parameter_data": {"mw": (35.453e-3, pyunits.kg/pyunits.mol)}}},
                         "phases": {"Liq": {"type": AqueousPhase,
                                            "equation_of_state": ENRTL,
                                            "equation_of_state_options": {"reference_state": Symmetric}}},
                         "base_units": {"time": pyunits.s,
                                        "length": pyunits.m,
                                        "mass": pyunits.kg,
                                        "amount": pyunits.mol,
                                        "temperature": pyunits.K},
                         "state_definition": FpcTP,
                         "state_components": StateIndex.true,
                         "pressure_ref": 101325,
                         "temperature_ref": 298.15,
                         # From reference [1]
                         "parameter_data": {"Liq_tau": {("H2O", "Na+, Cl-"): 8.885,
                                                        ("Na+, Cl-", "H2O"): -4.549}},
                         "default_scaling_factors": {
                             ("flow_mol_phase_comp", ("Liq", "Na+")): 1e1,
                             ("flow_mol_phase_comp", ("Liq", "Cl-")): 1e1,
                             ("flow_mol_phase_comp", ("Liq", "H2O")): 1e-1,
                             ("mole_frac_comp", "Na+"): 1e2,
                             ("mole_frac_comp", "Cl-"): 1e2,
                             ("mole_frac_comp", "H2O"): 1,
                             ("mole_frac_phase_comp", ("Liq", "Na+")): 1e2,
                             ("mole_frac_phase_comp", ("Liq", "Cl-")): 1e2,
                             ("mole_frac_phase_comp", ("Liq", "H2O")): 1,
                             ("flow_mol_phase_comp_apparent", ("Liq", "NaCl")): 1e1,
                             ("flow_mol_phase_comp_apparent", ("Liq", "H2O")): 1e-1,
                             ("mole_frac_phase_comp_apparent", ("Liq", "NaCl")): 1e3,
                             ("mole_frac_phase_comp_apparent", ("Liq", "H2O")): 1
                         }
        }
        m.fs.prop_enrtl = GenericParameterBlock(**configuration)

	##### STEP 4
        m.fs.enrtl_state = Block()
        m.fs.enrtl_state.properties = m.fs.prop_enrtl.build_state_block([0])
        sb_enrtl = m.fs.enrtl_state.properties[0]

        def populate_enrtl_state_vars(blk, base="FpcTP"):
            blk.temperature = 298.15
            blk.pressure = 101325

            if base == "FpcTP":
                feed_flow_mass = 10  # kg/s
                feed_mass_frac_comp = {"Na+": 0.013768116, "Cl-": 0.021231884}
                feed_mass_frac_comp["H2O"] = (1 - sum(x for x in feed_mass_frac_comp.values()))
                mw_comp = {"H2O": 18.015e-3, "Na+": 22.990e-3, "Cl-": 35.453e-3}
                for j in feed_mass_frac_comp:
                    blk.flow_mol_phase_comp["Liq", j] = feed_flow_mass * feed_mass_frac_comp[j] / mw_comp[j]
                    if j == "H2O":
                        blk.flow_mol_phase_comp["Liq", j] /= 2

        populate_enrtl_state_vars(sb_enrtl, base="FpcTP")

        ##### STEP 5
        m.fs.set_ions = Set(initialize=["Na+", "Cl-"])
        m.fs.ion_coeff = {"Na+": 1, "Cl-": 1}
        m.fs.enrtl_state.mol_mass_ion_molecule = sum(m.fs.ion_coeff[j] * sb_enrtl.mw_comp[j]
                                                     for j in m.fs.set_ions)
        m.fs.enrtl_state.mass_ratio_ion = {
            "Na+": sb_enrtl.mw_comp["Na+"] / m.fs.enrtl_state.mol_mass_ion_molecule,
            "Cl-": sb_enrtl.mw_comp["Cl-"] / m.fs.enrtl_state.mol_mass_ion_molecule
        }

        @m.fs.enrtl_state.Constraint()
        def eq_enrtl_temperature(b):
            return b.properties[0].temperature == m.fs.evaporator.properties_brine[0].temperature
        @m.fs.enrtl_state.Constraint()
        def eq_enrtl_pressure(b):
            return b.properties[0].pressure == m.fs.evaporator.properties_brine[0].pressure
        @m.fs.enrtl_state.Constraint()
        def eq_enrtl_flow_mass_H2O(b):
            return b.properties[0].flow_mass_phase_comp["Liq", "H2O"] ==
                m.fs.evaporator.properties_brine[0].flow_mass_phase_comp["Liq", "H2O"]
        @m.fs.enrtl_state.Constraint(m.fs.set_ions)
        def eq_enrtl_flow_mass_ion_comp(b, j):
            flow_mass_in = b.properties[0].flow_mass_phase_comp["Liq", j]
            return flow_mass_in == (m.fs.evaporator.properties_brine[0].flow_mass_phase_comp["Liq", "TDS"] * b.mass_ratio_ion[j])

        ##### STEP 6
        m.fs.evaporator.eq_brine_pressure.deactivate()
        @m.fs.Constraint(doc="Vapor-liquid equilibrium equation")
        def _eq_phase_equilibrium(b):
            return b.evaporator.properties_brine[0].pressure == (
                b.enrtl_state.properties[0].act_coeff_phase_comp["Liq", "H2O"] *
                b.evaporator.properties_brine[0].mole_frac_phase_comp["Liq", "H2O"] *
                b.evaporator.properties_vapor[0].pressure_sat
            )

        ##### STEP 7
        m.fs.properties_feed.set_default_scaling("flow_mass_phase_comp", 1, index=("Liq", "H2O"))
        m.fs.properties_feed.set_default_scaling("flow_mass_phase_comp", 1e2, index=("Liq", "TDS"))
        m.fs.properties_vapor.set_default_scaling("flow_mass_phase_comp", 1, index=("Vap", "H2O"))
        m.fs.properties_vapor.set_default_scaling("flow_mass_phase_comp", 1, index=("Liq", "H2O"))
        iscale.set_scaling_factor(m.fs.evaporator.area, 1e-3)
        iscale.set_scaling_factor(m.fs.evaporator.U, 1e-3)
        iscale.set_scaling_factor(m.fs.evaporator.delta_temperature_in, 1e-1)
        iscale.set_scaling_factor(m.fs.evaporator.delta_temperature_out, 1e-1)
        iscale.set_scaling_factor(m.fs.evaporator.lmtd, 1e-1)
        iscale.set_scaling_factor(m.fs.evaporator.heat_transfer, 1e-6)
        iscale.calculate_scaling_factors(m)

        ##### STEP 8
        m.fs.evaporator.inlet_feed.flow_mass_phase_comp[0, "Liq", "H2O"].fix(9.65) # kg/s
        m.fs.evaporator.inlet_feed.flow_mass_phase_comp[0, "Liq", "TDS"].fix(0.35) # kg/s
        m.fs.evaporator.inlet_feed.temperature[0].fix(273.15 + 50.52)  # K
        m.fs.evaporator.inlet_feed.pressure[0].fix(101325)  # Pa
        m.fs.evaporator.outlet_brine.temperature[0].fix(273.15 + 60) # K
        m.fs.evaporator.U.fix(1e3)  # W/K-m^2
        m.fs.evaporator.area.fix(400)  # m^2
        m.fs.evaporator.delta_temperature_in.fix(30) # K
        m.fs.evaporator.delta_temperature_out.fix(5) # K

        ##### STEP 9
        m.fs.evaporator.initialize(outlvl=idaeslog.WARNING)

	#### STEP 10
        solver = get_solver()
        results = solver.solve(m, tee=True)

    # Call main function
    if __name__ == "__main__":
        main()

