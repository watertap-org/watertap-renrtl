.. _how_to_setup_multielectrolytes_refined_enrtl:

How to setup multielectrolytes refined electrolyte NRTL
=====================================

The thermodynamic properties of a mixture depend on the interactions between the species of the mixture. The multielectrolytes refined electrolyte non-random two-liquid (r-eNRTL) method expresses the excess Gibbs free energy of the multielectrolyte solution as a function of long-range electrostatic contributions that account for ion-ion interactions and short-range interactions that account for local, ion-ion, ion-molecule, and molecule-molecule interactions. The long-range interactions are modeled using a thermodynamically consistent extension of the Debye-Huckel (DH) equation, while the short-range interactions of hydrated ions with each other and with solvent molecules are described using the NRTL model.

Hydration is included in the NRTL effective mole fractions to account for the changes in composition of the local neighborhoods. The effect of ions on the structure and mobility of water molecules is modeled by allowing hydration numbers to receive positive or negative values that depend on the number of free water molecules around the center specie. 

The multielectrolyte r-eNRTL method supports solutions with a single solvent and two electrolytes with a common cation or anion and an example on how to use it is given in :ref:`_how_to_use_multielectrolytes_refined_enrtl_in_mvc`. In this method, the electrolyte is dissociated in the solvent as shown in the equation below:

.. math:: \rm AB \rightarrow \rm A^{+} + \rm B^{-} 

where :math:`AB` represents the electrolyte while :math:`A` is the cation and :math:`B` the anion.


Main Assumptions
^^^^^^^^^^^^^^^^
Some of the main assumptions in this model are given below:

1. Hydration numbers are treated as ion-specific parameters, and their values are adjusted in the refined electrolyte NRTL model. They are referred to as hydration indices and are allowed to take both negative and positive vales in order to describe the structure of water in the local neighborhood.

2. The binary interaction parameters for the electrolyte-water specific pairs are taken from the refined electrolyte NRTL model, while electrolyte-electrolyte specific interaction parameters are fitted using the multielectrolyte refined eNRTL method.

3. The activity coefficients of the hydrated solution are converted to the experimentalist's level molal activity coefficients of the unhydrated solution while keeping the total Gibbs free energy of the solution constant.

4. The model uses unsymmetrical reference state

5. The average distance of closest approach of the ions is calculated by taking a weighted average of the distance of closest approach of all the ion pairs.

6. The material state is in terms of phase-component flow, temperature, and pressure (FcpTP), which includes full information on the phase equilibria within the state variables reducing the complexity of the problem.
   

Setup for Configuration Dictionary:
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
In addition to the core configuration options that are normally included for an IDAES control volume and the eNRTL method, the r-eNRTL method includes two different hydration models and at least one of them should be selected when using the method. The two hydration models are: constant hydration and stepwise hydration. To select which hydration model to use include `constant_hydration` or `stepwise_hydration` in the components `parameter_data` under the `hydration_model` key word. For the calculation of the hydration terms when using both models, the r-eNRTL requires the hydration constant for the electrolyte and the hydration number, ionic radius, partial molar volume, and number of available sites for the ions. To know more details about how to setup the default parameters in a configuration dictionary, refer to `how_to_use_apparent_and_True_chemical_species`.

An example of how to setup a configuration dictionary for the r-eNRTL is provided below:

.. code-block::

    from pyomo.environ import Param, units as pyunits
    from idaes.core import AqueousPhase, Solvent, Apparent, Anion, Cation
    from idaes.models.properties.modular_properties.base.generic_property import StateIndex
    from idaes.models.properties.modular_properties.state_definitions import FpcTP
    from idaes.models.properties.modular_properties.pure.electrolyte import relative_permittivity_constant

    # Import refined eNRTL method
    from refined_enrtl import rENRTL

    """
    References:
    [1] Islam, R.I., et al., Molecular thermodynamics for scaling
    prediction: Case of membrane distillation, Separation and Purification
    Technology, 2021, Vol. 276.

    [2] Xi Yang, Paul I. Barton, and George M. Bollas, Refined
    electrolyte-NRTL model: Inclusion of hydration for the detailed
    description of electrolyte solutions. Part I: Single electrolytes up
    to moderate concentrations, single salts up to solubility limit.

    [3] Y. Marcus, A simple empirical model describing the thermodynamics
    of hydration of ions of widely varying charges, sizes, and shapes,
    Biophys. Chem. 51 (1994) 111–127.

    [4] Simonin JP, Bernard O., Krebs S., Kunz W., Modelling of the
    thermodynamic properties of ionic solutions using a stepwise solvation
    equilibrium model. Fluid Phase Equil., 2006,242:176-188.

    [5] Y. Marcus, Ion solvation, Wiley-Interscience, New York, 1985.

    [6] Y. Marcus, Thermodynamics of solvation of ions. Part 5.—Gibbs free energy of hydration at
    # 298.15 K, J. Chem. Soc., Faraday Trans. 87 (1991) 2995–2999. doi:10.1039/FT9918702995.

    tau, hydration numbers, and hydration constant values are obtained
    from ref[2], ionic radii is taken from ref[3] and ref[6], and partial molar volume at infinite dilution
    from ref[5], and number of sites and minimum hydration number from
    ref[4].
    """

    def dens_mol_water_expr(b, s, T):
        return 1000

    configuration = {
        "components": {
            "H2O": {
                "type": Solvent,
                "dens_mol_liq_comp": dens_mol_water_expr,
                "relative_permittivity_liq_comp": relative_permittivity_constant,
                "parameter_data": {
                    "mw": (18.01528e-3, pyunits.kg / pyunits.mol),
                    "relative_permittivity_liq_comp": relative_permittivity_constant,
                },
            },
            "NaCl": {
                "type": Apparent,
                "dissociation_species": {"Na+": 1, "Cl-": 1},
                "parameter_data":{
                    "hydration_constant": 3.596
                }
            },
            "Na+": {
                "type": Cation,
                "charge": +1,
                "parameter_data": {
                    "mw": 22.990e-3,
                    "ionic_radius": 1.02,
                    "partial_vol_mol": -6.7,
		    "hydration_number": 1.51,
		    "min_hydration_number": 0,
		    "number_sites": 4
                }
            },
            "Cl-": {
                "type": Anion,
                "charge": -1,
                "parameter_data": {
                    "mw": 35.453e-3,
                    "ionic_radius": 1.81,
                    "partial_vol_mol": 24.2,
		    "hydration_number": 0.5,
		    "min_hydration_number": 0,
		    "number_sites": 4
                }
            },
        },
        "phases": {
            "Liq": {
                "type": AqueousPhase,
                "equation_of_state": rENRTL,
            }
        },
        "base_units": {
            "time": pyunits.s,
            "length": pyunits.m,
            "mass": pyunits.kg,
            "amount": pyunits.mol,
            "temperature": pyunits.K,
        },
        "state_definition": FpcTP,
        "state_components": StateIndex.true,
        "pressure_ref": 101325,
        "temperature_ref": 298.15,
        "parameter_data": {
	    "hydration_model": "constant_hydration",
            "Liq_tau": {
                ("H2O", "Na+, Cl-"): 7.951, 
                ("Na+, Cl-", "H2O"): -3.984,
            }
        },
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
            ("mole_frac_phase_comp_apparent", ("Liq", "H2O")): 1,
        }
    }
