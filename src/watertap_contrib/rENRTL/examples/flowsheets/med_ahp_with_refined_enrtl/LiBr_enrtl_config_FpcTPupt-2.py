###############################################################################
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
###############################################################################


"""Configuration dictionary for refined eNRTL model

This is a modified version of the eNRTL property configuration file:
https://github.com/PSORLab/NAWIConcentratedElectrolytes/blob/MED_models/flowsheets/benchmark_system/3MED_AHP/enrtl_config_FpcTP.py

This configuration file can also be used to run the IDAES eNRTL method

References:
[1] X. Yang, P. I. Barton, G. M. Bollas, Refined electrolyte-NRTL model: 
Inclusion of hydration for the detailed description of electrolyte solutions. 
Part I. Single electrolytes up to moderate concentrations, single salts up to the solubility limit. Under Review (2024).

[2] Y. Marcus, A simple empirical model describing the thermodynamics
of hydration of ions of widely varying charges, sizes, and shapes,
Biophys. Chem. 51 (1994) 111–127.

[3] JP. Simonin, O. Bernard, S. Krebs, W. Kunz, Modelling of the
thermodynamic properties of ionic solutions using a stepwise solvation
equilibrium model. Fluid Phase Equil. doi:10.1016/j.fluid.2006.01.019

[4] C.-C. Chen, H.I. Britt, J.F. Boston, L.B.Evans, Local Composition Model for Excess Gibbs Energy of Electrolyte Systems. 
Part I: Single solvent, single completely dissociated electrolyte systems. AIChE Journal, 28(4), 588-596. (1982)

tau, hydration numbers, and hydration constant values are obtained
from ref[1], ionic radii and partial molar volume at infinite dilution
from ref[2], and number of sites and minimum hydration number from
ref[3].

Modified by: Adaeze Maduako and Nazia Aslam from University of Connecticut
"""
# Import Pyomo components
from pyomo.environ import Param, units as pyunits

# Import IDAES libraries
from idaes.core import AqueousPhase, Solvent, Apparent, Anion, Cation
from idaes.models.properties.modular_properties.base.generic_property import StateIndex
from idaes.models.properties.modular_properties.state_definitions import FpcTP, FTPx
from idaes.models.properties.modular_properties.pure.electrolyte import (relative_permittivity_constant,)
from idaes.models.properties.modular_properties.eos.enrtl_reference_states import (Symmetric,Unsymmetric,)
from idaes.core.util.exceptions import ConfigurationError

refined_enrtl_method = True

if refined_enrtl_method:

    # Import refined eNRTL method
    from refined_enrtl import rENRTL

    # The hydration models supported by the refined eNRTL method are:
    # constant_hydration or stepwise_hydration.
    hydration_model = "constant_hydration"

    if hydration_model == "constant_hydration":
        tau_solvent_ionpair = 8.827
        tau_ionpair_solvent = -4.525
    elif hydration_model == "stepwise_hydration":
        tau_solvent_ionpair = 7.915
        tau_ionpair_solvent = -4.109
    else:
        raise ConfigurationError(f"The given hydration model is not supported by the refined model. "
                                    "Please, try 'constant_hydration' or 'stepwise_hydration'.")
        

    print()
    print("**Using " + hydration_model + " in refined eNRTL model in the LiBr config file")
    print()


    def dens_mol_water_expr(b, s, T):
        return 1000 / 18e-3 * pyunits.mol / pyunits.m**3


    def relative_permittivity_expr(b, s, T):
        AM = 78.54003
        BM = 31989.38
        CM = 298.15

        return AM + BM * (1 / T* pyunits.K - 1 / CM)


    configuration = {
        "components": {
            "H2O": {
                "type": Solvent,
                "dens_mol_liq_comp": dens_mol_water_expr,
                "relative_permittivity_liq_comp": relative_permittivity_expr,
                "parameter_data": {
                    "mw": (18.01528e-3, pyunits.kg / pyunits.mol),
                    "relative_permittivity_liq_comp": relative_permittivity_expr,
                },
            },
            "LiBr": {
                "type": Apparent,
                "dissociation_species": {"Li+": 1, "Br-": 1},
                "parameter_data": {"hydration_constant": 28.90},
            },
            "Li+": {
                "type": Cation,
                "charge": +1,
                "parameter_data": {
                    "mw": 6.941e-3,
                    "ionic_radius": 0.69,
                    "partial_vol_mol": -6.4,
                    "hydration_number": 2.48,
                    "min_hydration_number": 0,
                    "number_sites": 4,
                },
            },
            "Br-": {
                "type": Anion,
                "charge": -1,
                "parameter_data": {
                    "mw": 79.904e-3,
                    "ionic_radius": 1.96,
                    "partial_vol_mol": 30.2,
                    "hydration_number": 0.37,
                    "min_hydration_number": 0,
                    "number_sites": 4,
                },
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
            "hydration_model": hydration_model,
            "Liq_tau": {
                ("H2O", "Li+, Br-"): tau_solvent_ionpair,
                ("Li+, Br-", "H2O"): tau_ionpair_solvent,
            },
        },
        "default_scaling_factors": {
            ("flow_mol_phase_comp", ("Liq", "Li+")): 1e1,
            ("flow_mol_phase_comp", ("Liq", "Br-")): 1e1,
            ("flow_mol_phase_comp", ("Liq", "H2O")): 1e-1,
            ("mole_frac_comp", "Li+"): 1e2,
            ("mole_frac_comp", "Br-"): 1e2,
            ("mole_frac_comp", "H2O"): 1,
            ("mole_frac_phase_comp", ("Liq", "Li+")): 1e2,
            ("mole_frac_phase_comp", ("Liq", "Br-")): 1e2,
            ("mole_frac_phase_comp", ("Liq", "H2O")): 1,
            ("flow_mol_phase_comp_apparent", ("Liq", "LiBr")): 1e1,
            ("flow_mol_phase_comp_apparent", ("Liq", "H2O")): 1e-1,
            (
                "mole_frac_phase_comp_apparent",
                ("Liq", "LiBr"),
            ): 1e3, 
            ("mole_frac_phase_comp_apparent", ("Liq", "H2O")): 1,
        },
    }

else:
    print()
    print("**Using IDAES eNRTL model in the LiBr config file")
    print()
    # Import eNRTL method
    from idaes.models.properties.modular_properties.eos.enrtl import ENRTL

    class ConstantVolMol:
        def build_parameters(b):
            b.vol_mol_pure = Param(initialize=18e-6, units=pyunits.m**3 / pyunits.mol, mutable=True)

        def return_expression(b, cobj, T):
            return cobj.vol_mol_pure


    configuration = {
        "components": {
            "H2O": {
                "type": Solvent,
                "vol_mol_liq_comp": ConstantVolMol,
                "relative_permittivity_liq_comp": relative_permittivity_constant,
                "parameter_data": {
                    "mw": (18.01528e-3, pyunits.kg / pyunits.mol),
                    "relative_permittivity_liq_comp": 78.54,
                },
            },
            "LiBr": {
                "type": Apparent,
                "dissociation_species": {"Li+": 1, "Br-": 1},
            },
            "Li+": {
                "type": Cation,
                "charge": +1,
                "parameter_data": {
                    "mw": (6.941e-3, pyunits.kg / pyunits.mol)
                }
            },
            "Br-": {
                "type": Anion,
                "charge": -1,
                "parameter_data": {
                    "mw": (79.904e-3, pyunits.kg / pyunits.mol)
                }
            },
        },
        "phases": {
            "Liq": {
                "type": AqueousPhase,
                "equation_of_state": ENRTL,
                "equation_of_state_options": {"reference_state": Symmetric},
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
            "Liq_tau": {
                ("H2O", "Li+, Br-"): 10.449, # from ref [4]
                ("Li+, Br-", "H2O"): -5.348,
            }
        },
        "default_scaling_factors": {
            ("flow_mol_phase_comp", ("Liq", "Li+")): 1e1,
            ("flow_mol_phase_comp", ("Liq", "Br-")): 1e1,
            ("flow_mol_phase_comp", ("Liq", "H2O")): 1e-1,
            ("mole_frac_comp", "Li+"): 1e2,
            ("mole_frac_comp", "Br-"): 1e2,
            ("mole_frac_comp", "H2O"): 1,
            ("mole_frac_phase_comp", ("Liq", "Li+")): 1e2,
            ("mole_frac_phase_comp", ("Liq", "Br-")): 1e2,
            ("mole_frac_phase_comp", ("Liq", "H2O")): 1,
            ("flow_mol_phase_comp_apparent", ("Liq", "LiBr")): 1e1,
            ("flow_mol_phase_comp_apparent", ("Liq", "H2O")): 1e-1,
            ("mole_frac_phase_comp_apparent", ("Liq", "LiBr")): 1e3,
            ("mole_frac_phase_comp_apparent", ("Liq", "H2O")): 1,
        },
    }
