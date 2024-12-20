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

This is a modified version of the eNRTL property configuration
dictionary for synthetic hard water in the WaterTAP full treatment
train example:
https://github.com/watertap-org/watertap/blob/main/watertap/examples/flowsheets/full_treatment_train/model_components/eNRTL/entrl_config_FpcTP.py

References:
[1] R.I. Islam, et al., Molecular thermodynamics for scaling
prediction: Case of membrane distillation, Separation and Purification
Technology, 2021, Vol. 276.

[2] X. Yang, P. I. Barton, G. M. Bollas, Refined electrolyte-NRTL model: 
Inclusion of hydration for the detailed description of electrolyte solutions. 
Part I. Single electrolytes up to moderate concentrations, single salts up to the solubility limit. Under Review (2024).

[3] Y. Marcus, A simple empirical model describing the thermodynamics
of hydration of ions of widely varying charges, sizes, and shapes,
Biophys. Chem. 51 (1994) 111–127.

[4] JP. Simonin, O. Bernard, S. Krebs, W. Kunz, Modelling of the
thermodynamic properties of ionic solutions using a stepwise solvation
equilibrium model. Fluid Phase Equil. doi:10.1016/j.fluid.2006.01.019

tau, hydration numbers, and hydration constant values are obtained
from ref[2], ionic radii and partial molar volume at infinite dilution
from ref[3], and number of sites and minimum hydration number from
ref[4].

Modified by: Nazia Aslam

"""
# Import Pyomo components
from pyomo.environ import Param, units as pyunits

# Import IDAES libraries
from idaes.core import AqueousPhase, Solvent, Apparent, Anion, Cation
from idaes.models.properties.modular_properties.base.generic_property import StateIndex
from idaes.models.properties.modular_properties.state_definitions import FpcTP, FTPx
from idaes.models.properties.modular_properties.pure.electrolyte import (
    relative_permittivity_constant,
)
from idaes.models.properties.modular_properties.eos.enrtl_reference_states import (
    Symmetric,
    Unsymmetric,
)
from idaes.core.util.exceptions import ConfigurationError

refined_enrtl_method = True

if refined_enrtl_method:
    # Import refined eNRTL method
    from watertap_contrib.rENRTL.examples.flowsheets.mvc_with_refined_enrtl.refined_enrtl import (
        rENRTL,
    )

    # The hydration models supported by the refined eNRTL method are:
    # constant_hydration or stepwise_hydration.
    hydration_model = "constant_hydration"

    if hydration_model == "constant_hydration":
        tau_solvent_ionpair = 7.951
        tau_ionpair_solvent = -3.984
    elif hydration_model == "stepwise_hydration":
        tau_solvent_ionpair = 7.486
        tau_ionpair_solvent = -3.712
    else:
        raise ConfigurationError(
            f"The given hydration model is not supported by the refined model. "
            "Please, try 'constant_hydration' or 'stepwise_hydration'."
        )

    print()
    print(
        "**Using "
        + hydration_model
        + " refined eNRTL model in the single eNRTL config file"
    )
    print()

    def dens_mol_water_expr(b, s, T):
        return 1000 / 18e-3 * pyunits.mol / pyunits.m**3

    def relative_permittivity_expr(b, s, T):
        AM = 78.54003
        BM = 31989.38
        CM = 298.15

        return AM + BM * (1 / T - 1 / CM)

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
            "NaCl": {
                "type": Apparent,
                "dissociation_species": {"Na+": 1, "Cl-": 1},
                "parameter_data": {"hydration_constant": 3.60},
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
                    "number_sites": 4,
                },
            },
            "Cl-": {
                "type": Anion,
                "charge": -1,
                "parameter_data": {
                    "mw": 35.453e-3,
                    "ionic_radius": 1.81,
                    "partial_vol_mol": 23.3,
                    "hydration_number": 0.5,
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
                ("H2O", "Na+, Cl-"): tau_solvent_ionpair,
                ("Na+, Cl-", "H2O"): tau_ionpair_solvent,
            },
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
            (
                "mole_frac_phase_comp_apparent",
                ("Liq", "NaCl"),
            ): 1e3,
            ("mole_frac_phase_comp_apparent", ("Liq", "H2O")): 1,
        },
    }

else:
    print()
    print("**Using IDAES eNRTL model in the single eNRTL config file")
    print()
    # Import eNRTL method
    from idaes.models.properties.modular_properties.eos.enrtl import ENRTL

    class ConstantVolMol:
        def build_parameters(b):
            b.vol_mol_pure = Param(
                initialize=18e-6, units=pyunits.m**3 / pyunits.mol, mutable=True
            )

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
            "NaCl": {
                "type": Apparent,
                "dissociation_species": {"Na+": 1, "Cl-": 1},
            },
            "Na+": {
                "type": Cation,
                "charge": +1,
                "parameter_data": {"mw": (22.990e-3, pyunits.kg / pyunits.mol)},
            },
            "Cl-": {
                "type": Anion,
                "charge": -1,
                "parameter_data": {"mw": (35.453e-3, pyunits.kg / pyunits.mol)},
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
            "Liq_tau": {  # Table 1 [1]
                ("H2O", "Na+, Cl-"): 8.885,  # from ref [2]
                ("Na+, Cl-", "H2O"): -4.549,
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
        },
    }
