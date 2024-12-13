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
# Copyright 2023, Pengfei Xu and Matthew D. Stuber and the University
# of Connecticut.
#
# Please see the files COPYRIGHT.md and LICENSE.md for full copyright and license
# information, respectively. These files are also available online at the URL
# "https://github.com/watertap-org/watertap/"
#
###############################################################################


"""Configuration dictionary for multielectrolytes refined eNRTL model

This is a modified version of the single electrolyte configuration file:
https://github.com/watertap-org/watertap/blob/main/watertap/examples/flowsheets/full_treatment_train/model_components/eNRTL/entrl_config_FpcTP.py

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

[4] Y. Marcus, Ion solvation, Wiley-Interscience, New York, NY: Wiley-Interscience, 1985. ISBN 9780471907565, 0471907561. Table 5.8.

[5] Y. Marcus, Thermodynamics of solvation of ions. Part 5.—Gibbs free energy of hydration at
298.15 K, J. Chem. Soc., Faraday Trans. 87 (1991) 2995–2999. doi:10.1039/FT9918702995.

tau, hydration numbers, and hydration constant values are obtained from ref[1], 
ionic radii is taken from ref[2] and ref[5], partial molar volume at infinite dilution from ref[4], 
and number of sites and minimum hydration number from ref[3].

Modified by: Nazia Aslam from the University of Connecticut

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
from idaes.core.util.exceptions import ConfigurationError

# Import multielectrolytes refined eNRTL method
from refined_enrtl_multi import rENRTL

print()
print("**Using constant hydration refined eNRTL model in the multi config file")
print()

# The hydration models supported by the multielectrolytes refined eNRTL method are:
# constant_hydration or stepwise_hydration.
hydration_model = "constant_hydration"

if hydration_model == "constant_hydration":
    tau_solvent_ionpair1 = 7.951
    tau_ionpair_solvent1 = -3.984
    tau_solvent_ionpair2 = 7.578
    tau_ionpair_solvent2 = -3.532
    tau_ionpair1_ionpair2 = 0
    tau_ionpair2_ionpair1 = 0

else:
    raise ConfigurationError(
        f"The given hydration model is not supported by the refined model. "
        "Please, try 'constant_hydration'.")
    


def dens_mol_water_expr(b, s, T):
    return 1000 / 18e-3 * pyunits.mol / pyunits.m**3


def relative_permittivity_expr(b, s, T):
    AM = 78.54003
    BM = 31989.38
    CM = 298.15

    return AM + BM * (1 * pyunits.K / T - 1 / CM)


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
        "Na2SO4": {
            "type": Apparent,
            "dissociation_species": {"Na+": 2, "SO4_2-": 1},
            "parameter_data": {"hydration_constant":1.022},
        },
        "Na+": {
            "type": Cation,
            "charge": +1,
            "parameter_data": {
                "mw": 22.990e-3,
                "ionic_radius": 1.02,
                "partial_vol_mol": -7.6,
                "hydration_number": 1.51,
                "min_hydration_number": 0,
                "number_sites": 4,
            },
        },
        "SO4_2-": {
            "type": Anion,
            "charge": -2,
            "parameter_data": {
                "mw": 96.064e-3 ,
                "ionic_radius": 2.40 ,
                "partial_vol_mol": 26.8 ,
                "hydration_number": -0.31 ,
                 "min_hydration_number": 0,
                "number_sites": 8,
            },
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
            ("H2O", "Na+, Cl-"): tau_solvent_ionpair1,
            ("Na+, Cl-", "H2O"): tau_ionpair_solvent1,
            ("H2O", "Na+, SO4_2-"): tau_solvent_ionpair2,
            ("Na+, SO4_2-", "H2O"): tau_ionpair_solvent2,
            ("Na+, Cl-", "Na+, SO4_2-"): tau_ionpair1_ionpair2,
            ("Na+, SO4_2-", "Na+, Cl-"): tau_ionpair2_ionpair1,
        },
    },
    "default_scaling_factors": {
        ("flow_mol_phase_comp", ("Liq", "Na+")): 1e1,
        ("flow_mol_phase_comp", ("Liq", "Cl-")): 1e1,
        ("flow_mol_phase_comp", ("Liq", "SO4_2-")): 1e1,
        ("flow_mol_phase_comp", ("Liq", "H2O")): 1e-1,
        ("mole_frac_comp", "Na+"): 1e2,
        ("mole_frac_comp", "Cl-"): 1e2,
        ("mole_frac_comp", "SO4_2-"): 1e2,
        ("mole_frac_comp", "H2O"): 1,
        ("mole_frac_phase_comp", ("Liq", "Na+")): 1e2,
        ("mole_frac_phase_comp", ("Liq", "Cl-")): 1e2,
        ("mole_frac_phase_comp", ("Liq", "SO4_2-")): 1e2,
        ("mole_frac_phase_comp", ("Liq", "H2O")): 1,
        ("flow_mol_phase_comp_apparent", ("Liq", "NaCl")): 1e1,
        ("flow_mol_phase_comp_apparent", ("Liq", "Na2SO4")): 1e1,
        ("flow_mol_phase_comp_apparent", ("Liq", "H2O")): 1e-1,
        (
            "mole_frac_phase_comp_apparent",
            ("Liq", "NaCl"),
        ): 1e3,  
        (
            "mole_frac_phase_comp_apparent",
            ("Liq", "Na2SO4"),
        ): 1e3, 
        ("mole_frac_phase_comp_apparent", ("Liq", "H2O")): 1,
    },
}
