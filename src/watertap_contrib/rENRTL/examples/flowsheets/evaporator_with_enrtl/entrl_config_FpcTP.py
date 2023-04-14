###############################################################################
# WaterTAP Copyright (c) 2021, The Regents of the University of California,
# through Lawrence Berkeley National Laboratory, Oak Ridge National
# Laboratory, National Renewable Energy Laboratory, and National Energy
# Technology Laboratory (subject to receipt of any required approvals from
# the U.S. Dept. of Energy). All rights reserved.
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

"""eNRTL property configuration dictionary

This is a modified version of the eNRTL property configuration
dictionary for synthetic hard water in the WaterTAP full treatment
train example:
https://github.com/watertap-org/watertap/blob/main/watertap/examples/flowsheets/full_treatment_train/model_components/eNRTL/entrl_config_FpcTP.py

References:

[1] Local Composition Model for Excess Gibbs Energy of Electrolyte
Systems, Pt 1.  Chen, C.-C., Britt, H.I., Boston, J.F., Evans, L.B.,
AIChE Journal, 1982, Vol. 28(4), pgs. 588-596

Modified by: Soraya Rawlings

"""

from pyomo.environ import Param, units as pyunits

from idaes.core import AqueousPhase, Solvent, Apparent, Anion, Cation
from idaes.models.properties.modular_properties.eos.enrtl import ENRTL
from idaes.models.properties.modular_properties.eos.enrtl_reference_states import (
    Symmetric,
)
from idaes.models.properties.modular_properties.base.generic_property import StateIndex
from idaes.models.properties.modular_properties.state_definitions import FpcTP
from idaes.models.properties.modular_properties.pure.electrolyte import (
    relative_permittivity_constant,
)


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
        "NaCl": {"type": Apparent, "dissociation_species": {"Na+": 1, "Cl-": 1}},
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
        "Liq_tau": {
            # From reference [1]
            ("H2O", "Na+, Cl-"): 8.885,
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
