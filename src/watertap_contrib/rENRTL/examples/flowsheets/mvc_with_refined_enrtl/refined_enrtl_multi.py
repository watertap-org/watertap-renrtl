#################################################################################
# The Institute for the Design of Advanced Energy Systems Integrated Platform
# Framework (IDAES IP) was produced under the DOE Institute for the
# Design of Advanced Energy Systems (IDAES).
#
# Copyright (c) 2018-2023 by the software owners: The Regents of the
# University of California, through Lawrence Berkeley National Laboratory,
# National Technology & Engineering Solutions of Sandia, LLC, Carnegie Mellon
# University, West Virginia University Research Corporation, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md
# for full copyright and license information.
#
# Copyright 2023-2024, National Technology & Engineering Solutions of Sandia,
# LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the
# U.S. Government retains certain rights in this software.
#
# Copyright 2023-2024, Pengfei Xu, Matthew D. Stuber, and the University
# of Connecticut.
#
# Please see the files COPYRIGHT.md and LICENSE.md for full copyright and
# license information.
#################################################################################

"""
Model for the multi-electrolyte refined Electrolyte Nonrandom Two-Liquid (r-eNRTL) activity coefficient method. 
If you need further assistance to model multi-electrolyte solutions, please contact the author 
at pengfei.xu@uconn.edu.

This method extends the single-electrolyte refined eNRTL (single r-eNRTL) approach to multi-electrolyte solutions. 
Refer to the page of the single r-eNRTL for detailed information: 
https://github.com/watertap-org/watertap-renrtl/blob/main/src/watertap_contrib/rENRTL/examples/flowsheets/mvc_with_refined_enrtl/refined_enrtl.py.

#############################################################################
References:
[1] Song, Y. and Chen, C.-C., Symmetric Electrolyte Nonrandom
Two-Liquid Activity Coefficient Model, Ind. Eng. Chem. Res., 2009,
Vol. 48, pgs. 7788–7797

[2] G. M. Bollas, C. C. Chen, and P. I. Barton, Refined
Electrolyte-NRTL Model: Activity Coefficient Expressions for
Application to Multi-Electrolyte Systems. AIChE J., 2008, 54,
1608-1624
                                                                           
[3] Xi Yang, Paul I. Barton, and George M. Bollas, Refined                 
electrolyte-NRTL model: Inclusion of hydration for the detailed            
description of electrolyte solutions. Part I: Single electrolytes up       
to moderate concentrations, single salts up to solubility limit.           

*KEY LITERATURE[3].
*This source contains the primary parameter values and equations.

[4] Y. Marcus, Ion solvation, Wiley-Interscience, New York, 1985.

[5] E. Glueckauf, Molar volumes of ions, Trans. Faraday Soc. 61 (1965).

[6] Clegg, Simon L., and Kenneth S. Pitzer. "Thermodynamics of multicomponent,
miscible, ionic solutions: generalized equations for symmetrical electrolytes."
The Journal of Physical Chemistry 96, no. 8 (1992): 3513-3520.

[7] Maribo-Mogensen, B., Kontogeorgis, G. M., & Thomsen, K. (2012). Comparison 
of the Debye–Hückel and the Mean Spherical Approximation Theories for 
Electrolyte Solutions. Industrial & engineering chemistry 
research, 51(14), 5353-5363.

[8] Braus, M. (2019). The theory of electrolytes. I. Freezing point depression 
and related phenomena (Translation).

[9] Robinson, R. A., & Stokes, R. H. (2002). Electrolyte solutions. Courier Corporation.

Note that The term "charge number" in ref [1] denotes the absolute value
of the ionic charge.

Author: Pengfei Xu (University of Connecticut), Soraya Rawlings (Sandia) ,and Wajeha
Tauqir (University of Connecticut).

Data and model contributions by Prof. George M. Bollas and his research
group at the University of Connecticut.
"""

import pyomo.environ as pyo
from pyomo.environ import (
    Expression,
    NonNegativeReals,
    exp,
    log,
    Set,
    Var,
    units as pyunits,
    value,
    Any,
)

from idaes.models.properties.modular_properties.eos.ideal import Ideal
from idaes.models.properties.modular_properties.eos.enrtl_parameters import (
    ConstantAlpha,
    ConstantTau,
)
from idaes.models.properties.modular_properties.base.utility import (
    get_method,
    get_component_object as cobj,
)
from idaes.core.util.misc import set_param_from_config
from idaes.models.properties.modular_properties.base.generic_property import StateIndex
from idaes.core.util.constants import Constants
from idaes.core.util.exceptions import BurntToast
import idaes.logger as idaeslog


# Set up logger
_log = idaeslog.getLogger(__name__)


DefaultAlphaRule = ConstantAlpha
DefaultTauRule = ConstantTau


class rENRTL(Ideal):
    # Attribute indicating support for electrolyte systems.
    electrolyte_support = True

    @staticmethod
    def build_parameters(b):
        # Build additional indexing sets for component interactions.
        pblock = b.parent_block()
        ion_pair = []
        for i in pblock.cation_set:
            for j in pblock.anion_set:
                ion_pair.append(i + ", " + j)
        b.ion_pair_set = Set(initialize=ion_pair)

        comps = pblock.solvent_set | pblock.solute_set | b.ion_pair_set

        comp_pairs = []
        comp_pairs_sym = []
        for i in comps:
            for j in comps:
                if i in pblock.solvent_set | pblock.solute_set or i != j:
                    comp_pairs.append((i, j))
                    if (j, i) not in comp_pairs_sym:
                        comp_pairs_sym.append((i, j))
        b.component_pair_set = Set(initialize=comp_pairs)
        b.component_pair_set_symmetric = Set(initialize=comp_pairs_sym)

        # Check and apply configuration for alpha rule.
        if (
            b.config.equation_of_state_options is not None
            and "alpha_rule" in b.config.equation_of_state_options
        ):
            b.config.equation_of_state_options["alpha_rule"].build_parameters(b)
        else:
            DefaultAlphaRule.build_parameters(b)

        # Check and apply configuration for tau rule.
        if (
            b.config.equation_of_state_options is not None
            and "tau_rule" in b.config.equation_of_state_options
        ):
            b.config.equation_of_state_options["tau_rule"].build_parameters(b)
        else:
            DefaultTauRule.build_parameters(b)

    @staticmethod
    def common(b, pobj):
        pname = pobj.local_name

        molecular_set = b.params.solvent_set | b.params.solute_set

        # Check options for alpha rule
        if (
            pobj.config.equation_of_state_options is not None
            and "alpha_rule" in pobj.config.equation_of_state_options
        ):
            alpha_rule = pobj.config.equation_of_state_options[
                "alpha_rule"
            ].return_expression
        else:
            alpha_rule = DefaultAlphaRule.return_expression

        # Check options for tau rule
        if (
            pobj.config.equation_of_state_options is not None
            and "tau_rule" in pobj.config.equation_of_state_options
        ):
            tau_rule = pobj.config.equation_of_state_options[
                "tau_rule"
            ].return_expression
        else:
            tau_rule = DefaultTauRule.return_expression

        # ---------------------------------------------------------------------

        # Generate a list of apparent species that have
        # dissociation species.
        b.apparent_dissociation_species_list = []
        for a in b.params.apparent_species_set:
            if "dissociation_species" in b.params.get_component(a).config:
                b.apparent_dissociation_species_list.append(a)
        b.apparent_dissociation_species_set = pyo.Set(
            initialize=b.apparent_dissociation_species_list,
            doc="Set of apparent dissociated species",
        )

        # Set hydration model from configuration dictionary and make
        # sure that both ions have all the parameters needed for each
        # hydration model.
        for app in b.apparent_dissociation_species_set:
            if "parameter_data" not in b.params.config.components[app]:
                raise BurntToast(
                    "Missing 'parameter_data' in configuration dictionary for {}. Please, make sure you have all the needed parameters in the electrolyte.".format(
                        app
                    )
                )
            if (
                "hydration_constant"
                not in b.params.config.components[app]["parameter_data"]
            ):
                raise BurntToast(
                    "Missing parameter '{}' in configuration dictionary for {}. Please, include it to be able to use the refined eNRTL method.".format(
                        "hydration_constant", app
                    )
                )
        # Essential parameters for constant hydration model
        params_for_constant_hydration = [
            "hydration_number",
            "ionic_radius",
            "partial_vol_mol",
        ]
        if b.params.config.parameter_data["hydration_model"] == "constant_hydration":
            b.constant_hydration = True
            for ion in b.params.ion_set:
                for k in params_for_constant_hydration:
                    if k not in b.params.config.components[ion]["parameter_data"]:
                        raise BurntToast(
                            "Missing parameter '{}' in {} for 'constant_hydration' model. Please, make sure to include it in the configuration dictionary to use this hydration model.".format(
                                k, ion
                            )
                        )
        else:
            raise BurntToast(
                "'{}' is not a hydration model included in the multi-electrolyte refined eNRTL, but try again using 'constant_hydration'".format(
                    b.params.config.parameter_data["hydration_model"]
                )
            )

        # Declare electrolyte and ion parameters from 'parameter_data' in the configuration
        # dictionary as Pyomo variables 'Var' with
        # fixed values and default units from 'units_dict'
        # below. A default set of units is provided, followed by
        # an assertion to ensure the parameters given in the
        # configuration dictionary match those in
        # the default 'units_dict'. Note: If the units are specified in
        # the config dict, they should be provided as the second element
        # in a tuple (value_of_parameter, units).
        units_dict = {
            "beta": pyunits.dimensionless,
            "mw": pyunits.kg / pyunits.mol,
            "hydration_number": pyunits.dimensionless,
            "ionic_radius": pyunits.angstrom,
            "partial_vol_mol": pyunits.cm**3 / pyunits.mol,
            "min_hydration_number": pyunits.dimensionless,
            "number_sites": pyunits.dimensionless,
            "hydration_constant": pyunits.dimensionless,
        }

        for ion in b.params.ion_set:
            for i in b.params.config.components[ion]["parameter_data"].keys():
                assert i in (
                    units_dict.keys()
                ), f"Given parameter {i} in {ion} is not a parameter needed in this model. Please, check the notation in the configuration dictionary to match one of the following: {units_dict.keys()}."

        for ion in b.params.ion_set:
            for i in b.params.config.components[ion]["parameter_data"].keys():
                if i == "mw":
                    pass
                else:
                    if not hasattr(b, i):
                        b.add_component(
                            i,
                            pyo.Var(
                                b.params.ion_set,
                                units=units_dict[i],
                                doc=f"{i} parameter [{units_dict[i]}]",
                            ),
                        )

            for i in b.params.config.components[ion]["parameter_data"].keys():
                if i == "mw":
                    pass
                else:
                    pdata = b.params.config.components[ion]["parameter_data"][i]
                    if isinstance(pdata, tuple):
                        assert (
                            units_dict[i] == pdata[1]
                        ), f"Check the units for '{i}'. The default units for this parameter are in {units_dict[i]}."
                        getattr(b, i)[ion].fix(pdata[0] * pdata[1])
                    else:
                        getattr(b, i)[ion].fix(pdata * units_dict[i])

        # Add parameters for apparent species with dissociation
        # species as Pyomo variables 'Var' with fixed values and
        # default units. For now, it only includes the hydration
        # constant for each electrolyte.

        for ap in b.apparent_dissociation_species_set:
            for i in b.params.config.components[ap]["parameter_data"].keys():
                if i == "hydration_constant":
                    name_h = i
        b.add_component(
            name_h,
            pyo.Var(
                b.apparent_dissociation_species_set,
                units=units_dict[name_h],
                doc=f"{name_h} parameter [{units_dict[name_h]}]",
            ),
        )

        for ap in b.apparent_dissociation_species_set:
            for i in b.params.config.components[ap]["parameter_data"].keys():
                bdata = b.params.config.components[ap]["parameter_data"][i]
                if isinstance(bdata, tuple):
                    getattr(b, i)[ap].fix(bdata[0] * bdata[1])
                else:
                    getattr(b, i)[ap].fix(bdata * units_dict[i])

        # Declare a dictionary for stoichiometric coefficient using data
        # from configuration dictionary.

        b.stoichiometric_coeff = {}
        for ap in b.apparent_dissociation_species_set:
            for i in b.params.config.components[ap]["dissociation_species"]:
                b.stoichiometric_coeff[i, ap] = (
                    b.params.config.components[ap]["dissociation_species"].get(i, [])
                    * pyunits.dimensionless
                )

        # Add the beta constant, representing the radius of
        # electrostricted water in the hydration shell of ions,
        # which is specific to each electrolyte type.
        # Beta is determined by the charge of ion pairs (e.g., 1-1 for NaCl, 1-2 for Na2SO4).
        # Beta values are estimated following Xi Yang's method ref [3] (page 35, values multiplied by 5.187529);
        # original data used for parameter estimation are in ref [9].
        b.add_component(
            "beta",
            pyo.Var(
                b.apparent_dissociation_species_set,
                units=units_dict["beta"],
                doc="{} parameter [{}]".format("beta", units_dict["beta"]),
            ),
        )

        c_dict = {}
        a_dict = {}
        for ap in b.apparent_dissociation_species_set:
            for i in b.params.config.components[ap]["dissociation_species"]:
                if i in b.params.cation_set:
                    c_dict[ap] = i
                elif i in b.params.anion_set:
                    a_dict[ap] = i

        for ap in b.apparent_dissociation_species_set:
            if (abs(cobj(b, c_dict[ap]).config.charge) == 1) and (
                abs(cobj(b, a_dict[ap]).config.charge) == 1
            ):
                b.beta[ap].fix(0.9695492)
            elif (abs(cobj(b, c_dict[ap]).config.charge) == 2) and (
                abs(cobj(b, a_dict[ap]).config.charge) == 1
            ):
                b.beta[ap].fix(0.9192301707)
            elif (abs(cobj(b, c_dict[ap]).config.charge) == 1) and (
                abs(cobj(b, a_dict[ap]).config.charge) == 2
            ):
                b.beta[ap].fix(0.8144420812)
            elif (abs(cobj(b, c_dict[ap]).config.charge) == 2) and (
                abs(cobj(b, a_dict[ap]).config.charge) == 2
            ):
                b.beta[ap].fix(0.1245007)
            elif (abs(cobj(b, c_dict[ap]).config.charge) == 3) and (
                abs(cobj(b, a_dict[ap]).config.charge) == 1
            ):
                b.beta[ap].fix(0.7392229)
            else:
                raise BurntToast(
                    f"'beta' constant not known for system with cation with charge +{cobj(b, c_dict[ap]).config.charge} and anion with charge {cobj(b, a_dict[ap]).config.charge}. Please contact the development team if you are interested in solving a case not supported by this method.".format(
                        app
                    )
                )

        # Convert molar density to mass units (kg/m³) as a Pyomo
        # Expression. This density is used to calculate
        # vol_mol_solvent (Vt) and vol_mol_solution (Vi).
        def rule_dens_mass(b):
            if len(b.params.solvent_set) == 1:
                s = b.params.solvent_set.first()
                return (
                    get_method(b, "dens_mol_liq_comp", s)(b, cobj(b, s), b.temperature)
                    * b.params.get_component(s).mw
                )

        b.add_component(
            pname + "_dens_mass",
            pyo.Expression(
                rule=rule_dens_mass, doc="Mass density of solvent (water) in kg/m3"
            ),
        )

        # ---------------------------------------------------------------------

        # Add total hydration term as a variable so it can be
        # calculated later
        if b.constant_hydration:
            b.add_component(
                pname + "_total_hydration",
                pyo.Var(
                    bounds=(-1e3, 1e3),
                    initialize=0.1,
                    units=pyunits.mol / pyunits.s,
                    doc="Total hydration number [dimensionless]",
                ),
            )

        def rule_n(b, j):
            total_hydration = getattr(b, pname + "_total_hydration")

            if len(b.params.solvent_set) == 1:
                s = b.params.solvent_set.first()
                if (pname, j) not in b.params.true_phase_component_set:
                    return Expression.Skip
                elif j in b.params.cation_set or j in b.params.anion_set:
                    return b.flow_mol_phase_comp_true[pname, j]
                elif j in b.params.solvent_set:
                    return b.flow_mol_phase_comp_true[pname, j] - total_hydration

        b.add_component(
            pname + "_n",
            pyo.Expression(
                b.params.true_species_set,
                rule=rule_n,
                doc="Moles of dissociated electrolytes",
            ),
        )

        # Calculate total hydration value
        if b.constant_hydration:

            def rule_constant_total_hydration(b):
                n = getattr(b, pname + "_n")
                total_hydration = getattr(b, pname + "_total_hydration")

                return total_hydration == (
                    sum(b.hydration_number[i] * n[i] for i in b.params.ion_set)
                )

            b.add_component(
                pname + "_constant_total_hydration_eq",
                pyo.Constraint(rule=rule_constant_total_hydration),
            )

        # Effective mol fraction X
        def rule_X(b, j):
            n = getattr(b, pname + "_n")
            if j in b.params.ion_set:
                z = abs(cobj(b, j).config.charge)
            else:
                z = 1
            return z * n[j] / (sum(n[i] for i in b.params.true_species_set))

        b.add_component(
            pname + "_X",
            pyo.Expression(
                b.params.true_species_set,
                rule=rule_X,
                doc="Charge x mole fraction term",
            ),
        )

        def rule_Y(b, j):
            if cobj(b, j).config.charge < 0:
                # Anion
                dom = b.params.anion_set
            else:
                dom = b.params.cation_set

            X = getattr(b, pname + "_X")
            return X[j] / sum(X[i] for i in dom)  # Eqns 36 and 37 in ref [1]
            # Y is a charge ratio, and thus independent of x for symmetric state

        b.add_component(
            pname + "_Y",
            pyo.Expression(b.params.ion_set, rule=rule_Y, doc="Charge composition"),
        )

        # ---------------------------------------------------------------------
        # Long-range terms
        # Eqn 2 in ref [5]
        def rule_Vo(b, i):
            b.ionic_radius_m = pyo.units.convert(
                b.ionic_radius[i], to_units=pyo.units.m
            )
            # Empirical radius
            b.emp_a_radius = pyo.units.convert(
                0.55 * pyunits.angstrom, to_units=pyo.units.m
            )

            return (
                (4 / 3)
                * Constants.pi
                * Constants.avogadro_number
                * (b.ionic_radius_m + b.emp_a_radius) ** 3
            )

        b.add_component(
            pname + "_Vo",
            pyo.Expression(
                b.params.ion_set,
                rule=rule_Vo,
                doc="Intrinsic molar volume of ions in aqueous solution [m3/mol]",
            ),
        )

        def rule_Vq(b, i):
            return pyo.units.convert(
                b.partial_vol_mol[i], to_units=pyunits.m**3 / pyunits.mol
            )

        b.add_component(
            pname + "_Vq",
            pyo.Expression(
                b.params.ion_set,
                rule=rule_Vq,
                doc="Partial molar volume of ions at infinite dilution  [m3/mol]",
            ),
        )

        def rule_Xpsum(b):
            return sum(b.flow_mol_phase_comp_true[pname, e] for e in b.params.ion_set)

        b.add_component(
            pname + "_Xpsum",
            pyo.Expression(
                rule=rule_Xpsum,
                doc="Summation of mole fraction at unhydrated level of ions [dimensionless]",
            ),
        )

        def rule_Xp(b, e):
            Xpsum = getattr(b, pname + "_Xpsum")

            if (pname, e) not in b.params.true_phase_component_set:
                return Expression.Skip
            elif e in b.params.cation_set or e in b.params.anion_set:
                if len(b.params.solvent_set) == 1:
                    s = b.params.solvent_set.first()

                    return b.flow_mol_phase_comp_true[pname, e] / (
                        b.flow_mol_phase_comp_true[pname, s] + Xpsum
                    )
            elif e in b.params.solvent_set:
                return b.flow_mol_phase_comp_true[pname, e] / (
                    b.flow_mol_phase_comp_true[pname, e] + Xpsum
                )

        b.add_component(
            pname + "_Xp",
            pyo.Expression(
                b.params.true_species_set,
                rule=rule_Xp,
                doc="Mole fraction at unhydrated level [dimensionless]",
            ),
        )

        # Eqn 1 & 5 in ref [7]. "rule_vol_mol_solvent" is used to calculate the total volume of solution.
        def rule_vol_mol_solvent(b):
            n = getattr(b, pname + "_n")
            Vo = getattr(b, pname + "_Vo")
            Vq = getattr(b, pname + "_Vq")
            Xp = getattr(b, pname + "_Xp")
            dens_mass = getattr(b, pname + "_dens_mass")  # for first solvent

            if len(b.params.solvent_set) == 1:
                s = b.params.solvent_set.first()
                term0 = (
                    b.flow_mol_phase_comp_true[pname, s]
                    * b.params.get_component(s).mw
                    / dens_mass
                )
                b.sumxc = sum(Xp[c] for c in b.params.cation_set)
                b.sumxa = sum(Xp[a] for a in b.params.anion_set)
                return (
                    term0
                    + sum(
                        n[e] *
                        # The term below is Eqn 5 in ref [7]
                        (Vq[e] + (Vo[e] - Vq[e]) * (b.sumxc + b.sumxa))
                        for e in b.params.cation_set
                    )
                    + sum(
                        n[e] *
                        # The term below is Eqn 5 in ref [7]
                        (Vq[e] + (Vo[e] - Vq[e]) * (b.sumxc + b.sumxa))
                        for e in b.params.anion_set
                    )
                )

        b.add_component(
            pname + "_vol_mol_solvent",
            pyo.Expression(
                rule=rule_vol_mol_solvent, doc="Mean molar volume of solvent [m3]"
            ),
        )

        # Functions to calculate partial molar volumes
        # Partial molar volume of solvent/cation/anion (m3/mol) derived from Eqn 10 & 11 in ref [3]
        def rule_vol_mol_solution(b, j):
            """This function calculates the partial molar volumes for ions and
            solvent needed in the refined eNRTL model

            """
            Vo = getattr(b, pname + "_Vo")
            Vq = getattr(b, pname + "_Vq")
            Xp = getattr(b, pname + "_Xp")
            dens_mass = getattr(b, pname + "_dens_mass")  # for first solvent

            if j in b.params.ion_set:
                return (
                    Vq[j]
                    + (Vo[j] - Vq[j]) * sum(Xp[i] for i in b.params.ion_set)
                    + sum(
                        Xp[j]
                        * (Vo[j] - Vq[j])
                        * (1 - sum(Xp[i] for i in b.params.ion_set))
                        for j in b.params.ion_set
                    )
                )
            else:
                term0 = b.params.get_component(j).mw / dens_mass
                term1 = sum(
                    Xp[c]
                    * (Vo[c] - Vq[c])
                    * (
                        sum(Xp[c] for c in b.params.cation_set)
                        + sum(Xp[a] for a in b.params.anion_set)
                    )
                    for c in b.params.cation_set
                )
                term2 = sum(
                    Xp[a]
                    * (Vo[a] - Vq[a])
                    * (
                        sum(Xp[c] for c in b.params.cation_set)
                        + sum(Xp[i] for i in b.params.anion_set)
                    )
                    for a in b.params.anion_set
                )
                return term0 - term1 - term2

        b.add_component(
            pname + "_vol_mol_solution",
            pyo.Expression(
                b.params.true_species_set,
                rule=rule_vol_mol_solution,
                doc="Partial molar volume of solvent [m3/mol]",
            ),
        )

        # Ionic strength.
        # Function to calculate ionic strength in mole fraction scale (m3/mol)
        # Eqn 39 in ref [6]
        def rule_I(b):
            v = getattr(b, pname + "_vol_mol_solvent")  # Vt
            n = getattr(b, pname + "_n")

            return (
                # term1
                (1 / v)
                * 1
                / sum(
                    n[i] * abs(b.params.get_component(i).config.charge)
                    for i in b.params.ion_set
                )
                # term2
                * sum(
                    sum(
                        n[c]
                        * abs(b.params.get_component(c).config.charge)
                        * n[a]
                        * abs(b.params.get_component(a).config.charge)
                        * (
                            abs(b.params.get_component(c).config.charge)
                            + abs(b.params.get_component(a).config.charge)
                        )
                        for c in b.params.cation_set
                    )
                    for a in b.params.anion_set
                )
            )

        b.add_component(
            pname + "_ionic_strength",
            pyo.Expression(rule=rule_I, doc="Ionic strength [m3 mol]"),
        )

        # Mean relative permitivity of solvent
        def rule_eps_solvent(b):  # Eqn 78 in ref [1]
            if len(b.params.solvent_set) == 1:
                s = b.params.solvent_set.first()
                return get_method(b, "relative_permittivity_liq_comp", s)(
                    b, cobj(b, s), b.temperature
                )
            else:
                return sum(
                    b.mole_frac_phase_comp_true[pname, s]
                    * get_method(b, "relative_permittivity_liq_comp", s)(
                        b, cobj(b, s), b.temperature
                    )
                    * b.params.get_component(s).mw
                    for s in b.params.solvent_set
                ) / sum(
                    b.mole_frac_phase_comp_true[pname, s] * b.params.get_component(s).mw
                    for s in b.params.solvent_set
                )

        b.add_component(
            pname + "_relative_permittivity_solvent",
            pyo.Expression(
                rule=rule_eps_solvent,
                doc="Mean relative permittivity of solvent [dimensionless]",
            ),
        )

        b.distance_species = pyo.Param(
            initialize=1.9277,
            mutable=True,
            units=pyunits.angstrom,
            doc="Distance between a solute and solvent",
        )

        # Distance of Closest Approach (m)
        # Eqn 12 in ref [3]
        def rule_ar(b, j):
            return pyo.units.convert(
                sum(
                    (
                        (
                            max(
                                0,
                                sum(
                                    value(b.hydration_number[i])
                                    for i in b.params.ion_set
                                    if i
                                    in b.params.config.components[j][
                                        "dissociation_species"
                                    ]
                                )
                                / 2,
                            )
                            * (b.beta[j] * b.distance_species) ** 3
                            + b.ionic_radius[i] ** 3
                        )
                        ** (1 / 3)
                    )
                    for i in b.params.ion_set
                    if i in b.params.config.components[j]["dissociation_species"]
                ),
                to_units=pyunits.m,
            )

        b.add_component(
            pname + "_ar",
            pyo.Expression(
                b.apparent_dissociation_species_set,
                rule=rule_ar,
                doc="Distance of closest approach [m]",
            ),
        )

        def rule_ar_avg(b):
            ar = getattr(b, pname + "_ar")
            n = getattr(b, pname + "_n")
            denominator = sum(
                sum(n[a] * n[c] for a in b.params.anion_set)
                for c in b.params.cation_set
            )

            numerator = sum(
                sum(
                    sum(
                        n[a] * n[c] * ar[j]
                        for c in b.params.cation_set
                        if c in b.params.config.components[j]["dissociation_species"]
                    )
                    for a in b.params.anion_set
                    if a in b.params.config.components[j]["dissociation_species"]
                )
                for j in b.apparent_dissociation_species_set
            )

            return numerator / denominator

        b.add_component(
            pname + "_ar_avg",
            pyo.Expression(
                rule=rule_ar_avg,
                doc="Average value of distances of closest approach [m]",
            ),
        )

        # Functions to calculate parameters for long-range equations
        # b term
        # kappa is from first line of Eqn 2 in ref [3]
        # 'get_b' formula: b = kappa*a_i /I. The I represents the ionic strength.
        def rule_b_term(b):
            eps = getattr(b, pname + "_relative_permittivity_solvent")  # EM
            eps0 = Constants.vacuum_electric_permittivity  # E0

            return (
                2
                * Constants.faraday_constant**2
                / (eps0 * eps * Constants.gas_constant * b.temperature)
            ) ** 0.5

        b.b_term = pyo.Expression(rule=rule_b_term)

        # First line of Eqn 2 in ref [3]
        def rule_kappa(b):
            Ix = getattr(b, pname + "_ionic_strength")
            eps = getattr(b, pname + "_relative_permittivity_solvent")
            eps0 = Constants.vacuum_electric_permittivity
            aravg = getattr(b, pname + "_ar_avg")
            return (
                (
                    2
                    * (Constants.faraday_constant**2)
                    * Ix
                    / (eps0 * eps * Constants.gas_constant * b.temperature)
                )
                ** 0.5
            ) / 1e5

        b.kappa = pyo.Expression(rule=rule_kappa)

        # Eqn 33 in ref [8]
        def rule_sigma(b):
            aravg = getattr(b, pname + "_ar_avg")
            return (
                # term 1
                3
                / (b.kappa * 1e5 * aravg) ** 3
                *
                # term 2
                (
                    -2 * log(1 + b.kappa * 1e5 * aravg)
                    + (1 + b.kappa * 1e5 * aravg)
                    - 1 / (1 + b.kappa * 1e5 * aravg)
                )
            )

        b.sigma = pyo.Expression(rule=rule_sigma)

        # Eqn 27 in ref [8]
        def rule_tau2(b):
            aravg = getattr(b, pname + "_ar_avg")

            return (
                # term 1
                (3 / (b.kappa * 1e5 * aravg) ** 3)
                * (
                    # term 2
                    (log(1 + b.kappa * 1e5 * aravg))
                    -
                    # term 3
                    (b.kappa * 1e5 * aravg)
                    +
                    # term 4
                    (1 / 2 * (b.kappa * 1e5 * aravg) ** 2)
                )
            )

        b.add_component(
            pname + "_tau2",
            pyo.Expression(rule=rule_tau2, doc="Newly calculated tau in PDH"),
        )

        # Eqn 1 in ref [3] The denominator of the term before the sum term, multiplied by 3
        def rule_A_DH(b):
            eps = getattr(b, pname + "_relative_permittivity_solvent")
            eps0 = Constants.vacuum_electric_permittivity

            return 1 / (16 * Constants.pi * Constants.avogadro_number) * b.b_term**3

        b.add_component(
            pname + "_A_DH",
            pyo.Expression(rule=rule_A_DH, doc="Debye-Huckel parameter"),
        )

        # Long-range (PDH) contribution to activity coefficient.
        # This equation excludes the Born correction.
        # This term derives from the partial differentiation of A in Eqn 1 of ref [3],
        # expressed as dA/dN + dA/dV * Vi, where Vi is the partial volume of the same species as N.

        def rule_log_gamma_pdh(b, j):
            tau2 = getattr(b, pname + "_tau2")
            A = getattr(b, pname + "_A_DH")
            Ix = getattr(b, pname + "_ionic_strength")
            v = getattr(b, pname + "_vol_mol_solution")  # Vm and Vi
            aravg = getattr(b, pname + "_ar_avg")

            if j in b.params.ion_set:
                z = abs(cobj(b, j).config.charge)
                term1 = -A * z**2 * Ix**0.5 / (1 + b.b_term * aravg * Ix**0.5)
                term2 = (
                    v[j]
                    * 2
                    * A
                    / (b.b_term * aravg) ** 3
                    * (
                        (1 + b.b_term * aravg * Ix**0.5)
                        - 1 / (1 + b.b_term * aravg * Ix**0.5)
                        - 2 * log(1 + b.b_term * aravg * Ix**0.5)
                    )
                )
                return term1 + term2

            elif j in molecular_set:
                term1 = v[j] * 2 * A / ((b.b_term * aravg) ** 3)
                term2 = (
                    (1 + (b.b_term * aravg) * Ix**0.5)
                    - 1 / (1 + (b.b_term * aravg) * Ix**0.5)
                    - 2 * log(1 + (b.b_term * aravg) * Ix**0.5)
                )
                return term1 * term2
            else:
                raise BurntToast(
                    "{} eNRTL model encountered unexpected component.".format(b.name)
                )

        b.add_component(
            pname + "_log_gamma_pdh",
            pyo.Expression(
                b.params.true_species_set,
                rule=rule_log_gamma_pdh,
                doc="Long-range contribution to activity coefficient",
            ),
        )

        # ---------------------------------------------------------------------
        # Local Contribution Terms

        # Calculate alphas for all true species pairings
        def rule_alpha_expr(b, i, j):
            Y = getattr(b, pname + "_Y")
            if (pname, i) not in b.params.true_phase_component_set or (
                pname,
                j,
            ) not in b.params.true_phase_component_set:
                return Expression.Skip
            elif (i in molecular_set) and (j in molecular_set):
                # alpha equal user provided parameters
                return alpha_rule(b, pobj, i, j, b.temperature)
            elif i in b.params.cation_set and j in molecular_set:
                # Eqn 32 in ref [1]
                return sum(
                    Y[k] * alpha_rule(b, pobj, (i + ", " + k), j, b.temperature)
                    for k in b.params.anion_set
                )
            elif j in b.params.cation_set and i in molecular_set:
                # Eqn 32 in ref [1]
                return sum(
                    Y[k] * alpha_rule(b, pobj, (j + ", " + k), i, b.temperature)
                    for k in b.params.anion_set
                )
            elif i in b.params.anion_set and j in molecular_set:
                # Eqn 33 in ref [1]
                return sum(
                    Y[k] * alpha_rule(b, pobj, (k + ", " + i), j, b.temperature)
                    for k in b.params.cation_set
                )
            elif j in b.params.anion_set and i in molecular_set:
                # Eqn 33 in ref [1]
                return sum(
                    Y[k] * alpha_rule(b, pobj, (k + ", " + j), i, b.temperature)
                    for k in b.params.cation_set
                )
            elif i in b.params.cation_set and j in b.params.anion_set:
                # Eqn 34 in ref [1]
                if len(b.params.cation_set) > 1:
                    return sum(
                        Y[k]
                        * alpha_rule(
                            b, pobj, (i + ", " + j), (k + ", " + j), b.temperature
                        )
                        for k in b.params.cation_set
                    )
                else:
                    return 0.2
            elif i in b.params.anion_set and j in b.params.cation_set:
                # Eqn 35 in ref [1]
                if len(b.params.anion_set) > 1:
                    return sum(
                        Y[k]
                        * alpha_rule(
                            b, pobj, (j + ", " + i), (j + ", " + k), b.temperature
                        )
                        for k in b.params.anion_set
                    )
                else:
                    return 0.2
            elif (i in b.params.cation_set and j in b.params.cation_set) or (
                i in b.params.anion_set and j in b.params.anion_set
            ):
                # No like-ion interactions
                return Expression.Skip
            else:
                raise BurntToast(
                    "{} eNRTL model encountered unexpected component pair {}.".format(
                        b.name, (i, j)
                    )
                )

        b.add_component(
            pname + "_alpha",
            pyo.Expression(
                b.params.true_species_set,
                b.params.true_species_set,
                rule=rule_alpha_expr,
                doc="Non-randomness parameters",
            ),
        )

        # Calculate G terms
        def rule_G_expr(b, i, j):
            Y = getattr(b, pname + "_Y")

            def _G_appr(b, pobj, i, j, T):  # Eqn 23 in ref [1]
                if i != j:
                    return exp(
                        -alpha_rule(b, pobj, i, j, T) * tau_rule(b, pobj, i, j, T)
                    )
                else:
                    return 1

            if (pname, i) not in b.params.true_phase_component_set or (
                pname,
                j,
            ) not in b.params.true_phase_component_set:
                return Expression.Skip
            elif (i in molecular_set) and (j in molecular_set):
                # G comes directly from parameters
                return _G_appr(b, pobj, i, j, b.temperature)
            elif i in b.params.cation_set and j in molecular_set:
                # Eqn 38 in ref [1]
                return sum(
                    Y[k] * _G_appr(b, pobj, (i + ", " + k), j, b.temperature)
                    for k in b.params.anion_set
                )
            elif i in molecular_set and j in b.params.cation_set:
                # Eqn 40 in ref [1]
                return sum(
                    Y[k] * _G_appr(b, pobj, i, (j + ", " + k), b.temperature)
                    for k in b.params.anion_set
                )
            elif i in b.params.anion_set and j in molecular_set:
                # Eqn 39 in ref [1]
                return sum(
                    Y[k] * _G_appr(b, pobj, (k + ", " + i), j, b.temperature)
                    for k in b.params.cation_set
                )
            elif i in molecular_set and j in b.params.anion_set:
                # Eqn 41 in ref [1]
                return sum(
                    Y[k] * _G_appr(b, pobj, i, (k + ", " + j), b.temperature)
                    for k in b.params.cation_set
                )
            elif i in b.params.cation_set and j in b.params.anion_set:
                # Eqn 42 in ref [1]
                if len(b.params.cation_set) > 1:
                    return sum(
                        Y[k]
                        * _G_appr(
                            b, pobj, (i + ", " + j), (k + ", " + j), b.temperature
                        )
                        for k in b.params.cation_set
                    )
                else:
                    # This term does not exist for single cation systems
                    # However, need a valid result to calculate tau
                    return 1
            elif i in b.params.anion_set and j in b.params.cation_set:
                # Eqn 43 in ref [1]
                if len(b.params.anion_set) > 1:
                    return sum(
                        Y[k]
                        * _G_appr(
                            b, pobj, (j + ", " + i), (j + ", " + k), b.temperature
                        )
                        for k in b.params.anion_set
                    )
                else:
                    # This term does not exist for single anion systems
                    # However, need a valid result to calculate tau
                    return 1
            elif (i in b.params.cation_set and j in b.params.cation_set) or (
                i in b.params.anion_set and j in b.params.anion_set
            ):
                # No like-ion interactions
                return Expression.Skip
            else:
                raise BurntToast(
                    "{} eNRTL model encountered unexpected component pair {}.".format(
                        b.name, (i, j)
                    )
                )

        b.add_component(
            pname + "_G",
            pyo.Expression(
                b.params.true_species_set,
                b.params.true_species_set,
                rule=rule_G_expr,
                doc="Local interaction G term",
            ),
        )

        # Calculate tau terms
        def rule_tau_expr(b, i, j):
            if (pname, i) not in b.params.true_phase_component_set or (
                pname,
                j,
            ) not in b.params.true_phase_component_set:
                return Expression.Skip
            elif (i in molecular_set) and (j in molecular_set):
                # tau equal to parameter
                return tau_rule(b, pobj, i, j, b.temperature)
            elif (i in b.params.cation_set and j in b.params.cation_set) or (
                i in b.params.anion_set and j in b.params.anion_set
            ):
                # No like-ion interactions
                return Expression.Skip
            else:
                alpha = getattr(b, pname + "_alpha")
                G = getattr(b, pname + "_G")
                # Eqn 44 in ref [1]
                return -log(G[i, j]) / alpha[i, j]

        b.add_component(
            pname + "_tau",
            pyo.Expression(
                b.params.true_species_set,
                b.params.true_species_set,
                rule=rule_tau_expr,
                doc="Binary interaction energy parameters",
            ),
        )

        # Calculate new tau and G values equivalent to four-indexed
        # parameters.
        def _calculate_tau_alpha(b):
            """This function calculates and sets tau and alpha with four indices
            as mutable parameters. Note that the ca_m terms refer
            to the parameters with four indices as cm_mm and am_mm

            """

            G = getattr(b, pname + "_G")

            def _G_appr(b, pobj, i, j, T):  # Eqn 23 in ref [1]
                if i != j:
                    return exp(
                        -alpha_rule(b, pobj, i, j, T) * tau_rule(b, pobj, i, j, T)
                    )

            b.alpha_ij_ij = pyo.Param(
                b.params.true_species_set,
                b.params.true_species_set,
                b.params.true_species_set,
                b.params.true_species_set,
                mutable=True,
                initialize=0.2,
                units=pyunits.dimensionless,
            )
            b.tau_ij_ij = pyo.Var(
                b.params.true_species_set,
                b.params.true_species_set,
                b.params.true_species_set,
                b.params.true_species_set,
                initialize=1,
                units=pyunits.dimensionless,
            )

            for c in b.params.cation_set:
                for a in b.params.anion_set:
                    for m in molecular_set:
                        b.alpha_ij_ij[c, a, m, m] = alpha_rule(
                            b, pobj, (c + ", " + a), m, b.temperature
                        )
                        b.alpha_ij_ij[a, c, m, m] = alpha_rule(
                            b, pobj, (c + ", " + a), m, b.temperature
                        )
                        b.alpha_ij_ij[m, a, c, a] = alpha_rule(
                            b, pobj, m, (c + ", " + a), b.temperature
                        )
                        b.alpha_ij_ij[m, c, a, c] = alpha_rule(
                            b, pobj, m, (c + ", " + a), b.temperature
                        )

            for c in b.params.cation_set:
                for a in b.params.anion_set:
                    for ap in b.params.anion_set:
                        if a != ap:
                            b.alpha_ij_ij[a, c, ap, c] = alpha_rule(
                                b, pobj, (c + ", " + a), (c + ", " + a), b.temperature
                            )

            for s in b.params.true_species_set:
                for m in b.params.solvent_set:
                    b.tau_ij_ij[s, m, m, m].fix(0)
                    b.tau_ij_ij[m, m, m, m].fix(0)

            for a in b.params.anion_set:
                for c in b.params.cation_set:
                    b.tau_ij_ij[a, c, a, c].fix(0)
                    b.tau_ij_ij[c, a, c, a].fix(0)
                    b.tau_ij_ij[a, a, c, a].fix(0)
                    b.tau_ij_ij[c, c, a, c].fix(0)
                    for ap in b.params.anion_set:
                        if a != ap:
                            b.tau_ij_ij[a, ap, c, ap].fix(0)
                            b.tau_ij_ij[ap, a, c, a].fix(0)

            def rule_tau_ac_apc(b, a, c, ap):
                if a != ap:
                    return b.tau_ij_ij[a, c, ap, c] == (
                        tau_rule(
                            b, pobj, (c + ", " + a), (c + ", " + ap), b.temperature
                        )
                    )
                else:
                    return pyo.Constraint.Skip

            b.add_component(
                pname + "_constraint_tau_ac_apc",
                pyo.Constraint(
                    b.params.anion_set,
                    b.params.cation_set,
                    b.params.anion_set,
                    rule=rule_tau_ac_apc,
                ),
            )

            def rule_tau_mc_ac(b, m, c, a):
                Y = getattr(b, pname + "_Y")
                return b.tau_ij_ij[m, c, a, c] == (
                    -log(
                        sum(
                            _G_appr(b, pobj, (c + ", " + ap), m, b.temperature) * Y[ap]
                            for ap in b.params.anion_set
                        )
                    )
                    / alpha_rule(b, pobj, (c + ", " + a), m, b.temperature)
                    - tau_rule(b, pobj, (c + ", " + a), m, b.temperature)
                    + tau_rule(b, pobj, m, (c + ", " + a), b.temperature)
                )

            b.add_component(
                pname + "_constraint_tau_mc_ac",
                pyo.Constraint(
                    b.params.solvent_set,
                    b.params.cation_set,
                    b.params.anion_set,
                    rule=rule_tau_mc_ac,
                ),
            )

            def rule_tau_ma_ca(b, m, a, c):
                Y = getattr(b, pname + "_Y")
                return b.tau_ij_ij[m, a, c, a] == (
                    -log(
                        sum(
                            _G_appr(b, pobj, (cp + ", " + a), m, b.temperature) * Y[cp]
                            for cp in b.params.cation_set
                        )
                    )
                    / alpha_rule(b, pobj, (c + ", " + a), m, b.temperature)
                    - tau_rule(b, pobj, (c + ", " + a), m, b.temperature)
                    + tau_rule(b, pobj, m, (c + ", " + a), b.temperature)
                )

            b.add_component(
                pname + "_constraint_tau_ma_ca",
                pyo.Constraint(
                    b.params.solvent_set,
                    b.params.anion_set,
                    b.params.cation_set,
                    rule=rule_tau_ma_ca,
                ),
            )

            return b.tau_ij_ij, b.alpha_ij_ij

        _calculate_tau_alpha(b)

        def _calculate_G(b):
            """This function calculates G with three and four indices as a
            mutable parameter. With three indices, the only one
            that is calculated is G_ca.m (G_cm.mm, G_am.mm) since
            it is needed in the refined eNRTL. Note that this G is
            not needed in the general NRTL, so this function is not
            included in the method

            """

            def _G_appr(b, pobj, i, j, T):  # Eqn 23 in ref [1]
                if i != j:
                    return exp(
                        -alpha_rule(b, pobj, i, j, T) * tau_rule(b, pobj, i, j, T)
                    )

            b.G_ij_ij = pyo.Var(
                b.params.true_species_set,
                b.params.true_species_set,
                b.params.true_species_set,
                b.params.true_species_set,
                initialize=1,
                units=pyunits.dimensionless,
            )

            def rule_G_mc_ac(b, m, c, a):
                return b.G_ij_ij[m, c, a, c] == exp(
                    -b.alpha_ij_ij[m, c, a, c] * b.tau_ij_ij[m, c, a, c]
                )

            b.add_component(
                pname + "_constraint_G_mc_ac",
                pyo.Constraint(
                    b.params.solvent_set,
                    b.params.cation_set,
                    b.params.anion_set,
                    rule=rule_G_mc_ac,
                ),
            )

            def rule_G_ma_ca(b, m, a, c):
                return b.G_ij_ij[m, a, c, a] == exp(
                    -b.alpha_ij_ij[m, a, c, a] * b.tau_ij_ij[m, a, c, a]
                )

            b.add_component(
                pname + "_constraint_G_ma_ca",
                pyo.Constraint(
                    b.params.solvent_set,
                    b.params.anion_set,
                    b.params.cation_set,
                    rule=rule_G_ma_ca,
                ),
            )

            def rule_G_ca_mm(b, c, a, m):
                return b.G_ij_ij[c, a, m, m] == _G_appr(
                    b, pobj, (c + ", " + a), m, b.temperature
                )

            b.add_component(
                pname + "_constraint_G_ca_mm",
                pyo.Constraint(
                    b.params.cation_set,
                    b.params.anion_set,
                    b.params.solvent_set,
                    rule=rule_G_ca_mm,
                ),
            )

            for c in b.params.cation_set:
                for a in b.params.anion_set:
                    b.G_ij_ij[c, a, c, a].fix(1)
                    b.G_ij_ij[a, c, a, c].fix(1)
                    b.G_ij_ij[a, a, c, a].fix(0)
                    b.G_ij_ij[c, c, a, c].fix(0)
                    b.G_ij_ij[c, c, a, a].fix(0)
                    for ap in b.params.anion_set:
                        if a != ap:
                            b.G_ij_ij[a, ap, c, ap].fix(0)
                            b.G_ij_ij[ap, a, c, a].fix(0)

            def rule_G_ac_apc(b, a, c, ap):
                if a != ap:
                    return b.G_ij_ij[a, c, ap, c] == exp(
                        -b.alpha_ij_ij[a, c, ap, c] * b.tau_ij_ij[a, c, ap, c]
                    )
                else:
                    return pyo.Constraint.Skip

            b.add_component(
                pname + "_constraint_G_ac_apc",
                pyo.Constraint(
                    b.params.anion_set,
                    b.params.cation_set,
                    b.params.anion_set,
                    rule=rule_G_ac_apc,
                ),
            )

            return b.G_ij_ij

        _calculate_G(b)

        # Local contribution to activity coefficient
        def rule_log_gamma_lc_I(b, s):
            X = getattr(b, pname + "_X")
            G = getattr(b, pname + "_G")
            tau = getattr(b, pname + "_tau")

            return log_gamma_lc(b, pname, s, X, G, tau)

        b.add_component(
            pname + "_log_gamma_lc_I",
            pyo.Expression(
                b.params.true_species_set,
                rule=rule_log_gamma_lc_I,
                doc="Local contribution at actual state",
            ),
        )

        def rule_log_gamma_inf(b, s):
            X = getattr(b, pname + "_X")
            G = getattr(b, pname + "_G")
            tau = getattr(b, pname + "_tau")

            return log_gamma_inf(b, pname, s, X, G, tau)

        b.add_component(
            pname + "_log_gamma_inf",
            pyo.Expression(
                b.params.true_species_set,
                rule=rule_log_gamma_inf,
                doc="Infinite dilution contribution",
            ),
        )

        # local or short-range interactions
        def rule_log_gamma_lc(b, s):
            log_gamma_lc_I = getattr(b, pname + "_log_gamma_lc_I")
            log_gamma_inf_dil = getattr(b, pname + "_log_gamma_inf")

            if s in molecular_set:
                return log_gamma_lc_I[s]
            else:
                # Considering the infinite dilution 'log_gamma_inf' as
                # the reference state.
                return log_gamma_lc_I[s] - log_gamma_inf_dil[s]

        b.add_component(
            pname + "_log_gamma_lc",
            pyo.Expression(
                b.params.true_species_set,
                rule=rule_log_gamma_lc,
                doc="Local contribution contribution to activity coefficient",
            ),
        )

        # Overall log gamma
        def rule_log_gamma(b, j):
            """For the refined eNRTL, log_gamma includes three types of
            contributions: short range, long range, and infinite
            dilution contributions

            """
            pdh = getattr(b, pname + "_log_gamma_pdh")
            lc = getattr(b, pname + "_log_gamma_lc")

            # NOTES: The local or short-range interactions already
            # include the infinite dilution reference state.
            return pdh[j] + lc[j]

        b.add_component(
            pname + "_log_gamma",
            pyo.Expression(
                b.params.true_species_set,
                rule=rule_log_gamma,
                doc="Log of activity coefficient",
            ),
        )

        # Activity coefficient of apparent species

        def rule_log_gamma_pm(b, j):
            cobj = b.params.get_component(j)

            if "dissociation_species" in cobj.config:
                dspec = cobj.config.dissociation_species
                term_n = 0
                term_d = 0

                for s in dspec:
                    dobj = b.params.get_component(s)
                    ln_g = getattr(b, pname + "_log_gamma")[s]
                    n = getattr(b, pname + "_n")[s]
                    term_n += n * ln_g
                    term_d += n

                return term_n / term_d

            else:
                return getattr(b, pname + "_log_gamma")[j]

        b.add_component(
            pname + "_log_gamma_appr",
            pyo.Expression(
                b.params.apparent_species_set,
                rule=rule_log_gamma_pm,
                doc="Log of mean activity coefficient",
            ),
        )

        def rule_he(b, ap):
            n = getattr(b, pname + "_n")
            he = sum(
                sum(
                    b.hydration_number[c] * n[c] + b.hydration_number[a] * n[a]
                    for c in b.params.cation_set
                    if c in b.params.config.components[ap]["dissociation_species"]
                )
                for a in b.params.anion_set
                if a in b.params.config.components[ap]["dissociation_species"]
            ) / sum(
                n[e]
                for e in b.params.ion_set
                if e in b.params.config.components[ap]["dissociation_species"]
            )
            return he

        b.add_component(
            pname + "_he",
            pyo.Expression(
                b.apparent_dissociation_species_set,
                rule=rule_he,
                doc="Mean hydration number for specific ion pairs",
            ),
        )

        def rule_mean_log_ion_pair(b, ap):
            n = getattr(b, pname + "_n")
            log_gamma = getattr(b, pname + "_log_gamma")
            mean_log_a = sum(
                sum(
                    log_gamma[c] * n[c] + log_gamma[a] * n[a]
                    for c in b.params.cation_set
                    if c in b.params.config.components[ap]["dissociation_species"]
                )
                for a in b.params.anion_set
                if a in b.params.config.components[ap]["dissociation_species"]
            ) / sum(
                n[e]
                for e in b.params.ion_set
                if e in b.params.config.components[ap]["dissociation_species"]
            )
            return mean_log_a

        b.add_component(
            pname + "_mean_log_ion_pair",
            pyo.Expression(
                b.apparent_dissociation_species_set,
                rule=rule_mean_log_ion_pair,
                doc="Mean log activity coefficient for specific ion pairs",
            ),
        )

        # Mean molal log_gamma of ions

        def rule_log_gamma_molal(b, ap):
            X = getattr(b, pname + "_X")
            lc = getattr(b, pname + "_log_gamma_lc")
            log_gamma_appr = getattr(b, pname + "_log_gamma_appr")
            log_gamma = getattr(b, pname + "_log_gamma")
            n = getattr(b, pname + "_n")
            total_hydration = getattr(b, pname + "_total_hydration")
            v = getattr(b, pname + "_vol_mol_solution")  # Vm and Vi
            aravg = getattr(b, pname + "_ar_avg")
            Ix = getattr(b, pname + "_ionic_strength")
            pdh = getattr(b, pname + "_log_gamma_pdh")
            A = getattr(b, pname + "_A_DH")
            mean_log_a = getattr(b, pname + "_mean_log_ion_pair")
            he = getattr(b, pname + "_he")
            # Eqn 2 in ref [3]
            # NOTES: Select the first solvent and apparent specie.
            if len(b.params.solvent_set) == 1:
                s = b.params.solvent_set.first()

                if b.constant_hydration:
                    return (
                        mean_log_a[ap]
                        - he[ap]
                        * log(
                            X[s]
                            * exp(
                                log_gamma[s]
                                - v[s]
                                * 2
                                * A
                                / (b.b_term * aravg) ** 3
                                * (
                                    (1 + b.b_term * aravg * Ix**0.5)
                                    - 1 / (1 + b.b_term * aravg * Ix**0.5)
                                    - 2 * log(1 + b.b_term * aravg * Ix**0.5)
                                )
                            )
                        )
                        - log(
                            (
                                b.flow_mol_phase_comp_true[pname, s]
                                + sum(n[e] for e in b.params.ion_set)
                                -
                                # total_hydration
                                sum(
                                    n[c] * b.hydration_number[c]
                                    for c in b.params.cation_set
                                )
                                - sum(
                                    n[a] * b.hydration_number[a]
                                    for a in b.params.anion_set
                                )
                            )
                            / b.flow_mol_phase_comp_true[pname, s]
                        )
                    )

        b.add_component(
            pname + "_log_gamma_molal",
            pyo.Expression(
                b.apparent_dissociation_species_set,
                rule=rule_log_gamma_molal,
                doc="Log of molal ion mean activity coefficient",
            ),
        )

    @staticmethod
    def calculate_scaling_factors(b, pobj):
        pass

    @staticmethod
    def act_phase_comp(b, p, j):
        return b.mole_frac_phase_comp[p, j] * b.act_coeff_phase_comp[p, j]

    @staticmethod
    def act_phase_comp_true(b, p, j):
        ln_gamma = getattr(b, p + "_log_gamma")
        return b.mole_frac_phase_comp_true[p, j] * exp(ln_gamma[j])

    @staticmethod
    def act_phase_comp_appr(b, p, j):
        ln_gamma = getattr(b, p + "_log_gamma_appr")
        return b.mole_frac_phase_comp_apparent[p, j] * exp(ln_gamma[j])

    @staticmethod
    def act_coeff_phase_comp(b, p, j):
        if b.params.config.state_components == StateIndex.true:
            ln_gamma = getattr(b, p + "_log_gamma")
        else:
            ln_gamma = getattr(b, p + "_log_gamma_appr")
        return exp(ln_gamma[j])

    @staticmethod
    def act_coeff_phase_comp_true(b, p, j):
        ln_gamma = getattr(b, p + "_log_gamma")
        return exp(ln_gamma[j])

    @staticmethod
    def act_coeff_phase_comp_appr(b, p, j):
        ln_gamma = getattr(b, p + "_log_gamma_appr")
        return exp(ln_gamma[j])

    @staticmethod
    def vol_mol_phase(b, p):
        # eNRTL model uses apparent species for calculating molar volume
        # TODO : Need something more rigorus to handle concentrated solutions
        v_expr = 0
        for j in b.params.apparent_species_set:
            v_comp = rENRTL.get_vol_mol_pure(b, "liq", j, b.temperature)
            v_expr += b.mole_frac_phase_comp_apparent[p, j] * v_comp

        return v_expr


def log_gamma_lc(b, pname, s, X, G, tau):
    """General function for calculating local contributions

    The same method can be used for both actual state and reference
    state by providing different X, G and tau expressions.

    """

    # indices in expressions use same names as source paper
    # mp = m' and so on

    molecular_set = b.params.solvent_set | b.params.solute_set
    aqu_species = b.params.true_species_set - b.params._non_aqueous_set

    if (pname, s) not in b.params.true_phase_component_set:
        # Non-aqueous component
        return Expression.Skip
    # Eqn 6 in ref [2]
    if s in b.params.cation_set:
        c = s

        return abs(b.params.get_component(c).config.charge) * (
            # Term 1
            sum(
                X[m]
                / sum(X[i] * G[i, m] for i in aqu_species)
                * (
                    G[c, m]
                    * (
                        tau[c, m]
                        - (
                            sum(X[i] * G[i, m] * tau[i, m] for i in aqu_species)
                            / sum(X[i] * G[i, m] for i in aqu_species)
                        )
                    )
                    + sum(
                        X[a]
                        / (
                            b.alpha_ij_ij[a, c, m, m]
                            * sum(X[cp] for cp in b.params.cation_set)
                        )
                        * (
                            (b.G_ij_ij[c, a, m, m] - G[a, m])
                            * (b.alpha_ij_ij[a, c, m, m] * tau[a, m] - 1)
                        )
                        for a in b.params.anion_set
                    )
                    - (
                        sum(X[i] * G[i, m] * tau[i, m] for i in aqu_species)
                        / sum(X[i] * G[i, m] for i in aqu_species)
                    )
                    * sum(
                        X[a]
                        / sum(X[cp] for cp in b.params.cation_set)
                        * (b.G_ij_ij[c, a, m, m] - G[a, m])
                        for a in b.params.anion_set
                    )
                )
                for m in molecular_set
            )
            +
            # Term 2
            sum(
                X[a]
                / sum(X[ap] for ap in b.params.anion_set)
                * sum(
                    X[i] * b.G_ij_ij[i, c, a, c] * b.tau_ij_ij[i, c, a, c]
                    for i in aqu_species - b.params.cation_set
                )
                / sum(
                    X[i] * b.G_ij_ij[i, c, a, c]
                    for i in aqu_species - b.params.cation_set
                )
                for a in b.params.anion_set
            )
            +
            # Term 3
            sum(
                X[a]
                * (
                    sum(
                        X[cp]
                        / sum(X[cpp] for cpp in b.params.cation_set)
                        / sum(
                            X[i] * b.G_ij_ij[i, a, cp, a]
                            for i in (aqu_species - b.params.anion_set)
                        )
                        * (
                            b.G_ij_ij[c, a, cp, a]
                            * (
                                b.tau_ij_ij[c, a, cp, a]
                                - sum(
                                    X[i]
                                    * b.G_ij_ij[i, a, cp, a]
                                    * b.tau_ij_ij[i, a, cp, a]
                                    for i in (aqu_species - b.params.anion_set)
                                )
                                / sum(
                                    X[i] * b.G_ij_ij[i, a, cp, a]
                                    for i in (aqu_species - b.params.anion_set)
                                )
                            )
                            + sum(
                                X[m]
                                / (
                                    b.alpha_ij_ij[cp, a, m, m]
                                    * sum(
                                        X[cpp] * b.G_ij_ij[cpp, a, m, m]
                                        for cpp in b.params.cation_set
                                    )
                                )
                                * (
                                    (
                                        b.G_ij_ij[m, a, cp, a]
                                        * (b.G_ij_ij[c, a, m, m] - G[a, m])
                                        * (
                                            b.alpha_ij_ij[c, a, m, m]
                                            * b.tau_ij_ij[m, a, cp, a]
                                            - 1
                                        )
                                    )
                                )
                                for m in molecular_set
                            )
                            - sum(
                                X[i] * b.G_ij_ij[i, a, cp, a] * b.tau_ij_ij[i, a, cp, a]
                                for i in (aqu_species - b.params.anion_set)
                            )
                            / sum(
                                X[i] * b.G_ij_ij[i, a, cp, a]
                                for i in (aqu_species - b.params.anion_set)
                            )
                            * sum(
                                (
                                    X[m]
                                    / sum(
                                        X[cpp] * b.G_ij_ij[cpp, a, m, m]
                                        for cpp in b.params.cation_set
                                    )
                                )
                                * b.G_ij_ij[m, a, cp, a]
                                * (b.G_ij_ij[c, a, m, m] - G[a, m])
                                for m in molecular_set
                            )
                        )
                        for cp in b.params.cation_set
                    )
                    + (
                        1
                        / sum(X[cpp] for cpp in b.params.cation_set)
                        * (
                            sum(
                                X[i] * b.G_ij_ij[i, a, c, a] * b.tau_ij_ij[i, a, c, a]
                                for i in (aqu_species - b.params.anion_set)
                            )
                            / sum(
                                X[i] * b.G_ij_ij[i, a, c, a]
                                for i in (aqu_species - b.params.anion_set)
                            )
                            - sum(
                                (X[cp] / sum(X[cpp] for cpp in b.params.cation_set))
                                * (
                                    sum(
                                        X[i]
                                        * b.G_ij_ij[i, a, cp, a]
                                        * b.tau_ij_ij[i, a, cp, a]
                                        for i in (aqu_species - b.params.anion_set)
                                    )
                                    / sum(
                                        X[i] * b.G_ij_ij[i, a, cp, a]
                                        for i in (aqu_species - b.params.anion_set)
                                    )
                                )
                                for cp in b.params.cation_set
                            )
                        )
                    )
                )
                for a in b.params.anion_set
            )
        )
    # Eqn 7 in ref [2]
    elif s in b.params.anion_set:
        a = s

        return abs(b.params.get_component(a).config.charge) * (
            # Term 1
            sum(
                X[m]
                / sum(X[i] * G[i, m] for i in aqu_species)
                * (
                    G[a, m]
                    * (
                        tau[a, m]
                        - (
                            sum(X[i] * G[i, m] * tau[i, m] for i in aqu_species)
                            / sum(X[i] * G[i, m] for i in aqu_species)
                        )
                    )
                    + sum(
                        X[c]
                        / (
                            b.alpha_ij_ij[c, a, m, m]
                            * sum(X[ap] for ap in b.params.anion_set)
                        )
                        * (
                            (b.G_ij_ij[c, a, m, m] - G[c, m])
                            * (b.alpha_ij_ij[c, a, m, m] * tau[c, m] - 1)
                        )
                        for c in b.params.cation_set
                    )
                    - (
                        sum(X[i] * G[i, m] * tau[i, m] for i in aqu_species)
                        / sum(X[i] * G[i, m] for i in aqu_species)
                    )
                    * sum(
                        X[c]
                        / sum(X[ap] for ap in b.params.anion_set)
                        * (b.G_ij_ij[c, a, m, m] - G[c, m])
                        for c in b.params.cation_set
                    )
                )
                for m in molecular_set
            )
            +
            # Term 2
            sum(
                X[c]
                / sum(X[cp] for cp in b.params.cation_set)
                * sum(
                    X[i] * b.G_ij_ij[i, a, c, a] * b.tau_ij_ij[i, a, c, a]
                    for i in aqu_species - b.params.anion_set
                )
                / sum(
                    X[i] * b.G_ij_ij[i, a, c, a]
                    for i in aqu_species - b.params.anion_set
                )
                for c in b.params.cation_set
            )
            +
            # Term 3
            sum(
                X[c]
                * (
                    sum(
                        X[ap]
                        / sum(X[app] for app in b.params.anion_set)
                        / sum(
                            X[i] * b.G_ij_ij[i, c, ap, c]
                            for i in (aqu_species - b.params.cation_set)
                        )
                        * (
                            b.G_ij_ij[a, c, ap, c]
                            * (
                                b.tau_ij_ij[a, c, ap, c]
                                - sum(
                                    X[i]
                                    * b.G_ij_ij[i, c, ap, c]
                                    * b.tau_ij_ij[i, c, ap, c]
                                    for i in (aqu_species - b.params.cation_set)
                                )
                                / sum(
                                    X[i] * b.G_ij_ij[i, c, ap, c]
                                    for i in (aqu_species - b.params.cation_set)
                                )
                            )
                            + sum(
                                X[m]
                                / (
                                    b.alpha_ij_ij[c, ap, m, m]
                                    * sum(
                                        X[app] * b.G_ij_ij[c, app, m, m]
                                        for app in b.params.anion_set
                                    )
                                )
                                * (
                                    (
                                        b.G_ij_ij[m, c, ap, c]
                                        * (b.G_ij_ij[c, a, m, m] - G[c, m])
                                        * (
                                            b.alpha_ij_ij[c, ap, m, m]
                                            * b.tau_ij_ij[m, c, ap, c]
                                            - 1
                                        )
                                    )
                                )
                                for m in molecular_set
                            )
                            - sum(
                                X[i] * b.G_ij_ij[i, c, ap, c] * b.tau_ij_ij[i, c, ap, c]
                                for i in (aqu_species - b.params.cation_set)
                            )
                            / sum(
                                X[i] * b.G_ij_ij[i, c, ap, c]
                                for i in (aqu_species - b.params.cation_set)
                            )
                            * sum(
                                (
                                    X[m]
                                    / sum(
                                        X[app] * b.G_ij_ij[c, app, m, m]
                                        for app in b.params.anion_set
                                    )
                                )
                                * b.G_ij_ij[m, c, ap, c]
                                * (b.G_ij_ij[c, a, m, m] - G[c, m])
                                for m in molecular_set
                            )
                        )
                        for ap in b.params.anion_set
                    )
                    + (
                        1
                        / sum(X[app] for app in b.params.anion_set)
                        * (
                            sum(
                                X[i] * b.G_ij_ij[i, c, a, c] * b.tau_ij_ij[i, c, a, c]
                                for i in (aqu_species - b.params.cation_set)
                            )
                            / sum(
                                X[i] * b.G_ij_ij[i, c, a, c]
                                for i in (aqu_species - b.params.cation_set)
                            )
                            - sum(
                                (X[ap] / sum(X[app] for app in b.params.anion_set))
                                * (
                                    sum(
                                        X[i]
                                        * b.G_ij_ij[i, c, ap, c]
                                        * b.tau_ij_ij[i, c, ap, c]
                                        for i in (aqu_species - b.params.cation_set)
                                    )
                                    / sum(
                                        X[i] * b.G_ij_ij[i, c, ap, c]
                                        for i in (aqu_species - b.params.cation_set)
                                    )
                                )
                                for ap in b.params.anion_set
                            )
                        )
                    )
                )
                for c in b.params.cation_set
            )
        )
    # Eqn 8 in ref [2]
    else:
        m = s

        return (
            sum(X[i] * G[i, m] * tau[i, m] for i in aqu_species)
            / sum(X[i] * G[i, m] for i in aqu_species)
            + sum(
                (X[mp] * G[m, mp] / sum(X[i] * G[i, mp] for i in aqu_species))
                * (
                    tau[m, mp]
                    - sum(X[i] * G[i, mp] * tau[i, mp] for i in aqu_species)
                    / sum(X[i] * G[i, mp] for i in aqu_species)
                )
                for mp in molecular_set
            )
            + sum(
                sum(
                    X[a]
                    / sum(X[ap] for ap in b.params.anion_set)
                    * X[c]
                    * b.G_ij_ij[m, c, a, c]
                    / sum(X[i] * b.G_ij_ij[i, c, a, c] for i in aqu_species)
                    * (
                        b.tau_ij_ij[m, c, a, c]
                        - sum(
                            X[i] * b.G_ij_ij[i, c, a, c] * b.tau_ij_ij[i, c, a, c]
                            for i in aqu_species
                        )
                        / sum(X[i] * b.G_ij_ij[i, c, a, c] for i in aqu_species)
                    )
                    for a in b.params.anion_set
                )
                for c in b.params.cation_set
            )
            + sum(
                sum(
                    X[c]
                    / sum(X[cp] for cp in b.params.cation_set)
                    * X[a]
                    * b.G_ij_ij[m, a, c, a]
                    / sum(X[i] * b.G_ij_ij[i, a, c, a] for i in aqu_species)
                    * (
                        b.tau_ij_ij[m, a, c, a]
                        - sum(
                            X[i] * b.G_ij_ij[i, a, c, a] * b.tau_ij_ij[i, a, c, a]
                            for i in aqu_species
                        )
                        / sum(X[i] * b.G_ij_ij[i, a, c, a] for i in aqu_species)
                    )
                    for c in b.params.cation_set
                )
                for a in b.params.anion_set
            )
        )


def log_gamma_inf(b, pname, s, X, G, tau):
    """General function for calculating infinite dilution contributions"""

    if (pname, s) not in b.params.true_phase_component_set:
        # Non-aqueous component
        return Expression.Skip

    # Select first solvent
    if len(b.params.solvent_set) == 1:
        w = b.params.solvent_set.first()

    # Eqn 9 in ref [2]
    if s in b.params.cation_set:
        c = s

        return abs(b.params.get_component(c).config.charge) * (
            sum(
                (X[a] / sum(X[ap] for ap in b.params.anion_set))
                * b.tau_ij_ij[w, c, a, c]
                for a in b.params.anion_set
            )
            + G[c, w] * tau[c, w]
            + sum(
                (X[a] / sum(X[cp] for cp in b.params.cation_set))
                * (b.G_ij_ij[c, a, w, w] - G[a, w])
                * (
                    (b.alpha_ij_ij[c, a, w, w] * tau[a, w] - 1)
                    / b.alpha_ij_ij[c, a, w, w]
                )
                for a in b.params.anion_set
            )
            - sum(
                X[a]
                * (
                    sum(
                        (X[cp] / sum(X[cpp] for cpp in b.params.cation_set))
                        / b.G_ij_ij[w, a, cp, a]
                        * (
                            (b.G_ij_ij[c, a, w, w] - G[a, w])
                            * b.G_ij_ij[w, a, cp, a]
                            * (b.alpha_ij_ij[c, a, w, w] * b.tau_ij_ij[w, a, cp, a] - 1)
                            / (
                                b.alpha_ij_ij[a, c, w, w]
                                * sum(
                                    X[cpp] * b.G_ij_ij[cpp, a, w, w]
                                    for cpp in b.params.cation_set
                                )
                            )
                            - b.tau_ij_ij[w, a, cp, a]
                            * (b.G_ij_ij[c, a, w, w] - G[a, w])
                            * b.G_ij_ij[w, a, cp, a]
                            / sum(
                                X[cpp] * b.G_ij_ij[cpp, a, w, w]
                                for cpp in b.params.cation_set
                            )
                        )
                        for cp in b.params.cation_set
                    )
                    + (1 / sum(X[cpp] for cpp in b.params.cation_set))
                    * (
                        b.tau_ij_ij[w, a, c, a]
                        - sum(
                            (X[cp] / sum(X[cpp] for cpp in b.params.cation_set))
                            * b.tau_ij_ij[w, a, cp, a]
                            for cp in b.params.cation_set
                        )
                    )
                )
                for a in b.params.anion_set
            )
        )

    # Eqn 10 in ref [2]
    elif s in b.params.anion_set:
        a = s

        return abs(b.params.get_component(a).config.charge) * (
            sum(
                X[c]
                / sum(X[cp] for cp in b.params.cation_set)
                * b.tau_ij_ij[w, a, c, a]
                for c in b.params.cation_set
            )
            + G[a, w] * tau[a, w]
            + sum(
                (X[c] / sum(X[ap] for ap in b.params.anion_set))
                * (b.G_ij_ij[c, a, w, w] - G[c, w])
                * (
                    (b.alpha_ij_ij[c, a, w, w] * tau[c, w] - 1)
                    / b.alpha_ij_ij[c, a, w, w]
                )
                for c in b.params.cation_set
            )
            + sum(
                X[c]
                * (
                    sum(
                        (X[ap] / sum(X[app] for app in b.params.anion_set))
                        / b.G_ij_ij[w, c, ap, c]
                        * (
                            (b.G_ij_ij[c, a, w, w] - G[c, w])
                            * b.G_ij_ij[w, c, ap, c]
                            * (b.alpha_ij_ij[c, a, w, w] * b.tau_ij_ij[w, c, ap, c] - 1)
                            / (
                                b.alpha_ij_ij[a, c, w, w]
                                * sum(
                                    X[app] * b.G_ij_ij[c, app, w, w]
                                    for app in b.params.anion_set
                                )
                            )
                            - b.tau_ij_ij[w, c, ap, c]
                            * (b.G_ij_ij[c, a, w, w] - G[c, w])
                            * b.G_ij_ij[w, c, ap, c]
                            / sum(
                                X[app] * b.G_ij_ij[c, app, w, w]
                                for app in b.params.anion_set
                            )
                        )
                        for ap in b.params.anion_set
                    )
                    # This sign is "-" in single electrolyte eNRTL
                    # model
                    + (1 / sum(X[app] for app in b.params.anion_set))
                    * (
                        b.tau_ij_ij[w, c, a, c]
                        - sum(
                            (X[ap] / sum(X[app] for app in b.params.anion_set))
                            * b.tau_ij_ij[w, c, ap, c]
                            for ap in b.params.anion_set
                        )
                    )
                )
                for c in b.params.cation_set
            )
        )
    # This term is just 0 when water is the only solvent.
    else:
        m = s

        return tau[m, m] + G[m, m] * tau[m, m]
