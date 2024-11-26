#################################################################################
# The Institute for the Design of Advanced Energy Systems Integrated Platform
# Framework (IDAES IP) was produced under the DOE Institute for the
# Design of Advanced Energy Systems (IDAES).
#
# Copyright (c) 2018-2023 by the software owners: The Regents of the
# University of California, through Lawrence Berkeley National Laboratory,
# National Technology & Engineering Solutions of Sandia, LLC, Carnegie Mellon
# University, West Virginia University Research Corporation, et al.
# All rights reserved.  Please see the files COPYRIGHT.md and LICENSE.md
# for full copyright and license information.
#
# Copyright 2023-2024, National Technology & Engineering Solutions of Sandia,
# LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the
# U.S. Government retains certain rights in this software
#
# Copyright 2023-2024, Pengfei Xu and Matthew D. Stuber and the University
# of Connecticut.
#
# Please see the files COPYRIGHT.md and LICENSE.md for full copyright and
# license information.
#################################################################################

"""Model for refined ENRTL activity coefficient method using an
unsymmetrical reference state. This model is only applicable to
liquid/electrolyte phases with a single solvent and single
electrolyte.

This method is a modified version of the IDAES ENRTL activity
coefficient method, authored by Andrew Lee in collaboration with
C.-C. Chen and can be found in the link below:
https://github.com/IDAES/idaes-pse/blob/main/idaes/models/properties/modular_properties/eos/enrtl.py

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

[4] Y. Marcus, Ion solvation, Wiley-Interscience, New York, 1985.

[5] X. Yang, P. I. Barton, G. M. Bollas, The significance of
frameworks in electrolyte thermodynamic model development. Fluid Phase
Equilib., 2019

[6] E. Glueckauf, Molar volumes of ions, Trans. Faraday Soc. 61 (1965).

[7] Chen, Chau‐Chyun, Herbert I. Britt, J. F. Boston, and L. B. Evans. 
"Local composition model for excess Gibbs energy of electrolyte systems. 
Part I: Single solvent, single completely dissociated electrolyte systems."
AIChE Journal 28, no. 4 (1982): 588-596.


Note that "charge number" in the paper [1] refers to the absolute value
of the ionic charge.

Author: Soraya Rawlings in collaboration with Pengfei Xu, Wajeha
Tauqir, and Xi Yang from University of Connecticut

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
    # Add attribute indicating support for electrolyte systems
    electrolyte_support = True

    @staticmethod
    def build_parameters(b):
        # Build additional indexing sets
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

        # Check options for alpha rule
        if (
            b.config.equation_of_state_options is not None
            and "alpha_rule" in b.config.equation_of_state_options
        ):
            b.config.equation_of_state_options["alpha_rule"].build_parameters(b)
        else:
            DefaultAlphaRule.build_parameters(b)

        # Check options for tau rule
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

        # Create a list that includes the apparent species with
        # dissociation species.
        b.apparent_dissociation_species_list = []
        for a in b.params.apparent_species_set:
            if "dissociation_species" in b.params.get_component(a).config:
                b.apparent_dissociation_species_list.append(a)
        b.apparent_dissociation_species_set = pyo.Set(
            initialize=b.apparent_dissociation_species_list,
            doc="Set of apparent dissociated species",
        )
        assert (
            len(b.apparent_dissociation_species_set) == 1
        ), "This model does not support more than one electrolyte."

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
        elif b.params.config.parameter_data["hydration_model"] == "stepwise_hydration":
            b.constant_hydration = False
            params_for_stepwise_hydration = [
                "hydration_number",
                "ionic_radius",
                "partial_vol_mol",
                "min_hydration_number",
                "number_sites",
            ]
            for ion in b.params.ion_set:
                for k in params_for_stepwise_hydration:
                    if k not in b.params.config.components[ion]["parameter_data"]:
                        raise BurntToast(
                            "Missing '{}' parameter for {}. Please, include this parameter to the configuration dictionary to be able to use the stepwise hydration model.".format(
                                k, ion
                            )
                        )
        else:
            raise BurntToast(
                "'{}' is not a hydration model included in the refined eNRTL, but try again using one of the supported models: 'constant_hydration' or 'stepwise_hydration'".format(
                    b.params.config.parameter_data["hydration_model"]
                )
            )

        # Declare electrolyte and ions parameters in the configuration
        # dictionary in 'parameter_data' as Pyomo variables 'Var' with
        # fixed values and default units given in the 'units_dict'
        # below. First, a default set of units is declared followed by
        # an assertion to make sure the parameters given in the
        # configuration dictionary are the same as the ones given in
        # the default 'units_dict'. Note: If the units are provided in
        # the config dict, units should be provided as the second term
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
        # constant for electrolyte.
        for ap in b.apparent_dissociation_species_set:
            for i in b.params.config.components[ap]["parameter_data"].keys():
                b.add_component(
                    i,
                    pyo.Var(
                        b.apparent_dissociation_species_set,
                        units=units_dict[i],
                        doc=f"{i} parameter [{units_dict[i]}]",
                    ),
                )

        for ap in b.apparent_dissociation_species_set:
            for i in b.params.config.components[ap]["parameter_data"].keys():
                bdata = b.params.config.components[ap]["parameter_data"][i]
                if isinstance(bdata, tuple):
                    getattr(b, i)[ap].fix(bdata[0] * bdata[1])
                else:
                    getattr(b, i)[ap].fix(bdata * units_dict[i])

        # Declare dictionary for stoichiometric coefficient using data
        # from configuration dictionary.
        b.stoichiometric_coeff = {}
        if len(b.apparent_dissociation_species_set) == 1:
            a = b.apparent_dissociation_species_set.first()
            for i in b.params.config.components[a]["dissociation_species"]:
                b.stoichiometric_coeff[i] = (
                    b.params.config.components[a]["dissociation_species"].get(i, [])
                    * pyunits.dimensionless
                )

        # Add beta constant, which represents the radius of
        # electrostricted water in the hydration shell of ions and it
        # is specific to the type of electrolyte.
        # Beta is a parameter determined by the charge of the ion pairs, like NaCl is 1-1, Na2SO4 is 1-2
        # Beta is obtained using parameter estimation by Xi Yang ref[3] (page 35 values multiplied by 5.187529),
        # original data used for parameter estimation are from ref[4].
        b.add_component(
            "beta",
            pyo.Var(
                units=units_dict["beta"],
                doc="{} parameter [{}]".format("beta", units_dict["beta"]),
            ),
        )
        if len(b.params.cation_set) == 1:
            c = b.params.cation_set.first()
            if len(b.params.anion_set) == 1:
                a = b.params.anion_set.first()
            if (abs(cobj(b, c).config.charge) == 1) and (
                abs(cobj(b, a).config.charge) == 1
            ):
                b.beta.fix(0.9695492)
            elif (abs(cobj(b, c).config.charge) == 2) and (
                abs(cobj(b, a).config.charge) == 1
            ):
                b.beta.fix(0.9192301707)
            elif (abs(cobj(b, c).config.charge) == 1) and (
                abs(cobj(b, a).config.charge) == 2
            ):
                b.beta.fix(0.8144420812)
            elif (abs(cobj(b, c).config.charge) == 2) and (
                abs(cobj(b, a).config.charge) == 2
            ):
                b.beta.fix(0.1245007)
            elif (abs(cobj(b, c).config.charge) == 3) and (
                abs(cobj(b, a).config.charge) == 1
            ):
                b.beta.fix(0.7392229)
            else:
                raise BurntToast(
                    f"'beta' constant not known for system with cation with charge +{cobj(b, c).config.charge} and anion with charge {cobj(b, a).config.charge}. Please contact the development team if you are interested in solving a case not supported by this method.".format(
                        app
                    )
                )

                print()

        # Declare the (a) total stoichiometric coefficient for
        # electrolyte and the (b) total hydration number as Pyomo
        # parameters 'Param'. The 'total_hydration_init' is used in
        # the constant hydration model and as an initial value in the
        # stepwise hydration model.
        b.vca = pyo.Param(
            initialize=(sum(b.stoichiometric_coeff[j] for j in b.params.ion_set)),
            units=pyunits.dimensionless,
            doc="Total stoichiometric coefficient for electrolyte [dimensionless]",
        )
        b.total_hydration_init = pyo.Param(
            initialize=(
                sum(
                    b.stoichiometric_coeff[i] * b.hydration_number[i]
                    for i in b.params.ion_set
                )
            ),
            units=pyunits.dimensionless,
            doc="Initial total hydration number [dimensionless]",
        )

        # Convert given molar density to mass units (kg/m3) as a Pyomo
        # Expression. This density is needed in the calculation of
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

        # Add total hydration terms for each hydration model
        if b.constant_hydration:
            # In constant hydration model, the total hydration term is
            # an Expression and it is equal to the total hydration
            # parameter calculated using hydration numbers of ions.
            def rule_constant_total_hydration(b):
                return b.total_hydration_init

            b.add_component(
                pname + "_total_hydration",
                pyo.Expression(
                    rule=rule_constant_total_hydration,
                    doc="Total hydration number [dimensionless]",
                ),
            )
        else:
            # In the stepwise hydration model, a Pyomo variable 'Var'
            # is declared for the total hydration term and it is
            # calculated using the equations in function
            # 'rule_nonconstant_total_hydration_term' below. NOTES:
            # Improve initial value and bounds for this variable.
            if value(b.total_hydration_init) <= 0:
                min_val = -1e3
            else:
                min_val = 1e-3

            b.add_component(
                pname + "_total_hydration",
                pyo.Var(
                    bounds=(min_val, abs(b.total_hydration_init) * 1000),
                    initialize=b.total_hydration_init,
                    units=pyunits.dimensionless,
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
                    return (
                        b.stoichiometric_coeff[j] * b.flow_mol_phase_comp_true[pname, j]
                    )
                elif j in b.params.solvent_set:
                    # NOTES: 'flow_mol' could be either of cation or
                    # anion since we assume both flows are the same.
                    if len(b.params.cation_set) == 1:
                        c = b.params.cation_set.first()
                        return (
                            b.flow_mol_phase_comp_true[pname, j]
                            - total_hydration * b.flow_mol_phase_comp_true[pname, c]
                        )

        b.add_component(
            pname + "_n",
            pyo.Expression(
                b.params.true_species_set,
                rule=rule_n,
                doc="Moles of dissociated electrolytes",
            ),
        )

        # Effective mol fraction X
        def rule_X(b, j):
            n = getattr(b, pname + "_n")
            if j in b.params.ion_set:
                z = abs(cobj(b, j).config.charge)
            else:
                z = 1
            return z * n[j] / sum(n[i] for i in b.params.true_species_set)

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
            return X[j] / sum(X[i] for i in dom)  # Eqns 36 and 37 from ref[1]
            # Y is a charge ratio, and thus independent of x for symmetric state
            # TODO: This may need to change for the unsymmetric state

        b.add_component(
            pname + "_Y",
            pyo.Expression(b.params.ion_set, rule=rule_Y, doc="Charge composition"),
        )

        # ---------------------------------------------------------------------
        # Long-range terms
        # Eqn 22 from ref 6
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

        def rule_Xp(b, e):
            if len(b.params.solvent_set) == 1:
                s = b.params.solvent_set.first()

                return (
                    b.stoichiometric_coeff[e]
                    * b.flow_mol_phase_comp_true[pname, e]
                    / (
                        b.flow_mol_phase_comp_true[pname, s]
                        + b.vca * b.flow_mol_phase_comp_true[pname, e]
                    )
                )

        b.add_component(
            pname + "_Xp",
            pyo.Expression(
                b.params.ion_set,
                rule=rule_Xp,
                doc="Mole fraction at unhydrated level [dimensionless]",
            ),
        )
        # Function to calculate Volume of Solution [m3], this function is a combination of Eqn 9 & 10 from ref [3]

        # Average molar volume of solvent
        def rule_vol_mol_solvent(b):
            # Equation from ref[3], page 14

            Vo = getattr(b, pname + "_Vo")
            Vq = getattr(b, pname + "_Vq")
            Xp = getattr(b, pname + "_Xp")
            dens_mass = getattr(b, pname + "_dens_mass")  # for first solvent

            if len(b.params.solvent_set) == 1:
                s = b.params.solvent_set.first()
                return b.flow_mol_phase_comp_true[pname, s] * b.params.get_component(
                    s
                ).mw / dens_mass + sum(
                    b.stoichiometric_coeff[e] * b.flow_mol_phase_comp_true[pname, e] *
                    # Intrinsic molar volume from Eq. 10 in ref[3]
                    (Vq[e] + (Vo[e] - Vq[e]) * sum(Xp[j] for j in b.params.ion_set))
                    for e in b.params.ion_set
                )

        b.add_component(
            pname + "_vol_mol_solvent",
            pyo.Expression(
                rule=rule_vol_mol_solvent, doc="Mean molar volume of solvent [m3]"
            ),
        )

        # Partial molar volume of solution
        # Partial Molar Volume of Solvent/Cation/Anion (m3/mol) derived from Eqn 10 & 11 from ref [3]
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
                return b.params.get_component(j).mw / dens_mass - sum(
                    Xp[i] * (Vo[i] - Vq[i]) for i in b.params.ion_set
                )

        b.add_component(
            pname + "_vol_mol_solution",
            pyo.Expression(
                b.params.true_species_set,
                rule=rule_vol_mol_solution,
                doc="Partial molar volume of solvent [m3/mol]",
            ),
        )

        # Ionic Strength
        def rule_I(b):
            v = getattr(b, pname + "_vol_mol_solvent")
            n = getattr(b, pname + "_n")

            return (
                0.5
                / v
                * sum(
                    n[c] *
                    # zz or true ionic charge of components
                    # (Pitzer's equation)
                    (abs(b.params.get_component(c).config.charge) ** 2)
                    for c in b.params.ion_set
                )
            )

        b.add_component(
            pname + "_ionic_strength",
            pyo.Expression(rule=rule_I, doc="Ionic strength [m3 mol]"),
        )

        # Mean relative permitivity of solvent
        def rule_eps_solvent(b):  # Eqn 78 from ref[1]
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
                doc="Mean relative permittivity  of solvent [dimensionless]",
            ),
        )

        # Distance of Closest Approach (m)
        # Eqn 12 from ref [3]
        def rule_ar(b):
            b.distance_species = pyo.Param(
                initialize=1.9277,
                mutable=False,
                units=pyunits.angstrom,
                doc="Distance between a solute and solvent",
            )
            return pyo.units.convert(
                sum(
                    (
                        max(
                            0,
                            sum(value(b.hydration_number[i]) for i in b.params.ion_set)
                            / 2,
                        )
                        * (b.beta * b.distance_species) ** 3
                        + b.ionic_radius[i] ** 3
                    )
                    ** (1 / 3)
                    for i in b.params.ion_set
                ),
                to_units=pyunits.m,
            )

        b.add_component(
            pname + "_ar",
            pyo.Expression(rule=rule_ar, doc="Distance of closest approach [m]"),
        )

        # Functions to calculate parameters for long-range equations
        # b term
        # ref[3] eq#[2] first line
        # b_term = kappa*ar/I. The I term here is the ionic strength. kappa is from ref[3] eq+2 first line

        def rule_b_term(b):
            eps = getattr(b, pname + "_relative_permittivity_solvent")
            eps0 = Constants.vacuum_electric_permittivity
            ar = getattr(b, pname + "_ar")

            return (
                ar
                * (
                    2
                    * Constants.faraday_constant**2
                    / (eps0 * eps * Constants.gas_constant * b.temperature)
                )
                ** 0.5
            )

        b.b_term = pyo.Expression(rule=rule_b_term)

        def rule_A_DH(b):
            eps = getattr(b, pname + "_relative_permittivity_solvent")
            eps0 = Constants.vacuum_electric_permittivity
            ar = getattr(b, pname + "_ar")

            return (
                1
                / (16 * Constants.pi * Constants.avogadro_number)
                * (b.b_term / ar) ** 3
            )

        b.add_component(
            pname + "_A_DH",
            pyo.Expression(rule=rule_A_DH, doc="Debye-Huckel parameter"),
        )

        # Long-range (PDH) contribution to activity coefficient. Eqn derived from ref[5].
        def rule_log_gamma_pdh(b, j):
            A = getattr(b, pname + "_A_DH")
            Ix = getattr(b, pname + "_ionic_strength")
            v = getattr(b, pname + "_vol_mol_solution")

            if j in molecular_set:
                return (
                    v[j]
                    * 2
                    * A
                    / (b.b_term**3)
                    * (
                        (1 + b.b_term * (Ix**0.5))
                        - 1 / (1 + b.b_term * (Ix**0.5))
                        - 2 * log(1 + b.b_term * (Ix**0.5))
                    )
                )
            elif j in b.params.ion_set:
                z = abs(cobj(b, j).config.charge)
                return (-A * (z**2) * (Ix**0.5)) / (1 + b.b_term * (Ix**0.5)) + v[
                    j
                ] * 2 * A / (b.b_term**3) * (
                    (1 + b.b_term * (Ix**0.5))
                    - 1 / (1 + b.b_term * (Ix**0.5))
                    - 2 * log(1 + b.b_term * (Ix**0.5))
                )
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

        # For the symmetric state, all of these are independent of composition
        # TODO: For the unsymmetric state, it may be necessary to recalculate
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
                # Eqn 32 from ref[1]
                return sum(
                    Y[k] * alpha_rule(b, pobj, (i + ", " + k), j, b.temperature)
                    for k in b.params.anion_set
                )
            elif j in b.params.cation_set and i in molecular_set:
                # Eqn 32 from ref[1]
                return sum(
                    Y[k] * alpha_rule(b, pobj, (j + ", " + k), i, b.temperature)
                    for k in b.params.anion_set
                )
            elif i in b.params.anion_set and j in molecular_set:
                # Eqn 33 from ref[1]
                return sum(
                    Y[k] * alpha_rule(b, pobj, (k + ", " + i), j, b.temperature)
                    for k in b.params.cation_set
                )
            elif j in b.params.anion_set and i in molecular_set:
                # Eqn 33 from ref[1]
                return sum(
                    Y[k] * alpha_rule(b, pobj, (k + ", " + j), i, b.temperature)
                    for k in b.params.cation_set
                )
            elif i in b.params.cation_set and j in b.params.anion_set:
                # Eqn 34 from ref[1]
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
                # Eqn 35 from ref[1]
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

            def _G_appr(b, pobj, i, j, T):  # Eqn 23 from ref[1]
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
                # Eqn 38 from ref[1]
                return sum(
                    Y[k] * _G_appr(b, pobj, (i + ", " + k), j, b.temperature)
                    for k in b.params.anion_set
                )
            elif i in molecular_set and j in b.params.cation_set:
                # Eqn 40 from ref[1]
                return sum(
                    Y[k] * _G_appr(b, pobj, i, (j + ", " + k), b.temperature)
                    for k in b.params.anion_set
                )
            elif i in b.params.anion_set and j in molecular_set:
                # Eqn 39 from ref[1]
                return sum(
                    Y[k] * _G_appr(b, pobj, (k + ", " + i), j, b.temperature)
                    for k in b.params.cation_set
                )
            elif i in molecular_set and j in b.params.anion_set:
                # Eqn 41 from ref[1]
                return sum(
                    Y[k] * _G_appr(b, pobj, i, (k + ", " + j), b.temperature)
                    for k in b.params.cation_set
                )
            elif i in b.params.cation_set and j in b.params.anion_set:
                # Eqn 42 from ref[1]
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
                # Eqn 43 from ref[1]
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
                # Eqn 44 from ref[1]
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

        def _calculate_tau_alpha(b):
            """This function calculates and sets tau and alpha with four indicies
            as mutable parameters. Note that the ca_m terms refer
            to the parameters with four indicies as cm_mm and am_mm

            """

            b.alpha_ij_ij = pyo.Param(
                b.params.true_species_set,
                b.params.true_species_set,
                b.params.true_species_set,
                b.params.true_species_set,
                mutable=True,
                initialize=1,
            )
            b.tau_ij_ij = pyo.Param(
                b.params.true_species_set,
                b.params.true_species_set,
                b.params.true_species_set,
                b.params.true_species_set,
                mutable=True,
                initialize=1,
            )

            for c in b.params.cation_set:
                for a in b.params.anion_set:
                    for m in molecular_set:
                        b.alpha_ij_ij[c, m, m, m] = alpha_rule(
                            b, pobj, (c + ", " + a), m, b.temperature
                        )
                        b.alpha_ij_ij[a, m, m, m] = alpha_rule(
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
                    for m in b.params.solvent_set:
                        b.tau_ij_ij[a, c, a, c] = 0
                        b.tau_ij_ij[c, a, c, a] = 0
                        b.tau_ij_ij[m, c, a, c] = (
                            tau_rule(b, pobj, (c + ", " + a), m, b.temperature)
                            - tau_rule(b, pobj, (c + ", " + a), m, b.temperature)
                            + tau_rule(b, pobj, m, (c + ", " + a), b.temperature)
                        )
                        b.tau_ij_ij[m, a, c, a] = (
                            tau_rule(b, pobj, (c + ", " + a), m, b.temperature)
                            - tau_rule(b, pobj, (c + ", " + a), m, b.temperature)
                            + tau_rule(b, pobj, m, (c + ", " + a), b.temperature)
                        )

            return b.tau_ij_ij, b.alpha_ij_ij

        _calculate_tau_alpha(b)

        def _calculate_G(b):
            """This function calculates G with three and four indicies as a
            mutable parameter. With three indicies, the only one
            that is calculated is G_ca.m (G_cm.mm, G_am.mm) since
            it is needed in the refined eNRTL. Note that this G is
            not needed in the general NRTL, so this function is not
            included in the method

            """

            def _G_appr(b, pobj, i, j, T):  # Eqn 23 from ref[1]
                if i != j:
                    return exp(
                        -alpha_rule(b, pobj, i, j, T) * tau_rule(b, pobj, i, j, T)
                    )
                else:
                    return 1

            b.G_ij_ij = pyo.Param(
                b.params.true_species_set,
                b.params.true_species_set,
                b.params.true_species_set,
                b.params.true_species_set,
                mutable=True,
                initialize=1,
            )

            for m in molecular_set:
                for a in b.params.anion_set:
                    for c in b.params.cation_set:
                        b.G_ij_ij[c, m, m, m] = _G_appr(
                            b, pobj, (c + ", " + a), m, b.temperature
                        )
                        b.G_ij_ij[a, m, m, m] = _G_appr(
                            b, pobj, (c + ", " + a), m, b.temperature
                        )

            for t in b.params.true_species_set:
                for a in b.params.anion_set:
                    for c in b.params.cation_set:
                        if t == c:
                            b.G_ij_ij[t, c, a, c] = 0
                        elif t == a:
                            b.G_ij_ij[t, a, c, a] = 0
                        else:
                            b.G_ij_ij[t, c, a, c] = exp(
                                -b.alpha_ij_ij[t, c, a, c] * b.tau_ij_ij[t, c, a, c]
                            )
                            b.G_ij_ij[t, a, c, a] = exp(
                                -b.alpha_ij_ij[t, c, a, c] * b.tau_ij_ij[t, a, c, a]
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
                doc="Local contribution to activity coefficient",
            ),
        )

        # Calculate stepwise total hydration term. This equation
        # calculates a non-constant hydration term using the
        # long-range interactions and given parameters, such as
        # hydration constant, minimum hydration numbers, and number of
        # sites.
        if not b.constant_hydration:

            def rule_total_hydration_stepwise(b):
                X = getattr(b, pname + "_X")
                lc = getattr(b, pname + "_log_gamma_lc")
                total_hydration = getattr(b, pname + "_total_hydration")

                # NOTES: Select the first solvent and the first
                # apparent specie with dissociation species.
                if len(b.params.solvent_set) == 1:
                    s = b.params.solvent_set.first()
                    if len(b.apparent_dissociation_species_set) == 1:
                        ap = b.apparent_dissociation_species_set.first()

                        b.constant_a = pyo.Var(
                            b.params.ion_set,
                            units=pyunits.dimensionless,
                            doc="Constant factor in stepwise hydration model",
                        )
                        for ion in b.params.ion_set:
                            if ion in b.params.cation_set:
                                b.constant_a[ion].fix(1)
                            elif ion in b.params.anion_set:
                                b.constant_a[ion].fix(0)

                        return total_hydration == sum(
                            b.stoichiometric_coeff[i] * b.min_hydration_number[i]
                            + b.stoichiometric_coeff[i]
                            * (
                                b.number_sites[i]
                                - b.constant_a[i] * b.min_hydration_number[i]
                            )
                            * b.constant_a[i]
                            * b.hydration_constant[ap]
                            * X[s]
                            * exp(lc[s])
                            / (
                                1
                                + b.constant_a[i]
                                * b.hydration_constant[ap]
                                * X[s]
                                * exp(lc[s])
                            )
                            for i in b.params.ion_set
                        )

            b.add_component(
                pname + "_total_hydration_stepwise_eq",
                pyo.Constraint(rule=rule_total_hydration_stepwise),
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
                n = 0
                d = 0

                for s in dspec:
                    dobj = b.params.get_component(s)
                    ln_g = getattr(b, pname + "_log_gamma")[s]
                    n += b.stoichiometric_coeff[s] * ln_g
                    d += b.stoichiometric_coeff[s]

                return n / d
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

        # Mean molal log_gamma of ions, Eqn 20 from ref [3] for constant hydration model and Eqn 21 from ref [3] for stepwise hydration model
        def rule_log_gamma_molal(b):
            X = getattr(b, pname + "_X")
            lc = getattr(b, pname + "_log_gamma_lc")
            log_gamma_appr = getattr(b, pname + "_log_gamma_appr")
            n = getattr(b, pname + "_n")
            total_hydration = getattr(b, pname + "_total_hydration")

            # NOTES: Select the first solvent and apparent specie.
            if len(b.params.solvent_set) == 1:
                s = b.params.solvent_set.first()
                if len(b.apparent_dissociation_species_set) == 1:
                    ap = b.apparent_dissociation_species_set.first()

                    # NOTES: 'flow_mol' could be either of cation or
                    # anion since we assume both flows are the same.
                    if len(b.params.cation_set) == 1:
                        c = b.params.cation_set.first()

                        if b.constant_hydration:
                            return (
                                log_gamma_appr[ap]
                                - (total_hydration / b.vca) * log(X[s] * exp(lc[s]))
                                - log(
                                    1
                                    + (b.vca - total_hydration)
                                    / (
                                        b.flow_mol_phase_comp_true[pname, s]
                                        / b.flow_mol_phase_comp_true[pname, c]
                                    )
                                )
                            )
                        else:
                            sum_n = sum(n[i] for i in b.params.true_species_set)
                            return log_gamma_appr[ap] + (1 / b.vca) * (
                                b.vca
                                * log(b.flow_mol_phase_comp_true[pname, s] / sum_n)
                                - sum(
                                    b.stoichiometric_coeff[i]
                                    * b.min_hydration_number[i]
                                    for i in b.params.ion_set
                                )
                                * (log(X[s]) + lc[s])
                                + sum(
                                    b.stoichiometric_coeff[i]
                                    * (
                                        b.number_sites[i]
                                        - b.constant_a[i] * b.min_hydration_number[i]
                                    )
                                    * log(
                                        (1 + b.constant_a[i] * b.hydration_constant[ap])
                                        / (
                                            1
                                            + b.constant_a[i]
                                            * b.hydration_constant[ap]
                                            * X[s]
                                            * exp(lc[s])
                                        )
                                    )
                                    for i in b.params.ion_set
                                )
                            )

        b.add_component(
            pname + "_log_gamma_molal",
            pyo.Expression(
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
    def pressure_osm_phase(b, p):
        return (
            -rENRTL.gas_constant(b)
            * b.temperature
            * b.log_act_phase_solvents[p]
            / b.vol_mol_phase[p]
        )

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

    # Indicies in expressions use same names as source paper
    # mp = m' and so on

    molecular_set = b.params.solvent_set | b.params.solute_set
    aqu_species = b.params.true_species_set - b.params._non_aqueous_set

    if (pname, s) not in b.params.true_phase_component_set:
        # Non-aqueous component
        return Expression.Skip

    if s in b.params.cation_set:
        c = s
        Z = b.params.get_component(c).config.charge

        # Eqn 6 from ref[2]. This equation uses G and tau with four
        # indicies and ignores simplifications.
        return Z * (
            # Term 1
            sum(
                (X[m] / sum(X[i] * G[i, m] for i in aqu_species))
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
                        (X[a] / sum(X[cp] for cp in b.params.cation_set))
                        * (b.G_ij_ij[c, m, m, m] - G[a, m])
                        * (  # Gam instead of Gcm
                            (
                                (b.alpha_ij_ij[a, m, m, m] * tau[a, m] - 1)
                                / b.alpha_ij_ij[a, m, m, m]  # tau_am instead of tau_cm
                            )
                            - (
                                sum(X[i] * G[i, m] * tau[i, m] for i in aqu_species)
                                / sum(X[i] * G[i, m] for i in aqu_species)
                            )
                        )
                        for a in b.params.anion_set
                    )
                )
                for m in molecular_set
            )
            +
            # Term 2
            sum(
                (X[a] / sum(X[ap] for ap in b.params.anion_set))
                * (
                    sum(
                        X[i] * b.G_ij_ij[i, c, a, c] * b.tau_ij_ij[i, c, a, c]
                        for i in aqu_species - b.params.cation_set
                    )
                    / sum(
                        X[i] * b.G_ij_ij[i, c, a, c]
                        for i in aqu_species - b.params.cation_set
                    )
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
                                - (
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
                            )
                            + sum(
                                (X[m] / sum(X[cpp] for cpp in b.params.cation_set))
                                * b.G_ij_ij[m, a, cp, a]
                                * ((b.G_ij_ij[a, m, m, m] - G[a, m]) / G[a, m])
                                * (
                                    (
                                        (
                                            b.alpha_ij_ij[c, m, m, m]
                                            * b.tau_ij_ij[m, a, cp, a]
                                            - 1
                                        )
                                        / b.alpha_ij_ij[c, m, m, m]
                                    )
                                    - (
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
                                )
                                for m in molecular_set
                            )
                        )
                        for cp in b.params.cation_set
                    )
                    + (
                        (1 / sum(X[cpp] for cpp in b.params.cation_set))
                        * (
                            (
                                sum(
                                    X[i]
                                    * b.G_ij_ij[i, a, c, a]
                                    * b.tau_ij_ij[i, a, c, a]
                                    for i in (aqu_species - b.params.anion_set)
                                )
                                / sum(
                                    X[i] * b.G_ij_ij[i, a, c, a]
                                    for i in (aqu_species - b.params.anion_set)
                                )
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
    elif s in b.params.anion_set:
        a = s
        Z = abs(b.params.get_component(a).config.charge)

        # Eqn 7 from ref[2]. This equation uses G with four indicies
        # and ignores simplifications.
        return Z * (
            # Term 1
            sum(
                (X[m] / sum(X[i] * G[i, m] for i in aqu_species))
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
                        (X[c] / sum(X[ap] for ap in b.params.anion_set))
                        * (b.G_ij_ij[c, m, m, m] - G[c, m])
                        * (
                            (
                                (b.alpha_ij_ij[c, m, m, m] * tau[c, m] - 1)
                                / b.alpha_ij_ij[c, m, m, m]
                            )
                            - (
                                sum(X[i] * G[i, m] * tau[i, m] for i in aqu_species)
                                / sum(X[i] * G[i, m] for i in aqu_species)
                            )
                        )
                        for c in b.params.cation_set
                    )
                )
                for m in molecular_set
            )
            +
            # Term 2
            sum(
                (X[c] / sum(X[cp] for cp in b.params.cation_set))
                * (
                    sum(
                        X[i] * b.G_ij_ij[i, a, c, a] * b.tau_ij_ij[i, a, c, a]
                        for i in aqu_species - b.params.anion_set
                    )
                    / sum(
                        X[i] * b.G_ij_ij[i, a, c, a]
                        for i in aqu_species - b.params.anion_set
                    )
                )
                for c in b.params.cation_set
            )
            +
            # Term 3
            sum(
                X[c]
                * (
                    sum(
                        (X[ap] / sum(X[app] for app in b.params.anion_set))
                        * (
                            1
                            / sum(
                                X[i] * b.G_ij_ij[i, c, ap, c]
                                for i in (aqu_species - b.params.cation_set)
                            )
                        )
                        * (
                            b.G_ij_ij[a, c, ap, c]
                            * (
                                b.tau_ij_ij[a, c, ap, c]
                                - (
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
                            )
                            + sum(
                                (X[m] / sum(X[app] for app in b.params.anion_set))
                                * b.G_ij_ij[m, c, ap, c]
                                * ((b.G_ij_ij[c, m, m, m] - G[c, m]) / G[c, m])
                                * (
                                    (
                                        (
                                            b.alpha_ij_ij[c, m, m, m]
                                            * b.tau_ij_ij[m, c, ap, c]
                                            - 1
                                        )
                                        / b.alpha_ij_ij[c, m, m, m]
                                    )
                                    - (
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
                                )
                                for m in molecular_set
                            )
                        )
                        for ap in b.params.anion_set
                    )
                )
                + (
                    (1 / sum(X[app] for app in b.params.anion_set))
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
                for c in b.params.cation_set
            )
        )
    else:
        m = s

        # Eqn 8 from ref[2]
        return (
            sum(X[i] * G[i, m] * tau[i, m] for i in aqu_species)
            / sum(X[i] * G[i, m] for i in aqu_species)
            + sum(
                (X[mp] * G[m, mp] / sum(X[i] * G[i, mp] for i in aqu_species))
                * (
                    tau[m, mp]
                    - (
                        sum(X[i] * G[i, mp] * tau[i, mp] for i in aqu_species)
                        / sum(X[i] * G[i, mp] for i in aqu_species)
                    )
                )
                for mp in molecular_set
            )
            + sum(
                (
                    X[c]
                    * G[m, c]
                    / sum(X[i] * G[i, c] for i in (aqu_species - b.params.cation_set))
                )
                * (
                    tau[m, c]
                    - (
                        sum(
                            X[i] * G[i, c] * tau[i, c]
                            for i in (aqu_species - b.params.cation_set)
                        )
                        / sum(
                            X[i] * G[i, c] for i in (aqu_species - b.params.cation_set)
                        )
                    )
                )
                for c in b.params.cation_set
            )
            + sum(
                (
                    X[a]
                    * G[m, a]
                    / sum(X[i] * G[i, a] for i in (aqu_species - b.params.anion_set))
                )
                * (
                    tau[m, a]
                    - (
                        sum(
                            X[i] * G[i, a] * tau[i, a]
                            for i in (aqu_species - b.params.anion_set)
                        )
                        / sum(
                            X[i] * G[i, a] for i in (aqu_species - b.params.anion_set)
                        )
                    )
                )
                for a in b.params.anion_set
            )
        )


def log_gamma_inf(b, pname, s, X, G, tau):
    """General function for calculating infinite dilution contributions"""

    if (pname, s) not in b.params.true_phase_component_set:
        # Non-aqueous component
        return Expression.Skip

    # Select one solvent
    if len(b.params.solvent_set) == 1:
        w = b.params.solvent_set.first()

    if s in b.params.cation_set:
        c = s
        Z = b.params.get_component(c).config.charge

        # Eqn 9 from ref[2]
        return Z * (
            sum(
                (X[a] / sum(X[ap] for ap in b.params.anion_set))
                * b.tau_ij_ij[w, c, a, c]
                for a in b.params.anion_set
            )
            + G[c, w] * tau[c, w]
            + sum(
                (X[a] / sum(X[cp] for cp in b.params.cation_set))
                * (b.G_ij_ij[a, w, w, w] - G[a, w])
                * (
                    (b.alpha_ij_ij[a, w, w, w] * tau[a, w] - 1)
                    / b.alpha_ij_ij[a, w, w, w]
                )
                for a in b.params.anion_set
            )
            - sum(
                X[a]
                * (
                    sum(
                        (X[cp] / sum(X[cpp] for cpp in b.params.cation_set))
                        * (1 / sum(X[cpp] for cpp in b.params.cation_set))
                        * (
                            (b.G_ij_ij[c, w, w, w] - G[a, w])
                            / (b.alpha_ij_ij[c, w, w, w] * G[a, w])
                        )
                        for cp in b.params.cation_set
                    )
                    - (1 / sum(X[cpp] for cpp in b.params.cation_set))
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

    elif s in b.params.anion_set:
        a = s
        Z = abs(b.params.get_component(a).config.charge)

        # Eqn 10 from ref[2]
        return Z * (
            sum(
                (X[c] / sum(X[cp] for cp in b.params.cation_set))
                * b.tau_ij_ij[w, a, c, a]
                for c in b.params.cation_set
            )
            + G[a, w] * tau[a, w]
            + sum(
                (X[c] / sum(X[ap] for ap in b.params.anion_set))
                * (b.G_ij_ij[a, w, w, w] - G[c, w])
                * (
                    (b.alpha_ij_ij[a, w, w, w] * tau[c, w] - 1)
                    / b.alpha_ij_ij[a, w, w, w]
                )
                for c in b.params.cation_set
            )
            - sum(
                X[c]
                * (
                    sum(
                        (X[ap] / sum(X[app] for app in b.params.anion_set))
                        * (1 / sum(X[app] for app in b.params.anion_set))
                        * (
                            (b.G_ij_ij[a, w, w, w] - G[c, w])
                            / (b.alpha_ij_ij[a, w, w, w] * G[c, w])
                        )
                        for ap in b.params.anion_set
                    )
                    - (1 / sum(X[app] for app in b.params.anion_set))
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

    else:
        m = s

        return tau[m, m] + G[m, m] * tau[m, m]
