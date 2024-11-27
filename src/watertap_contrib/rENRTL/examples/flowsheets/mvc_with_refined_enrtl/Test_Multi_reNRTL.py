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
# Copyright 2023-2024, Pengfei Xu, Matthew D. Stuber, and the University
# of Connecticut.
#
# Please see the files COPYRIGHT.md and LICENSE.md for full copyright and
# license information.
#################################################################################
"""
Test for multi-electrolyte refined eNRTL (Multi r-eNRTL) methods.

This module tests values of key terms in the Multi r-eNRTL method.

Author: Pengfei Xu (University of Connecticut)

References:

[1] Xi Yang, Paul I. Barton, and George M. Bollas, Refined
electrolyte-NRTL model: Inclusion of hydration for the detailed
description of electrolyte solutions. Part I: Single electrolytes up
to moderate concentrations, single salts up to solubility limit.

Note that ref [1] contains the primary parameter values and equations.

[2] Y. Marcus, Ion solvation, Wiley-Interscience, New York, 1985.

[3] Y. Marcus, Thermodynamics of solvation of ions. Part 5.—Gibbs free energy of hydration at
298.15 K, J. Chem. Soc., Faraday Trans. 87 (1991) 2995–2999. doi:10.1039/FT9918702995

Experimental data reference:

[4] Galleguillos-Castro, H. R., Hernández-Luis, F., Fernández-Mérida, 
L., & Esteso, M. A. (1999). Thermodynamic Study of the NaCl + Na2SO4 + H2O 
System by Emf Measurements at Four Temperatures. Journal of Solution 
Chemistry, 28(6), 791–807. https://doi.org/10.1023/A:1021780414613 


[5] The reference values used for comparison in this test file are based on the model's
output from the author's validation file, covering all reference properties for this test.
"""


# Test for refined eNRTL method

import pytest

from pyomo.environ import (
    ConcreteModel,
    Expression,
    exp,
    log,
    Set,
    units as pyunits,
    value,
    Var,
)
from pyomo.util.check_units import assert_units_equivalent

from idaes.core import AqueousPhase, Solvent, Solute, Apparent, Anion, Cation
from idaes.core.util.constants import Constants
from idaes.models.properties.modular_properties.eos.enrtl import ENRTL
from refined_enrtl_multi import rENRTL
from idaes.models.properties.modular_properties.base.generic_property import (
    GenericParameterBlock,
    StateIndex,
)
from idaes.models.properties.modular_properties.state_definitions import FTPx
from idaes.models.properties.modular_properties.pure.electrolyte import (
    relative_permittivity_constant,
)
from idaes.core.util.exceptions import ConfigurationError
import idaes.logger as idaeslog

import csv
import json
from math import exp, log
import matplotlib.pyplot as plt
import numpy as np

# Import Pyomo components
import pyomo.environ as pyo
from pyomo.environ import ConcreteModel, Expression, Set, Param
from pyomo.environ import units as pyunits, value

# Import IDAES libraries
from idaes.core.solvers import get_solver
from idaes.core import AqueousPhase, Solvent, Apparent, Anion, Cation, FlowsheetBlock
from idaes.models.properties.modular_properties.base.generic_property import (
    GenericParameterBlock,
    StateIndex,
)
from idaes.models.properties.modular_properties.state_definitions import FTPx, FpcTP
from idaes.models.properties.modular_properties.pure.electrolyte import (
    relative_permittivity_constant,
)

# Import refined eNRTL
from idaes.models.properties.modular_properties.eos.enrtl import ENRTL
from refined_enrtl_multi import rENRTL

from idaes.core.util.model_statistics import degrees_of_freedom


class Prop:
    def __init__(
        self,
        gamma1Calc,
        gamma2Calc,
        PDH,
        SR,
        SR_inf,
        IX,
        tau_maca,
        tau_mcac,
        n,
        X,
        Y,
        Xp,
        Vt,
        Vo,
        Vq,
        b,
        aravg,
        bw,
        AX,
        kappa,
    ):
        self.gamma1Calc = gamma1Calc
        self.gamma2Calc = gamma2Calc
        self.PDH = PDH
        self.SR = SR
        self.SR_inf = SR_inf
        self.IX = IX
        self.tau_maca = tau_maca
        self.tau_mcac = tau_mcac
        self.n = n
        self.X = X
        self.Y = Y
        self.Xp = Xp
        self.Vt = Vt
        self.Vo = Vo
        self.Vq = Vq
        self.b = b
        self.aravg = aravg
        self.bw = bw
        self.AX = AX
        self.kappa = kappa


# Add electrolyte and solvent to solve
solvent = "H2O"

elect_1 = "NaCl"
elect_2 = "Na2SO4"
cation_1 = "Na+"
anion_1 = "Cl-"
anion_2 = "SO4-"

electrolyte = [elect_1, elect_2]
cation = [cation_1]
anion = [anion_1, anion_2]

ion_pair = {}
ion_pair[elect_1] = "Na+, Cl-"
ion_pair[elect_2] = "Na+, SO4-"

# constant_hydration
hydration_model = "constant_hydration"


# Declare functions needed in the configuration dictionary, such as
# relative permittivity, molar density, and molar volume.
def relative_permittivity_expr(b, s, T):
    """Calculation of dielectric constant of water

    This equation requires the constants for dielectric constants estimation:
    AM -- Dielectric constant of water at room temperature [dimensionless]
    BM -- Temperature correction factor [K]
    CM -- Room temperature [K]
    """
    AM = 78.54003
    BM = 31989.38 * pyunits.K
    CM = 298.15 * pyunits.K
    return AM + BM * (1 / T - 1 / CM)


def dens_mol_water_expr(b, s, T):
    b.water_dens = pyo.Param(
        initialize=1,
        units=pyunits.g / pyunits.cm**3,
        mutable=True,
        doc="Water density",
    )
    b.water_mw = pyo.Param(initialize=18.01528e-3, units=pyunits.g / pyunits.mol)
    b.correction_param_a = pyo.Param(
        initialize=4.8e-6, units=pyunits.g / pyunits.cm**3 / pyunits.K**2
    )
    b.correction_param_b = pyo.Param(initialize=273.15, units=pyunits.K)
    b.water_dens_kgm3 = pyo.units.convert(
        b.water_dens, to_units=pyunits.kg / pyunits.m**3
    )

    return (
        b.water_dens_kgm3
        - pyo.units.convert(
            b.correction_param_a, to_units=pyunits.kg / pyunits.m**3 / pyunits.K**2
        )
        * (T - b.correction_param_b) ** 2
    ) / b.water_mw


def vol_mol_water_expr(b, s, T):
    # Calculate the vol mol as the inverse of the density function
    # given above
    return 1 / dens_mol_water_expr(b, s, T)


# Declare configuration dictionary for refined eNRTL
equation_of_state = rENRTL  # method class for config dic

configuration = {
    "components": {
        solvent: {
            "type": Solvent,
            "dens_mol_liq_comp": dens_mol_water_expr,
            "vol_mol_liq_comp": vol_mol_water_expr,
            "relative_permittivity_liq_comp": relative_permittivity_expr,
            "parameter_data": {
                "mw": (18.01528e-3, pyunits.kg / pyunits.mol),
                "relative_permittivity_liq_comp": relative_permittivity_expr,
            },
        },
        "NaCl": {
            "type": Apparent,
            "dissociation_species": {"Na+": 1, "Cl-": 1},  # stoichiometric coefficient
            "parameter_data": {"hydration_constant": 3.596},
        },
        "Na2SO4": {
            "type": Apparent,
            "dissociation_species": {"Na+": 2, "SO4-": 1},  # stoichiometric coefficient
            "parameter_data": {"hydration_constant": 1.022},
        },
        # hydration_number: Table 2 in ref [1]
        # ionic_radius: Table 1 in ref [3]
        # partial_vol_mol: Table 5.8 in ref [2]
        "Na+": {
            "type": Cation,
            "charge": "+1",
            "parameter_data": {
                "mw": 22.9897693e-3,
                "hydration_number": 1.51,
                "ionic_radius": 1.02,
                "partial_vol_mol": -7.6,
                "min_hydration_number": 0,
                "number_sites": 4,
            },
        },
        "Cl-": {
            "type": Anion,
            "charge": "-1",
            "parameter_data": {
                "mw": 35.45e-3,
                "hydration_number": 0.4994,
                "ionic_radius": 1.81,
                "partial_vol_mol": 24.2,
                "min_hydration_number": 0,
                "number_sites": 8,
            },
        },
        "SO4-": {
            "type": Anion,
            "charge": "-2",
            "parameter_data": {
                "mw": 96.066e-3,
                "hydration_number": -0.31,
                "ionic_radius": 2.30,
                "partial_vol_mol": 26.8,
                "min_hydration_number": 0,
                "number_sites": 2,
            },
        },
    },
    "phases": {
        "Liq": {
            "type": AqueousPhase,
            "equation_of_state": equation_of_state,
            "equation_of_state_options": {},
        }
    },
    "base_units": {
        "time": pyunits.s,
        "length": pyunits.m,
        "mass": pyunits.kg,
        "amount": pyunits.mol,
        "temperature": pyunits.K,
    },
    "state_definition": FTPx,
    "state_components": StateIndex.true,
    "pressure_ref": 101325,
    "temperature_ref": 298.15,
    "parameter_data": {
        "hydration_model": hydration_model,  # constant_hydration
        "Liq_tau": {
            # Table 3 from ref [1]
            (solvent, ion_pair[elect_1]): 7.951,
            (ion_pair[elect_1], solvent): -3.984,
            (solvent, ion_pair[elect_2]): 7.578,
            (ion_pair[elect_2], solvent): -3.532,
            (ion_pair[elect_1], ion_pair[elect_2]): -0.27889951,
            (ion_pair[elect_2], ion_pair[elect_1]): 0.27889951,
        },
    },
}


class Test_reNRTL_Multi(object):
    @pytest.fixture(scope="class")
    def model(self, molal_conc_elect1=None, molal_conc_elect2=None):
        m = ConcreteModel()
        m.fs = FlowsheetBlock(dynamic=False)
        m.fs.properties = GenericParameterBlock(**configuration)
        m.fs.state = m.fs.properties.build_state_block([0])
        m.fs.app_species_set = Set(initialize=[elect_1, elect_2])
        m.fs.species_set = Set(initialize=[cation_1, anion_1, anion_2, solvent])
        m.fs.solvent_set = Set(initialize=[solvent])
        m.fs.ion_set = Set(initialize=[cation_1, anion_1, anion_2])

        @m.fs.Expression(m.fs.app_species_set, doc="Calculated molal gamma")
        def mean_molal_gamma(b, ap):
            return b.state[0].Liq_log_gamma_molal[ap]

        return m

    @pytest.fixture(scope="class")
    def ExpData(self):
        # ref [4]
        gamma = [
            0.777732056,
            0.775228981,
            0.771738204,
            0.771755974,
            0.769236707,
            0.708810268,
            0.703930852,
            0.697461115,
            0.692643878,
            0.684715249,
            0.681396659,
            0.673426186,
            0.665043419,
            0.65761241,
            0.649067704,
            0.657400455,
            0.643798559,
            0.627191263,
            0.611420506,
            0.598260046,
            0.668682565,
            0.641741315,
            0.612167121,
            0.585962738,
            0.558303049,
            0.713723506,
            0.670563639,
            0.628188527,
            0.587773498,
            0.547507408,
            0.78204186,
            0.718488796,
            0.661149997,
            0.602809378,
            0.552458933,
            0.98666563,
            0.870362157,
            0.766125645,
            0.670038874,
            0.625604696,
        ]
        ninCA1 = [
            0.1,
            0.08,
            0.06,
            0.04,
            0.02,
            0.3,
            0.24,
            0.18,
            0.12,
            0.06,
            0.5,
            0.4,
            0.3,
            0.2,
            0.1,
            1.0,
            0.8,
            0.6,
            0.4,
            0.2,
            2.0,
            1.6,
            1.2,
            0.8,
            0.4,
            3.0,
            2.4,
            1.8,
            1.2,
            0.6,
            4.0,
            3.2,
            2.4,
            1.6,
            0.8,
            6.0,
            4.8,
            3.6,
            2.4,
            1.2,
        ]
        ninCA2 = [
            0.0,
            0.006666667,
            0.013333333,
            0.02,
            0.026666667,
            0.0,
            0.02,
            0.04,
            0.06,
            0.08,
            0.0,
            0.033333333,
            0.066666667,
            0.1,
            0.133333333,
            0.0,
            0.066666667,
            0.133333333,
            0.2,
            0.266666667,
            0.0,
            0.133333333,
            0.266666667,
            0.4,
            0.533333333,
            0.0,
            0.2,
            0.4,
            0.6,
            0.8,
            0.0,
            0.266666667,
            0.533333333,
            0.8,
            1.066666667,
            0.0,
            0.4,
            0.8,
            1.2,
            1.6,
        ]
        return ninCA1, ninCA2, gamma

    @pytest.mark.unit
    def test_parameters(self, model):
        m = model
        m.params = GenericParameterBlock(**configuration)
        assert value(m.params.Liq.tau["H2O", "Na+, Cl-"]) == 7.951
        assert value(m.params.Liq.tau["Na+, Cl-", "H2O"]) == -3.984
        assert value(m.params.Liq.tau["H2O", "Na+, SO4-"]) == 7.578
        assert value(model.params.Liq.tau["Na+, SO4-", "H2O"]) == -3.532

    @pytest.fixture(scope="function")
    def set_model_inputs(self, model, conc_elect1=1, conc_elect2=1):
        m = model
        # Add units to molal concentration from experimental data
        molal_conc_elect1 = conc_elect1 * (pyunits.mol / pyunits.kg)
        molal_conc_elect2 = conc_elect2 * (pyunits.mol / pyunits.kg)

        m.mass_solvent = pyo.Param(initialize=1, units=pyunits.kg / pyunits.s)
        m.solvent_mol = pyo.Param(
            initialize=55.50,
            units=pyunits.mol / pyunits.s,
        )
        m.electrolyte_mol = pyo.Var(
            m.fs.app_species_set,
            units=pyunits.mol / pyunits.s,
        )

        @m.Constraint(m.fs.app_species_set)
        def rule_electrolyte_mol(b, ap):
            return m.electrolyte_mol[ap] == molal_conc_elect1 * m.mass_solvent

        m.total_mol = pyo.Var(
            m.fs.app_species_set,
            units=pyunits.mol / pyunits.s,
        )

        @m.Constraint(m.fs.app_species_set)
        def rule_total_mol(b, ap):
            return m.total_mol[ap] == m.solvent_mol + (2 * m.electrolyte_mol[ap])

        m.fs.state[0].temperature.fix(298.15)
        m.fs.state[0].pressure.fix(101325)
        m.fs.state[0].flow_mol_phase_comp["Liq", cation_1].set_value(
            1 * molal_conc_elect1 * m.mass_solvent
            + 2 * molal_conc_elect2 * m.mass_solvent
        )
        m.fs.state[0].flow_mol_phase_comp["Liq", anion_1].set_value(
            molal_conc_elect1 * m.mass_solvent
        )
        m.fs.state[0].flow_mol_phase_comp["Liq", anion_2].set_value(
            molal_conc_elect2 * m.mass_solvent
        )
        m.fs.state[0].flow_mol_phase_comp["Liq", solvent].set_value(m.solvent_mol)
        return m

    @pytest.fixture(scope="function")
    def Retrieve_All_Properties(self, model, ExpData):
        optarg = {
            "max_iter": 300,
            "tol": 1e-8,
            "linear_solver": "ma27",
        }
        solver = get_solver("ipopt", optarg)
        ninCA1, ninCA2, gamma = ExpData

        IX = [0 for i in range(40)]
        gamma1Calc = [0 for i in range(40)]
        gamma2Calc = [0 for i in range(40)]
        aravg = [0 for i in range(40)]
        b = [0 for i in range(40)]
        bw = [0 for i in range(40)]
        AX = [0 for i in range(40)]
        Vt = [0 for i in range(40)]
        kappa = [0 for i in range(40)]
        PDH = np.zeros((40, 4))
        SR = np.zeros((40, 4))
        SR_inf = np.zeros((40, 4))
        he = np.zeros((40, 2))
        tau_maca = np.zeros((40, 2))
        tau_mcac = np.zeros((40, 2))
        log_mean_primitive = np.zeros((40, 2))
        n = np.zeros((40, 4))
        X = np.zeros((40, 4))
        Vo = np.zeros((40, 4))
        Vq = np.zeros((40, 4))
        Xp = np.zeros((40, 4))
        Y = np.zeros((40, 4))

        for i in range(0, len(ninCA1)):
            m = model
            # Add units to molal concentration from experimental data
            molal_conc_elect1 = ninCA1[i] * (pyunits.mol / pyunits.kg)
            molal_conc_elect2 = ninCA2[i] * (pyunits.mol / pyunits.kg)

            m.mass_solvent = pyo.Param(initialize=1, units=pyunits.kg / pyunits.s)
            m.solvent_mol = pyo.Param(
                initialize=55.50,
                units=pyunits.mol / pyunits.s,
            )
            m.electrolyte_mol = pyo.Var(
                m.fs.app_species_set,
                units=pyunits.mol / pyunits.s,
            )

            @m.Constraint(m.fs.app_species_set)
            def rule_electrolyte_mol(b, ap):
                return m.electrolyte_mol[ap] == molal_conc_elect1 * m.mass_solvent

            m.total_mol = pyo.Var(
                m.fs.app_species_set,
                units=pyunits.mol / pyunits.s,
            )

            @m.Constraint(m.fs.app_species_set)
            def rule_total_mol(b, ap):
                return m.total_mol[ap] == m.solvent_mol + (2 * m.electrolyte_mol[ap])

            m.fs.state[0].temperature.fix(298.15)
            m.fs.state[0].pressure.fix(101325)
            # 1 and 2 represent the stoichiometric coefficient or number of
            # cations
            m.fs.state[0].flow_mol_phase_comp["Liq", cation_1].set_value(
                1 * molal_conc_elect1 * m.mass_solvent
                + 2 * molal_conc_elect2 * m.mass_solvent
            )
            m.fs.state[0].flow_mol_phase_comp["Liq", anion_1].set_value(
                molal_conc_elect1 * m.mass_solvent
            )
            m.fs.state[0].flow_mol_phase_comp["Liq", anion_2].set_value(
                molal_conc_elect2 * m.mass_solvent
            )
            m.fs.state[0].flow_mol_phase_comp["Liq", solvent].set_value(m.solvent_mol)
            results = solver.solve(m, tee=True)
            gamma1Calc[i] = exp(value(m.fs.mean_molal_gamma[elect_1]))
            gamma2Calc[i] = exp(value(m.fs.mean_molal_gamma[elect_2]))
            IX[i] = value(m.fs.state[0].Liq_ionic_strength)
            Vt[i] = value(m.fs.state[0].Liq_vol_mol_solvent)
            b[i] = value(m.fs.state[0].b_term)
            aravg[i] = value(m.fs.state[0].Liq_ar_avg)
            bw[i] = b[i] * aravg[i]
            AX[i] = value(m.fs.state[0].Liq_A_DH)
            kappa[i] = value(m.fs.state[0].kappa)
            tau_maca[i, :] = [
                value(m.fs.state[0].tau_ij_ij["H2O", "Cl-", "Na+", "Cl-"]),
                value(m.fs.state[0].tau_ij_ij["H2O", "SO4-", "Na+", "SO4-"]),
            ]
            tau_mcac[i, :] = [
                value(m.fs.state[0].tau_ij_ij["H2O", "Na+", "Cl-", "Na+"]),
                value(m.fs.state[0].tau_ij_ij["H2O", "Na+", "SO4-", "Na+"]),
            ]

            k = 0
            for j in m.fs.species_set:
                PDH[i, k] = value(m.fs.state[0].Liq_log_gamma_pdh[j])
                SR[i, k] = value(m.fs.state[0].Liq_log_gamma_lc[j])
                SR_inf[i, k] = value(m.fs.state[0].Liq_log_gamma_inf[j])
                n[i, k] = value(m.fs.state[0].Liq_n[j])
                X[i, k] = value(m.fs.state[0].Liq_X[j])
                k = k + 1

            k = 0
            for j in m.fs.ion_set:
                Vo[i, k] = value(m.fs.state[0].Liq_Vo[j])
                Vq[i, k] = value(m.fs.state[0].Liq_Vq[j])
                Xp[i, k] = value(m.fs.state[0].Liq_Xp[j])
                Y[i, k] = value(m.fs.state[0].Liq_Y[j])
                k = k + 1

        return Prop(
            gamma1Calc,
            gamma2Calc,
            PDH,
            SR,
            SR_inf,
            IX,
            tau_maca,
            tau_mcac,
            n,
            X,
            Y,
            Xp,
            Vt,
            Vo,
            Vq,
            b,
            aravg,
            bw,
            AX,
            kappa,
        )

    @pytest.mark.unit
    def test_Log_Mean_Gamma(self, Retrieve_All_Properties):
        Properties = Retrieve_All_Properties

        # ref [5]
        gamma1ref = [
            0.7777209890681371,
            0.7757400402048191,
            0.7733986399701855,
            0.7704989221341023,
            0.766635231981608,
            0.7072234352767693,
            0.7021308519718867,
            0.6962088247038548,
            0.6890191610330922,
            0.6796748632116283,
            0.6790847469671388,
            0.6711358358564812,
            0.6619883949123276,
            0.6510238248722816,
            0.6370015520615392,
            0.6551854086960756,
            0.6402846485099757,
            0.6234635210542319,
            0.6037652265368347,
            0.5793045161175804,
            0.6692027649228939,
            0.6396258129627914,
            0.607192850488807,
            0.5705155366853356,
            0.5269296578228733,
            0.7172055886852967,
            0.6708565800438754,
            0.6212496015681783,
            0.566758515272512,
            0.5043324690170674,
            0.787092401063391,
            0.7211421589399183,
            0.6520694486819765,
            0.5781431878414264,
            0.4961885576494383,
            0.9835972249036236,
            0.8673692275095422,
            0.750244255430322,
            0.6305386446221195,
            0.5053631731313195,
        ]

        # ref [5]
        gamma2ref = [
            0.7789684994052755,
            0.7347222036888417,
            0.6934873189099119,
            0.6547907562199693,
            0.61799296860761,
            0.710533557309058,
            0.6471666155353714,
            0.5909584095541306,
            0.5405686267229206,
            0.4945238656192273,
            0.684369958292553,
            0.6085746236895396,
            0.5436133540344625,
            0.4872195586388261,
            0.4371606277467339,
            0.6655896698074943,
            0.5647447550663234,
            0.48369448713545,
            0.41746047399762093,
            0.3618543283575285,
            0.6919870331469314,
            0.5441371448375604,
            0.4359375352287419,
            0.3549462624864621,
            0.2923133677889821,
            0.7569089282760384,
            0.5566966005935452,
            0.42074391999403143,
            0.3257801723413064,
            0.2569352618870726,
            0.8502268529210172,
            0.5878385922968342,
            0.42099125290485867,
            0.3112264352985737,
            0.23593956850148226,
            1.1229297162618446,
            0.6927427856634997,
            0.44930478344970914,
            0.3050913873273755,
            0.21517281647445483,
        ]
        assert np.allclose(Properties.gamma1Calc, gamma1ref, rtol=1e-9, atol=1e-9)
        assert np.allclose(Properties.gamma2Calc, gamma2ref, rtol=1e-9, atol=1e-9)

    @pytest.mark.LR
    def test_PDH(self, Retrieve_All_Properties):
        Properties = Retrieve_All_Properties
        # ref [5]
        PDHref = [
            [-2.56204364e-01, -2.55759637e-01, -1.02402063e00, 2.52941789e-04],
            [-2.56721802e-01, -2.56275554e-01, -1.02608748e00, 2.53783167e-04],
            [-2.57422429e-01, -2.56974162e-01, -1.02888619e00, 2.54907424e-04],
            [-2.58430951e-01, -2.57979827e-01, -1.03291498e00, 2.56508044e-04],
            [-2.60018460e-01, -2.59562899e-01, -1.03925688e00, 2.59006978e-04],
            [-3.62050436e-01, -3.60380424e-01, -1.44523263e00, 9.51549714e-04],
            [-3.63119294e-01, -3.61440093e-01, -1.44948977e00, 9.56527018e-04],
            [-3.64557468e-01, -3.62866340e-01, -1.45521929e00, 9.63059770e-04],
            [-3.66618516e-01, -3.64910833e-01, -1.46343202e00, 9.72223285e-04],
            [-3.69857008e-01, -3.68123984e-01, -1.47633882e00, 9.86380983e-04],
            [-4.14788203e-01, -4.11832772e-01, -1.65393842e00, 1.68685187e-03],
            [-4.16224936e-01, -4.13249097e-01, -1.65964373e00, 1.69776342e-03],
            [-4.18149024e-01, -4.15147097e-01, -1.66728832e00, 1.71189937e-03],
            [-4.20896481e-01, -4.17858849e-01, -1.67820925e00, 1.73150064e-03],
            [-4.25204453e-01, -4.22112804e-01, -1.69533962e00, 1.76151216e-03],
            [-4.85938248e-01, -4.79846371e-01, -1.93320443e00, 3.49058426e-03],
            [-4.88002491e-01, -4.81852295e-01, -1.94133859e00, 3.52121743e-03],
            [-4.90741494e-01, -4.84518676e-01, -1.95214633e00, 3.55994816e-03],
            [-4.94623034e-01, -4.88303345e-01, -1.96748128e00, 3.61243974e-03],
            [-5.00677542e-01, -4.94214672e-01, -1.99142638e00, 3.69125795e-03],
            [-5.52575342e-01, -5.40811740e-01, -2.19064965e00, 6.78286904e-03],
            [-5.55437850e-01, -5.43512522e-01, -2.20175441e00, 6.86711024e-03],
            [-5.59183701e-01, -5.47063053e-01, -2.21633285e00, 6.97022116e-03],
            [-5.64427618e-01, -5.52054915e-01, -2.23680403e00, 7.10539829e-03],
            [-5.72529048e-01, -5.59795943e-01, -2.26851700e00, 7.30208476e-03],
            [-5.87762412e-01, -5.71053512e-01, -2.32406558e00, 9.67812736e-03],
            [-5.91190506e-01, -5.74191814e-01, -2.33716736e00, 9.82993626e-03],
            [-5.95629262e-01, -5.78286692e-01, -2.35421560e00, 1.00118851e-02],
            [-6.01781974e-01, -5.84004881e-01, -2.37796185e00, 1.02448300e-02],
            [-6.11207175e-01, -5.92823185e-01, -2.41450269e00, 1.05754458e-02],
            [-6.10453056e-01, -5.89385753e-01, -2.40885958e00, 1.22401290e-02],
            [-6.14335049e-01, -5.92832994e-01, -2.42349371e00, 1.24702485e-02],
            [-6.19319724e-01, -5.97307401e-01, -2.44240442e00, 1.27421895e-02],
            [-6.26172722e-01, -6.03524818e-01, -2.46857187e00, 1.30843523e-02],
            [-6.36592910e-01, -6.13072682e-01, -2.50860788e00, 1.35604073e-02],
            [-6.38567235e-01, -6.10153712e-01, -2.51236009e00, 1.65513127e-02],
            [-6.43163669e-01, -6.13994074e-01, -2.52928602e00, 1.69605606e-02],
            [-6.48995702e-01, -6.18948665e-01, -2.55094143e00, 1.74366795e-02],
            [-6.56914392e-01, -6.25793000e-01, -2.58060997e00, 1.80226836e-02],
            [-6.68810537e-01, -6.36249022e-01, -2.62558746e00, 1.88150374e-02],
        ]
        assert np.allclose(Properties.PDH, PDHref, rtol=1e-6, atol=1e-6)

    @pytest.mark.SR
    def test_SR(self, Retrieve_All_Properties):
        Properties = Retrieve_All_Properties
        # ref [5]
        SRref = [
            [9.48654053e-04, 9.48654053e-04, -4.47929587e-02, -1.81422722e-06],
            [-2.19125748e-03, 8.51415639e-04, -4.03433449e-02, 3.56348883e-06],
            [-5.19852068e-03, 6.18069606e-04, -3.61625461e-02, 7.99450132e-06],
            [-8.03939180e-03, 2.14901315e-04, -3.23179104e-02, 1.14778451e-05],
            [-1.06679543e-02, -4.04026940e-04, -2.89010713e-02, 1.40125375e-05],
            [3.81119226e-03, 3.81119226e-03, -1.31107660e-01, -2.33681895e-05],
            [-5.58258999e-03, 3.47224563e-03, -1.18116426e-01, 2.55253308e-05],
            [-1.46007609e-02, 2.72601808e-03, -1.05906648e-01, 6.59869982e-05],
            [-2.31423396e-02, 1.47153648e-03, -9.46800089e-02, 9.79897970e-05],
            [-3.10699040e-02, -4.28750230e-04, -8.47109159e-02, 1.21506538e-04],
            [7.86336665e-03, 7.86336665e-03, -2.13292953e-01, -8.33556083e-05],
            [-7.73888812e-03, 7.23150927e-03, -1.92201039e-01, 5.36475983e-05],
            [-2.27521528e-02, 5.92193304e-03, -1.72373291e-01, 1.67487860e-04],
            [-3.70086809e-02, 3.76661168e-03, -1.54145127e-01, 2.58040773e-04],
            [-5.02801140e-02, 5.36704587e-04, -1.37972949e-01, 3.25181025e-04],
            [2.25033430e-02, 2.25033430e-02, -4.02365324e-01, -4.99990178e-04],
            [-8.34903115e-03, 2.09958865e-02, -3.62698086e-01, 5.70155610e-05],
            [-3.82057805e-02, 1.81338447e-02, -3.25393786e-01, 5.24140660e-04],
            [-6.67353334e-02, 1.35822722e-02, -2.91118843e-01, 9.00422367e-04],
            [-9.34855125e-02, 6.88530459e-03, -2.60780529e-01, 1.18488451e-03],
            [6.64223311e-02, 6.64223311e-02, -7.21792749e-01, -3.07058260e-03],
            [6.65968772e-03, 6.30503044e-02, -6.50496742e-01, -8.10712949e-04],
            [-5.18044841e-02, 5.69479092e-02, -5.83438257e-01, 1.11373198e-03],
            [-1.08331608e-01, 4.74479898e-02, -5.21927987e-01, 2.69568816e-03],
            [-1.62044673e-01, 3.36440342e-02, -4.67754065e-01, 3.92788977e-03],
            [1.22763937e-01, 1.22763937e-01, -9.80952931e-01, -8.69219836e-03],
            [3.67213500e-02, 1.18037358e-01, -8.82960959e-01, -3.62364062e-03],
            [-4.83173737e-02, 1.09163416e-01, -7.90832132e-01, 7.46442904e-04],
            [-1.31445785e-01, 9.51427477e-02, -7.06495254e-01, 4.39642781e-03],
            [-2.11404944e-01, 7.46198603e-02, -6.32589299e-01, 7.30376751e-03],
            [1.85473571e-01, 1.85473571e-01, -1.19600363e00, -1.77753517e-02],
            [7.60429830e-02, 1.80347576e-01, -1.07427155e00, -8.89866697e-03],
            [-3.31665795e-02, 1.69623686e-01, -9.59902304e-01, -1.16524712e-03],
            [-1.41029117e-01, 1.51968983e-01, -8.55416662e-01, 5.37869108e-03],
            [-2.45957218e-01, 1.25578516e-01, -7.64274861e-01, 1.06843472e-02],
            [3.15144419e-01, 3.15144419e-01, -1.53703117e00, -4.63933976e-02],
            [1.67533789e-01, 3.12702719e-01, -1.37215438e00, -2.74394413e-02],
            [1.75230709e-02, 3.01737270e-01, -1.21746319e00, -1.06474682e-02],
            [-1.33481839e-01, 2.80244835e-01, -1.07658819e00, 3.85678550e-03],
            [-2.83416505e-01, 2.45518583e-01, -9.54549131e-01, 1.59374945e-02],
        ]
        assert np.allclose(Properties.SR, SRref, rtol=1e-6, atol=1e-6)

    @pytest.mark.SR
    def test_SR_inf(self, Retrieve_All_Properties):
        Properties = Retrieve_All_Properties
        # ref [5]
        SR_infref = [
            [-0.88722743, -0.88722743, 3.49919861, 0.0],
            [-0.697715, -1.07627478, 3.10808602, 0.0],
            [-0.48020355, -1.29209047, 2.6614017, 0.0],
            [-0.22800998, -1.54072996, 2.14652303, 0.0],
            [0.0678528, -1.83019768, 1.54674564, 0.0],
            [-0.88722743, -0.88722743, 3.49919861, 0.0],
            [-0.697715, -1.07627477, 3.10808603, 0.0],
            [-0.48020355, -1.29209048, 2.66140169, 0.0],
            [-0.22800998, -1.54072996, 2.14652303, 0.0],
            [0.0678528, -1.83019768, 1.54674565, 0.0],
            [-0.88722743, -0.88722743, 3.49919861, 0.0],
            [-0.69771501, -1.07627477, 3.10808604, 0.0],
            [-0.48020355, -1.29209048, 2.66140168, 0.0],
            [-0.22800998, -1.54072996, 2.14652303, 0.0],
            [0.0678528, -1.83019768, 1.54674565, 0.0],
            [-0.88722743, -0.88722743, 3.49919861, 0.0],
            [-0.697715, -1.07627477, 3.10808603, 0.0],
            [-0.48020355, -1.29209048, 2.66140169, 0.0],
            [-0.22800998, -1.54072996, 2.14652303, 0.0],
            [0.0678528, -1.83019768, 1.54674565, 0.0],
            [-0.88722743, -0.88722743, 3.49919861, 0.0],
            [-0.697715, -1.07627477, 3.10808603, 0.0],
            [-0.48020355, -1.29209048, 2.66140169, 0.0],
            [-0.22800998, -1.54072996, 2.14652303, 0.0],
            [0.0678528, -1.83019768, 1.54674565, 0.0],
            [-0.88722743, -0.88722743, 3.49919861, 0.0],
            [-0.697715, -1.07627477, 3.10808603, 0.0],
            [-0.48020355, -1.29209048, 2.66140169, 0.0],
            [-0.22800998, -1.54072996, 2.14652303, 0.0],
            [0.0678528, -1.83019768, 1.54674565, 0.0],
            [-0.88722743, -0.88722743, 3.49919861, 0.0],
            [-0.697715, -1.07627477, 3.10808603, 0.0],
            [-0.48020355, -1.29209048, 2.66140169, 0.0],
            [-0.22800998, -1.54072996, 2.14652303, 0.0],
            [0.0678528, -1.83019768, 1.54674565, 0.0],
            [-0.88722743, -0.88722743, 3.49919861, 0.0],
            [-0.697715, -1.07627477, 3.10808603, 0.0],
            [-0.48020355, -1.29209048, 2.66140169, 0.0],
            [-0.22800998, -1.54072996, 2.14652303, 0.0],
            [0.0678528, -1.83019768, 1.54674565, 0.0],
        ]
        assert np.allclose(Properties.SR_inf, SR_infref, rtol=1e-6, atol=1e-6)

    @pytest.mark.SR
    def test_tau_maca(self, Retrieve_All_Properties):
        Properties = Retrieve_All_Properties
        # ref [5]
        tau_macaref = [
            [7.951, 7.578],
            [7.951, 7.578],
            [7.951, 7.578],
            [7.951, 7.578],
            [7.951, 7.578],
            [7.951, 7.578],
            [7.951, 7.578],
            [7.951, 7.578],
            [7.951, 7.578],
            [7.951, 7.578],
            [7.951, 7.578],
            [7.951, 7.578],
            [7.951, 7.578],
            [7.951, 7.578],
            [7.951, 7.578],
            [7.951, 7.578],
            [7.951, 7.578],
            [7.951, 7.578],
            [7.951, 7.578],
            [7.951, 7.578],
            [7.951, 7.578],
            [7.951, 7.578],
            [7.951, 7.578],
            [7.951, 7.578],
            [7.951, 7.578],
            [7.951, 7.578],
            [7.951, 7.578],
            [7.951, 7.578],
            [7.951, 7.578],
            [7.951, 7.578],
            [7.951, 7.578],
            [7.951, 7.578],
            [7.951, 7.578],
            [7.951, 7.578],
            [7.951, 7.578],
            [7.951, 7.578],
            [7.951, 7.578],
            [7.951, 7.578],
            [7.951, 7.578],
            [7.951, 7.578],
        ]
        assert np.allclose(Properties.tau_maca, tau_macaref, rtol=1e-6, atol=1e-6)

    @pytest.mark.SR
    def test_tau_mcac(self, Retrieve_All_Properties):
        Properties = Retrieve_All_Properties
        # ref [5]
        tau_mcacref = [
            [7.951, 7.126],
            [8.01312313, 7.18812313],
            [8.08577611, 7.26077611],
            [8.17189414, 7.34689414],
            [8.27562001, 7.45062001],
            [7.951, 7.126],
            [8.01312313, 7.18812313],
            [8.08577612, 7.26077612],
            [8.17189414, 7.34689414],
            [8.27562001, 7.45062001],
            [7.951, 7.126],
            [8.01312313, 7.18812313],
            [8.08577612, 7.26077612],
            [8.17189414, 7.34689414],
            [8.27562001, 7.45062001],
            [7.951, 7.126],
            [8.01312313, 7.18812313],
            [8.08577612, 7.26077612],
            [8.17189414, 7.34689414],
            [8.27562001, 7.45062001],
            [7.951, 7.126],
            [8.01312313, 7.18812313],
            [8.08577612, 7.26077612],
            [8.17189414, 7.34689414],
            [8.27562001, 7.45062001],
            [7.951, 7.126],
            [8.01312313, 7.18812313],
            [8.08577612, 7.26077612],
            [8.17189414, 7.34689414],
            [8.27562001, 7.45062001],
            [7.951, 7.126],
            [8.01312313, 7.18812313],
            [8.08577612, 7.26077612],
            [8.17189414, 7.34689414],
            [8.27562001, 7.45062001],
            [7.951, 7.126],
            [8.01312313, 7.18812313],
            [8.08577612, 7.26077612],
            [8.17189414, 7.34689414],
            [8.27562001, 7.45062001],
        ]
        assert np.allclose(Properties.tau_mcac, tau_mcacref, rtol=1e-6, atol=1e-6)

    @pytest.mark.SR
    def test_n(self, Retrieve_All_Properties):
        Properties = Retrieve_All_Properties
        # ref [5]
        nref = [
            [1.00000000e-01, 1.00000000e-01, 0.00000000e00, 5.52990600e01],
            [9.33333340e-02, 8.00000000e-02, 6.66666700e-03, 5.53211813e01],
            [8.66666660e-02, 6.00000000e-02, 1.33333330e-02, 5.53433027e01],
            [8.00000000e-02, 4.00000000e-02, 2.00000000e-02, 5.53654240e01],
            [7.33333340e-02, 2.00000000e-02, 2.66666670e-02, 5.53875453e01],
            [3.00000000e-01, 3.00000000e-01, 0.00000000e00, 5.48971800e01],
            [2.80000000e-01, 2.40000000e-01, 2.00000000e-02, 5.49635440e01],
            [2.60000000e-01, 1.80000000e-01, 4.00000000e-02, 5.50299080e01],
            [2.40000000e-01, 1.20000000e-01, 6.00000000e-02, 5.50962720e01],
            [2.20000000e-01, 6.00000000e-02, 8.00000000e-02, 5.51626360e01],
            [5.00000000e-01, 5.00000000e-01, 0.00000000e00, 5.44953000e01],
            [4.66666666e-01, 4.00000000e-01, 3.33333330e-02, 5.46059067e01],
            [4.33333334e-01, 3.00000000e-01, 6.66666670e-02, 5.47165133e01],
            [4.00000000e-01, 2.00000000e-01, 1.00000000e-01, 5.48271200e01],
            [3.66666666e-01, 1.00000000e-01, 1.33333333e-01, 5.49377267e01],
            [1.00000000e00, 1.00000000e00, 0.00000000e00, 5.34906000e01],
            [9.33333334e-01, 8.00000000e-01, 6.66666670e-02, 5.37118133e01],
            [8.66666666e-01, 6.00000000e-01, 1.33333333e-01, 5.39330267e01],
            [8.00000000e-01, 4.00000000e-01, 2.00000000e-01, 5.41542400e01],
            [7.33333334e-01, 2.00000000e-01, 2.66666667e-01, 5.43754533e01],
            [2.00000000e00, 2.00000000e00, 0.00000000e00, 5.14812000e01],
            [1.86666667e00, 1.60000000e00, 1.33333333e-01, 5.19236267e01],
            [1.73333333e00, 1.20000000e00, 2.66666667e-01, 5.23660533e01],
            [1.60000000e00, 8.00000000e-01, 4.00000000e-01, 5.28084800e01],
            [1.46666667e00, 4.00000000e-01, 5.33333333e-01, 5.32509067e01],
            [3.00000000e00, 3.00000000e00, 0.00000000e00, 4.94718000e01],
            [2.80000000e00, 2.40000000e00, 2.00000000e-01, 5.01354400e01],
            [2.60000000e00, 1.80000000e00, 4.00000000e-01, 5.07990800e01],
            [2.40000000e00, 1.20000000e00, 6.00000000e-01, 5.14627200e01],
            [2.20000000e00, 6.00000000e-01, 8.00000000e-01, 5.21263600e01],
            [4.00000000e00, 4.00000000e00, 0.00000000e00, 4.74624000e01],
            [3.73333333e00, 3.20000000e00, 2.66666667e-01, 4.83472533e01],
            [3.46666667e00, 2.40000000e00, 5.33333333e-01, 4.92321067e01],
            [3.20000000e00, 1.60000000e00, 8.00000000e-01, 5.01169600e01],
            [2.93333333e00, 8.00000000e-01, 1.06666667e00, 5.10018133e01],
            [6.00000000e00, 6.00000000e00, 0.00000000e00, 4.34436000e01],
            [5.60000000e00, 4.80000000e00, 4.00000000e-01, 4.47708800e01],
            [5.20000000e00, 3.60000000e00, 8.00000000e-01, 4.60981600e01],
            [4.80000000e00, 2.40000000e00, 1.20000000e00, 4.74254400e01],
            [4.40000000e00, 1.20000000e00, 1.60000000e00, 4.87527200e01],
        ]
        assert np.allclose(Properties.n, nref, rtol=1e-6, atol=1e-6)

    @pytest.mark.SR
    def test_X(self, Retrieve_All_Properties):
        Properties = Retrieve_All_Properties
        # ref [5]
        Xref = [
            [1.80183232e-03, 1.80183232e-03, 0.00000000e00, 9.96396335e-01],
            [1.68164590e-03, 1.44141076e-03, 2.40235139e-04, 9.96756826e-01],
            [1.56146863e-03, 1.08101675e-03, 4.80451878e-04, 9.97117289e-01],
            [1.44130058e-03, 7.20650292e-04, 7.20650292e-04, 9.97477724e-01],
            [1.32114172e-03, 3.60311375e-04, 9.60830346e-04, 9.97838132e-01],
            [5.40568007e-03, 5.40568007e-03, 0.00000000e00, 9.89188640e-01],
            [5.04472291e-03, 4.32404821e-03, 7.20674701e-04, 9.90270892e-01],
            [4.68384851e-03, 3.24266435e-03, 1.44118416e-03, 9.91352895e-01],
            [4.32305685e-03, 2.16152843e-03, 2.16152843e-03, 9.92434651e-01],
            [3.96234790e-03, 1.08064034e-03, 2.88170756e-03, 9.93516158e-01],
            [9.00977200e-03, 9.00977200e-03, 0.00000000e00, 9.81980456e-01],
            [8.40751361e-03, 7.20644025e-03, 1.20107336e-03, 9.83785509e-01],
            [7.80548539e-03, 5.40379757e-03, 2.40168782e-03, 9.85589873e-01],
            [7.20368714e-03, 3.60184357e-03, 3.60184357e-03, 9.87393548e-01],
            [6.60211874e-03, 1.80057784e-03, 4.80154090e-03, 9.89196533e-01],
            [1.80210702e-02, 1.80210702e-02, 0.00000000e00, 9.63957860e-01],
            [1.68132381e-02, 1.44113469e-02, 2.40189117e-03, 9.67574469e-01],
            [1.56063287e-02, 1.08043814e-02, 4.80194727e-03, 9.71188316e-01],
            [1.44003410e-02, 7.20017050e-03, 7.20017050e-03, 9.74799403e-01],
            [1.31952740e-02, 3.59871109e-03, 9.59656291e-03, 9.78407733e-01],
            [3.60482470e-02, 3.60482470e-02, 0.00000000e00, 9.27903506e-01],
            [3.36193217e-02, 2.88165615e-02, 4.80276023e-03, 9.35162737e-01],
            [3.11941056e-02, 2.15959192e-02, 9.59818634e-03, 9.42410882e-01],
            [2.87725901e-02, 1.43862950e-02, 1.43862950e-02, 9.49647967e-01],
            [2.63547668e-02, 7.18766367e-03, 1.91671031e-02, 9.56874018e-01],
            [5.40815333e-02, 5.40815333e-02, 0.00000000e00, 8.91836933e-01],
            [5.04182554e-02, 4.32156475e-02, 7.20260792e-03, 9.02764793e-01],
            [4.67633637e-02, 3.23746364e-02, 1.43887273e-02, 9.13667636e-01],
            [4.31168294e-02, 2.15584147e-02, 2.15584147e-02, 9.24545549e-01],
            [3.94786238e-02, 1.07668974e-02, 2.87117264e-02, 9.35398616e-01],
            [7.21209324e-02, 7.21209324e-02, 0.00000000e00, 8.55758135e-01],
            [6.72100439e-02, 5.76086090e-02, 9.60143485e-03, 8.70380630e-01],
            [6.23141361e-02, 4.31405558e-02, 1.91735803e-02, 8.84958518e-01],
            [5.74331406e-02, 2.87165703e-02, 2.87165703e-02, 8.99492004e-01],
            [5.25669895e-02, 1.43364517e-02, 3.82305378e-02, 9.13981290e-01],
            [1.08218081e-01, 1.08218081e-01, 0.00000000e00, 7.83563838e-01],
            [1.00772203e-01, 8.63761740e-02, 1.43960290e-02, 8.05653609e-01],
            [9.33603552e-02, 6.46340920e-02, 2.87262631e-02, 8.27642421e-01],
            [8.59823048e-02, 4.29911524e-02, 4.29911524e-02, 8.49530967e-01],
            [7.86378214e-02, 2.14466786e-02, 5.71911428e-02, 8.71319929e-01],
        ]
        assert np.allclose(Properties.X, Xref, rtol=1e-6, atol=1e-6)

    @pytest.mark.SR
    def test_Y(self, Retrieve_All_Properties):
        Properties = Retrieve_All_Properties
        # ref [5]
        Yref = [
            [1.0, 1.0, 0.0, 0.0],
            [1.0, 0.85714285, 0.14285715, 0.0],
            [1.0, 0.6923077, 0.3076923, 0.0],
            [1.0, 0.5, 0.5, 0.0],
            [1.0, 0.27272727, 0.72727273, 0.0],
            [1.0, 1.0, 0.0, 0.0],
            [1.0, 0.85714286, 0.14285714, 0.0],
            [1.0, 0.69230769, 0.30769231, 0.0],
            [1.0, 0.5, 0.5, 0.0],
            [1.0, 0.27272727, 0.72727273, 0.0],
            [1.0, 1.0, 0.0, 0.0],
            [1.0, 0.85714286, 0.14285714, 0.0],
            [1.0, 0.69230769, 0.30769231, 0.0],
            [1.0, 0.5, 0.5, 0.0],
            [1.0, 0.27272727, 0.72727273, 0.0],
            [1.0, 1.0, 0.0, 0.0],
            [1.0, 0.85714286, 0.14285714, 0.0],
            [1.0, 0.69230769, 0.30769231, 0.0],
            [1.0, 0.5, 0.5, 0.0],
            [1.0, 0.27272727, 0.72727273, 0.0],
            [1.0, 1.0, 0.0, 0.0],
            [1.0, 0.85714286, 0.14285714, 0.0],
            [1.0, 0.69230769, 0.30769231, 0.0],
            [1.0, 0.5, 0.5, 0.0],
            [1.0, 0.27272727, 0.72727273, 0.0],
            [1.0, 1.0, 0.0, 0.0],
            [1.0, 0.85714286, 0.14285714, 0.0],
            [1.0, 0.69230769, 0.30769231, 0.0],
            [1.0, 0.5, 0.5, 0.0],
            [1.0, 0.27272727, 0.72727273, 0.0],
            [1.0, 1.0, 0.0, 0.0],
            [1.0, 0.85714286, 0.14285714, 0.0],
            [1.0, 0.69230769, 0.30769231, 0.0],
            [1.0, 0.5, 0.5, 0.0],
            [1.0, 0.27272727, 0.72727273, 0.0],
            [1.0, 1.0, 0.0, 0.0],
            [1.0, 0.85714286, 0.14285714, 0.0],
            [1.0, 0.69230769, 0.30769231, 0.0],
            [1.0, 0.5, 0.5, 0.0],
            [1.0, 0.27272727, 0.72727273, 0.0],
        ]
        assert np.allclose(Properties.Y, Yref, rtol=1e-6, atol=1e-6)

    @pytest.mark.LR
    def test_IX(self, Retrieve_All_Properties):
        Properties = Retrieve_All_Properties
        # ref [5]
        IXref = [
            99.54943328164065,
            99.57479877613014,
            99.60016829290261,
            99.62554779928548,
            99.65093430364232,
            297.6423582521856,
            297.87133488991566,
            298.10051046157514,
            298.32988457674395,
            298.55945683952734,
            494.36936945533785,
            495.00708717254395,
            495.64577169050386,
            496.28541450971295,
            496.9260159587566,
            980.0346807386318,
            982.5979448458894,
            985.1697037849586,
            987.7499390705054,
            990.3386216020521,
            1923.8255485550408,
            1934.1106363488698,
            1944.4740794818124,
            1954.9159505774333,
            1965.4362914731596,
            2829.2285135813586,
            2852.2706353822323,
            2875.6033127298992,
            2899.2288864216225,
            2923.1495196583837,
            3695.0827493881375,
            3735.6223467208283,
            3776.8945680713746,
            3818.9117464990713,
            3861.685894149594,
            5307.244215571142,
            5395.311396947033,
            5486.002624834118,
            5579.414647981235,
            5675.646831655127,
        ]
        assert np.allclose(Properties.IX, IXref, rtol=1e-6, atol=1e-6)

    @pytest.mark.LR
    def test_Vt(self, Retrieve_All_Properties):
        Properties = Retrieve_All_Properties
        # ref [5]
        Vtref = [
            0.0010045260601040755,
            0.0010042701790924614,
            0.001004014357746079,
            0.0010037585961531566,
            0.0010035028943661888,
            0.0010079210558660366,
            0.0010071462569933795,
            0.001006371976805688,
            0.0010055982169725488,
            0.0010048249791707216,
            0.0010113895214642152,
            0.0010100865461461883,
            0.001008784962080167,
            0.0010074847766661785,
            0.0010061859973970423,
            0.0010203720538198922,
            0.0010177102509173678,
            0.0010150535437276078,
            0.0010124019860138108,
            0.0010097556322526521,
            0.0010395953008848296,
            0.0010340670080670842,
            0.0010285557529946527,
            0.0010230618863226577,
            0.001017585768450899,
            0.0010603597360902006,
            0.0010517936000831037,
            0.0010432593350826291,
            0.0010347579020236496,
            0.0010262903008637744,
            0.0010825197353597435,
            0.0010707720507430966,
            0.0010590711302387642,
            0.0010474188107821394,
            0.0010358170267188095,
            0.0011305302255351945,
            0.0011120766826165274,
            0.0010936925135323688,
            0.0010753816266677622,
            0.0010571482296142598,
        ]
        assert np.allclose(Properties.Vt, Vtref, rtol=1e-6, atol=1e-6)

    @pytest.mark.LR
    def test_aravg(self, Retrieve_All_Properties):
        Properties = Retrieve_All_Properties
        # ref [5]
        aravgref = [
            4.2879932127265835e-10,
            4.261154088315522e-10,
            4.2245552865259376e-10,
            4.1716903456464767e-10,
            4.0886168680926003e-10,
            4.2879932127265835e-10,
            4.2611540895542515e-10,
            4.224555285228344e-10,
            4.171690345646477e-10,
            4.088616869160688e-10,
            4.2879932127265835e-10,
            4.2611540898019973e-10,
            4.2245552849688244e-10,
            4.1716903456464777e-10,
            4.0886168693743053e-10,
            4.2879932127265835e-10,
            4.2611540894303787e-10,
            4.224555285358103e-10,
            4.1716903456464777e-10,
            4.0886168690538793e-10,
            4.2879932127265835e-10,
            4.261154089616187e-10,
            4.2245552851634647e-10,
            4.1716903456464777e-10,
            4.0886168692140923e-10,
            4.287993212726584e-10,
            4.261154089554251e-10,
            4.2245552852283446e-10,
            4.1716903456464767e-10,
            4.088616869160688e-10,
            4.2879932127265835e-10,
            4.2611540895232834e-10,
            4.224555285260784e-10,
            4.1716903456464777e-10,
            4.088616869133985e-10,
            4.287993212726584e-10,
            4.261154089554251e-10,
            4.2245552852283446e-10,
            4.1716903456464767e-10,
            4.088616869160688e-10,
        ]
        assert np.allclose(Properties.aravg, aravgref, rtol=1e-6, atol=1e-6)

    @pytest.mark.LR
    def test_b(self, Retrieve_All_Properties):
        Properties = Retrieve_All_Properties
        # ref [5]
        bref = [
            103925446.29701309,
            103925446.29701309,
            103925446.29701309,
            103925446.29701309,
            103925446.29701309,
            103925446.29701309,
            103925446.29701309,
            103925446.29701309,
            103925446.29701309,
            103925446.29701309,
            103925446.29701309,
            103925446.29701309,
            103925446.29701309,
            103925446.29701309,
            103925446.29701309,
            103925446.29701309,
            103925446.29701309,
            103925446.29701309,
            103925446.29701309,
            103925446.29701309,
            103925446.29701309,
            103925446.29701309,
            103925446.29701309,
            103925446.29701309,
            103925446.29701309,
            103925446.29701309,
            103925446.29701309,
            103925446.29701309,
            103925446.29701309,
            103925446.29701309,
            103925446.29701309,
            103925446.29701309,
            103925446.29701309,
            103925446.29701309,
            103925446.29701309,
            103925446.29701309,
            103925446.29701309,
            103925446.29701309,
            103925446.29701309,
            103925446.29701309,
        ]
        assert np.allclose(Properties.b, bref, rtol=1e-6, atol=1e-6)

    @pytest.mark.LR
    def test_AX(self, Retrieve_All_Properties):
        Properties = Retrieve_All_Properties
        # ref [5]
        AXref = [
            0.03708044495436347,
            0.03708044495436347,
            0.03708044495436347,
            0.03708044495436347,
            0.03708044495436347,
            0.03708044495436347,
            0.03708044495436347,
            0.03708044495436347,
            0.03708044495436347,
            0.03708044495436347,
            0.03708044495436347,
            0.03708044495436347,
            0.03708044495436347,
            0.03708044495436347,
            0.03708044495436347,
            0.03708044495436347,
            0.03708044495436347,
            0.03708044495436347,
            0.03708044495436347,
            0.03708044495436347,
            0.03708044495436347,
            0.03708044495436347,
            0.03708044495436347,
            0.03708044495436347,
            0.03708044495436347,
            0.03708044495436347,
            0.03708044495436347,
            0.03708044495436347,
            0.03708044495436347,
            0.03708044495436347,
            0.03708044495436347,
            0.03708044495436347,
            0.03708044495436347,
            0.03708044495436347,
            0.03708044495436347,
            0.03708044495436347,
            0.03708044495436347,
            0.03708044495436347,
            0.03708044495436347,
            0.03708044495436347,
        ]
        assert np.allclose(Properties.AX, AXref, rtol=1e-6, atol=1e-6)

    @pytest.mark.LR
    def test_kappa(self, Retrieve_All_Properties):
        Properties = Retrieve_All_Properties
        # ref [5]
        kapparef = [
            10369.105524045906,
            10370.426479515387,
            10371.747476174616,
            10373.068824668631,
            10374.390369137389,
            17929.54492036308,
            17936.44020512397,
            17943.338827297895,
            17950.24077129134,
            17957.146021368284,
            23107.21873764872,
            23122.117651876408,
            23137.02953747268,
            23151.95417112045,
            23166.891535841376,
            32534.38640812882,
            32576.905192778693,
            32619.50918954597,
            32662.197759353294,
            32704.970090523144,
            45583.19288512608,
            45704.878087753525,
            45827.16351795728,
            45950.04523714014,
            46073.518964626644,
            55278.4569252014,
            55503.10297603249,
            55729.65901902024,
            55958.12444528182,
            56188.496754274216,
            63173.360903482426,
            63518.96026704127,
            63868.88420367684,
            64223.16572865656,
            64581.83357172032,
            75710.55579981925,
            76336.13302078145,
            76975.0365418995,
            77627.61036621635,
            78294.19803302018,
        ]
        assert np.allclose(Properties.kappa, kapparef, rtol=1e-6, atol=1e-6)

    @pytest.mark.LR
    def test_Vo(self, Retrieve_All_Properties):
        Properties = Retrieve_All_Properties
        # ref [5]
        Voref = [
            [9.76199248e-06, 3.31570222e-05, 5.83947888e-05, 0.00000000e00],
            [9.76199248e-06, 3.31570222e-05, 5.83947888e-05, 0.00000000e00],
            [9.76199248e-06, 3.31570222e-05, 5.83947888e-05, 0.00000000e00],
            [9.76199248e-06, 3.31570222e-05, 5.83947888e-05, 0.00000000e00],
            [9.76199248e-06, 3.31570222e-05, 5.83947888e-05, 0.00000000e00],
            [9.76199248e-06, 3.31570222e-05, 5.83947888e-05, 0.00000000e00],
            [9.76199248e-06, 3.31570222e-05, 5.83947888e-05, 0.00000000e00],
            [9.76199248e-06, 3.31570222e-05, 5.83947888e-05, 0.00000000e00],
            [9.76199248e-06, 3.31570222e-05, 5.83947888e-05, 0.00000000e00],
            [9.76199248e-06, 3.31570222e-05, 5.83947888e-05, 0.00000000e00],
            [9.76199248e-06, 3.31570222e-05, 5.83947888e-05, 0.00000000e00],
            [9.76199248e-06, 3.31570222e-05, 5.83947888e-05, 0.00000000e00],
            [9.76199248e-06, 3.31570222e-05, 5.83947888e-05, 0.00000000e00],
            [9.76199248e-06, 3.31570222e-05, 5.83947888e-05, 0.00000000e00],
            [9.76199248e-06, 3.31570222e-05, 5.83947888e-05, 0.00000000e00],
            [9.76199248e-06, 3.31570222e-05, 5.83947888e-05, 0.00000000e00],
            [9.76199248e-06, 3.31570222e-05, 5.83947888e-05, 0.00000000e00],
            [9.76199248e-06, 3.31570222e-05, 5.83947888e-05, 0.00000000e00],
            [9.76199248e-06, 3.31570222e-05, 5.83947888e-05, 0.00000000e00],
            [9.76199248e-06, 3.31570222e-05, 5.83947888e-05, 0.00000000e00],
            [9.76199248e-06, 3.31570222e-05, 5.83947888e-05, 0.00000000e00],
            [9.76199248e-06, 3.31570222e-05, 5.83947888e-05, 0.00000000e00],
            [9.76199248e-06, 3.31570222e-05, 5.83947888e-05, 0.00000000e00],
            [9.76199248e-06, 3.31570222e-05, 5.83947888e-05, 0.00000000e00],
            [9.76199248e-06, 3.31570222e-05, 5.83947888e-05, 0.00000000e00],
            [9.76199248e-06, 3.31570222e-05, 5.83947888e-05, 0.00000000e00],
            [9.76199248e-06, 3.31570222e-05, 5.83947888e-05, 0.00000000e00],
            [9.76199248e-06, 3.31570222e-05, 5.83947888e-05, 0.00000000e00],
            [9.76199248e-06, 3.31570222e-05, 5.83947888e-05, 0.00000000e00],
            [9.76199248e-06, 3.31570222e-05, 5.83947888e-05, 0.00000000e00],
            [9.76199248e-06, 3.31570222e-05, 5.83947888e-05, 0.00000000e00],
            [9.76199248e-06, 3.31570222e-05, 5.83947888e-05, 0.00000000e00],
            [9.76199248e-06, 3.31570222e-05, 5.83947888e-05, 0.00000000e00],
            [9.76199248e-06, 3.31570222e-05, 5.83947888e-05, 0.00000000e00],
            [9.76199248e-06, 3.31570222e-05, 5.83947888e-05, 0.00000000e00],
            [9.76199248e-06, 3.31570222e-05, 5.83947888e-05, 0.00000000e00],
            [9.76199248e-06, 3.31570222e-05, 5.83947888e-05, 0.00000000e00],
            [9.76199248e-06, 3.31570222e-05, 5.83947888e-05, 0.00000000e00],
            [9.76199248e-06, 3.31570222e-05, 5.83947888e-05, 0.00000000e00],
            [9.76199248e-06, 3.31570222e-05, 5.83947888e-05, 0.00000000e00],
        ]
        assert np.allclose(Properties.Vo, Voref, rtol=1e-6, atol=1e-6)

    @pytest.mark.LR
    def test_Vq(self, Retrieve_All_Properties):
        Properties = Retrieve_All_Properties
        # ref [5]
        Vqref = [
            [-7.60e-06, 2.42e-05, 2.68e-05, 0.00e00],
            [-7.60e-06, 2.42e-05, 2.68e-05, 0.00e00],
            [-7.60e-06, 2.42e-05, 2.68e-05, 0.00e00],
            [-7.60e-06, 2.42e-05, 2.68e-05, 0.00e00],
            [-7.60e-06, 2.42e-05, 2.68e-05, 0.00e00],
            [-7.60e-06, 2.42e-05, 2.68e-05, 0.00e00],
            [-7.60e-06, 2.42e-05, 2.68e-05, 0.00e00],
            [-7.60e-06, 2.42e-05, 2.68e-05, 0.00e00],
            [-7.60e-06, 2.42e-05, 2.68e-05, 0.00e00],
            [-7.60e-06, 2.42e-05, 2.68e-05, 0.00e00],
            [-7.60e-06, 2.42e-05, 2.68e-05, 0.00e00],
            [-7.60e-06, 2.42e-05, 2.68e-05, 0.00e00],
            [-7.60e-06, 2.42e-05, 2.68e-05, 0.00e00],
            [-7.60e-06, 2.42e-05, 2.68e-05, 0.00e00],
            [-7.60e-06, 2.42e-05, 2.68e-05, 0.00e00],
            [-7.60e-06, 2.42e-05, 2.68e-05, 0.00e00],
            [-7.60e-06, 2.42e-05, 2.68e-05, 0.00e00],
            [-7.60e-06, 2.42e-05, 2.68e-05, 0.00e00],
            [-7.60e-06, 2.42e-05, 2.68e-05, 0.00e00],
            [-7.60e-06, 2.42e-05, 2.68e-05, 0.00e00],
            [-7.60e-06, 2.42e-05, 2.68e-05, 0.00e00],
            [-7.60e-06, 2.42e-05, 2.68e-05, 0.00e00],
            [-7.60e-06, 2.42e-05, 2.68e-05, 0.00e00],
            [-7.60e-06, 2.42e-05, 2.68e-05, 0.00e00],
            [-7.60e-06, 2.42e-05, 2.68e-05, 0.00e00],
            [-7.60e-06, 2.42e-05, 2.68e-05, 0.00e00],
            [-7.60e-06, 2.42e-05, 2.68e-05, 0.00e00],
            [-7.60e-06, 2.42e-05, 2.68e-05, 0.00e00],
            [-7.60e-06, 2.42e-05, 2.68e-05, 0.00e00],
            [-7.60e-06, 2.42e-05, 2.68e-05, 0.00e00],
            [-7.60e-06, 2.42e-05, 2.68e-05, 0.00e00],
            [-7.60e-06, 2.42e-05, 2.68e-05, 0.00e00],
            [-7.60e-06, 2.42e-05, 2.68e-05, 0.00e00],
            [-7.60e-06, 2.42e-05, 2.68e-05, 0.00e00],
            [-7.60e-06, 2.42e-05, 2.68e-05, 0.00e00],
            [-7.60e-06, 2.42e-05, 2.68e-05, 0.00e00],
            [-7.60e-06, 2.42e-05, 2.68e-05, 0.00e00],
            [-7.60e-06, 2.42e-05, 2.68e-05, 0.00e00],
            [-7.60e-06, 2.42e-05, 2.68e-05, 0.00e00],
            [-7.60e-06, 2.42e-05, 2.68e-05, 0.00e00],
        ]
        assert np.allclose(Properties.Vq, Vqref, rtol=1e-6, atol=1e-6)
