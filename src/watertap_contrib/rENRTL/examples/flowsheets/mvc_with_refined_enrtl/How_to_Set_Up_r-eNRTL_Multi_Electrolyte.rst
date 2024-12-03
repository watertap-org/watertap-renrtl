.. How to set up refined electrolyte NRTL (r-eNRTL) model for multi-electrolyte mixtures:

How to set up refined electrolyte NRTL model for multi-electrolyte solutions
=====================================

The multi-electrolyte case of the refined electrolyte NRTL (multi r-eNRTL) resembles the single-electrolyte configuration (single r-eNRTL) but includes several distinct configuration adjustments. To demonstrate, we will use an aqueous solution of :math:`NaCl-Na₂SO₄` as an example to guide the set up of the configuration file. 

The multi r-eNRTL method supports aqueous solutions with up to two electrolytes. In this method, the electrolytes dissociate in water as illustrated in the equations below:    

.. math::

    NaCl → Na⁺ + Cl⁻
    Na₂SO₄ → 2Na⁺ + SO₄²⁻

In the following sections, we explain key parameters and their calculations. A code example for setting up the multi r-eNRTL can be found in the `Code Example` section. A demonstration of the multi r-eNRTL model's application in a flowsheet is available in the guide, `Multi r-eNRTL MED Flowsheet <https://github.com/watertap-org/watertap-renrtl/blob/main/src/watertap_contrib/rENRTL/examples/flowsheets/mvc_with_refined_enrtl/how_to_use_refined_enrtl_in_mvc.rst>`_. 

Main Assumptions
^^^^^^^^^^^^^^^^
Main Assumptions 1-4 from `single-electrolyte refined eNRTL model (single r-eNRTL) <https://github.com/watertap-org/watertap-renrtl/blob/main/src/watertap_contrib/rENRTL/examples/flowsheets/mvc_with_refined_enrtl/how_to_setup_refined_enrtl.rst>`_ are carried over into the multi r-eNRTL method. Additionally, new assumptions specific to the multi-electrolyte case are made:

1. **Mean molal ionic activity coefficient (MIAC)**: This activity coefficient is specific to each electrolyte or ion-pair. In the single-electrolyte case, there is only one MIAC. For the multi-electrolyte case, however, the number of MIACs is the product of the number of cation (:math:`Num_c`) and anion species (:math:`Num_a`) present in the solution.

2. **New term** :math:`ar_avg`: The :math:`ar_avg` term represents the mean distance between cations and anions in the Pitzer-Debye-Hückel (PDH) interaction, which describes the Coulombic interactions between ions. This term is calculated as a weighted average of the :math:`ar` term, the direct distance between ions. For detailed calculations of :math:`ar_avg`, refer to the function :math:`rule_ar_avg` in `refined_eNRTL_multi.py <https://github.com/PSORLab/NAWIConcentratedElectrolytes/blob/Pengfei-Xu-1993-patch-1/flowsheets/benchmark_system/mvc/Multiple%20Electrolyte%20r-eNRTL/refined_enrtl_multi.py>`_.

3. **Solvent**: In the multi r-eNRTL method, water is the only supported solvent, even though r-eNRTL model theoretically supports additional solvents [4].

4. **Hydration**: The multi r-eNRTL method supports only the constant hydration model, :math:`constant_hydration`. In contrast, the single r-eNRTL method supports both the constant hydration model (:math:`constant_hydration`) and the stepwise hydration model (:math:`stepwise_hydration`).

Set up for Configuration Dictionary
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The entries in the configuration dictionary are outlined in this section:

:math:`Components`

1. :math:`Solvent`: In the multi r-eNRTL, water is the default solvent (refer to Main Assumption 3). Users should specify :math:`solvent` in their configuration file exactly as shown in the `Code Example` section below.

2. :math:`electrolyte`: The :math:`electrolyte` entry specifies all possible cation-anion combinations in the solution. For more details, refer to `How to Use Apparent and True Chemical Species <https://watertap.readthedocs.io/en/latest/how_to_guides/how_to_use_apparent_and_true_chemical_species.html>`_. In the :math:`parameter_data` entry, note that the :math:`hydration_constant` keyword applies only to the stepwise hydration model, which the multi r-eNRTL does not support. Users can set it to an arbitrary number.

3. :math:`ions`: The :math:`ions` entry includes :math:`type`, :math:`charge`, and :math:`parameter_data`. The first two are straightforward. Detailed information on :math:`parameter_data` is provided in the table below:

.. csv-table::
    :header: "Data", "Units", "Definition", "Resources"

    ":math:`mw`", ":math:`kg/mole`", "Molecular weight of ions", "Available from any database or literature source"
    ":math:`hydration_number`", ":math:`dimensionless`", "Hydration number for each ions", "Table 2 in reference [1] or through `parameter estimation <https://idaes-pse.readthedocs.io/en/1.11.0/user_guide/workflow/data_rec_parmest.html>`_"
    ":math:`ionic_radius`", ":math:`angstrom`", "Ionic radius", "Table 1 in reference [3]"
    ":math:`partial_vol_mol`", ":math:`cm**3 / mole`", "Partial molar volume of ions", "Table 5.8 in reference [5]"
    ":math:`min_hydration_number`", ":math:`dimensionless`", "Parameter for stepwise hydration model (not supported here)", "Set to 0"
    ":math:`number_sites`", ":math:`dimensionless`", "Parameter for stepwise hydration model (not supported here)", "Set to 4 (reference [2])"


:math:`state_definition` 

For more information, please refer to the `State Definition <https://idaes-pse.readthedocs.io/en/stable/explanations/components/property_package/general/state_definition.html>`_ page.

:math:`state_components`

The :math:`StateIndex` must be defined as :math:`true`, as the WaterTAP defines all speciation reactions based on true species rather than apparent species.

:math:`pressure_ref` and :math:`temperature_ref`

These represent the reference pressure and temperature.

:math:`parameter_data`

1. **Hydration Model**: The multi r-eNRTL model supports only the constant hydration model, :math:`constant_hydration`. The parameters :math:`min_hydration_number` and :math:`number_sites` should be set to the fixed value specified in the table above.

2. **Liq_tau**: The binary parameters :math:`\tau` are categorized into two types:

   - (a) :math:`\tau_{(solvent, ion_pair[elect_i])}` and :math:`\tau_{(ion_pair[elect_i], solvent)}`
   - (b) :math:`\tau_{(ion_pair[elect_i],ion_pair[elect_j])}`
   
     where :math:`elect_i` differs from :math:`elect_j`.

     Note that :math:`elect_i` and :math:`elect_j` must share one, and only one, common ion.

   The :math:`\tau` values in type (a) are obtained through parameter estimation using experimental data for activity coefficients in single-electrolyte solutions. For this case, the :math:`\tau` values are sourced from Table 3 in reference [1]. Note that :math:`\tau` are temperature-dependent parameters but are treated as constants in this method. Users should make sure that the selected :math:`\tau` parameters match the system's temperature, as :math:`\tau` values generally vary across different temperatures. 

   The :math:`\tau` values in type (b) are obtained through parameter estimation using experimental data for activity coefficients in multi-electrolyte solutions. In this example, these :math:`\tau` values are set to 0. This is common in configuring the regular eNRTL model. For a system with :math:`x` species of cations and :math:`y` species of anions, users must provide the corresponding number of :math:`\tau` values as shown below:

   - :math:`N_{\tau_{(solvent, ion_pair[elect_i])}} = x * y`

     :math:`N_{\tau_{(ion_pair[elect_i], solvent)}}= x * y`

     where :math:`N` represents the number of :math:`\tau` parameters.
     
   - :math:`N_{\tau_{(ion_pair[elect_i],ion_pair[elect_j])}} = x * y * (x - 1)`

     or

     :math:`N_{\tau_{(ion_pair[elect_i],ion_pair[elect_j])}} = y * x * (y - 1)`

     where these do not need to be provided if they are set to 0.

Code Example
^^^^^^^^^^^^
An example configuration dictionary set up for the multi r-eNRTL is provided below:

.. code-block::

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
   
   # Import r-eNRTL
   from idaes.models.properties.modular_properties.eos.enrtl import ENRTL
   from refined_enrtl_multi import rENRTL
   
   from idaes.core.util.model_statistics import degrees_of_freedom

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
       return AM + BM * (1 / T * pyunits.K - 1 / CM)
   
   
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
   

   # Declare configuration dictionary for r-eNRTL
   equation_of_state = rENRTL  # method class for config dic
   # Only one hydration model is available in the multi-electrolyte:
   # constant_hydration
   hydration_model = "constant_hydration"

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
        # hydration_number: Table 2 from ref [1]
        # ionic_radius: Table 1 from ref [3]
        # partial_vol_mol: Table 5.8 from ref [2] 
        # number_sites: ref [1] Page 4: "all the number of sites are assigned as 4". Please note that number_sites are not used in this method.
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
               "equation_of_state_options": {
                   # "reference_state": Unsymmetric}}},
                   # "reference_state": Symmetric
               },
           }
       },
       "base_units": {
           "time": pyunits.s,
           "length": pyunits.m,
           "mass": pyunits.kg,
           "amount": pyunits.mol,
           "temperature": pyunits.K,
       },
       # "state_definition": FpcTP,
       "state_definition": FTPx,
       "state_components": StateIndex.true,
       "pressure_ref": 101325,
       "temperature_ref": 298.15,
       "parameter_data": {
           "hydration_model": hydration_model,  # constant_hydration or stepwise_hydration
           "Liq_tau": {
               # Table 3 from ref [1]
               (solvent, ion_pair[elect_1]): 7.951,
               (ion_pair[elect_1], solvent): -3.984,
               (solvent, ion_pair[elect_2]): 7.578,
               (ion_pair[elect_2], solvent): -3.532,
               (ion_pair[elect_1], ion_pair[elect_2]): 0,
               (ion_pair[elect_2], ion_pair[elect_1]): 0,
           },
       },
   }


Reference:
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

[1] Xi Yang, Paul I. Barton, and George M. Bollas, Refined
electrolyte-NRTL model: Inclusion of hydration for the detailed
description of electrolyte solutions. Part I: Single electrolytes up
to moderate concentrations, single salts up to solubility limit.
Journal publication currently in progress.

[2] Y. Marcus, Ion solvation, Wiley-Interscience, New York, 1985.

[3] Y. Marcus, Thermodynamics of solvation of ions. Part 5.—Gibbs free energy of hydration at
298.15 K, J. Chem. Soc., Faraday Trans. 87 (1991) 2995–2999. doi:10.1039/FT9918702995

[4] Bollas, G. M., Chen, C. C., & Barton, P. I. (2008). Refined electrolyte‐NRTL model: Activity
coefficient expressions for application to multi‐electrolyte systems. AIChE journal, 54(6), 1608
-1624. doi:10.1002/aic.11485