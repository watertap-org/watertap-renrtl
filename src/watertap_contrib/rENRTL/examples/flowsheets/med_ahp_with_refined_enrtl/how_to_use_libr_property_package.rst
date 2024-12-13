LiBr Property Package
=====================

This package implements property relationships for a LiBr-water solution.

This LiBr property package:
   * supports only H2O (solvent) and TDS (LiBr solute) components 
   * supports only liquid phase
   * is formulated on a mass basis
   * does not support dynamics

Sets
----
.. csv-table::
   :header: "Description", "Symbol", "Indices"

   "Components", ":math:`j`", "'H2O', 'TDS'"
   "Phases", ":math:`p`", "'Liq'"

State variables
---------------
.. csv-table::
   :header: "Description", "Symbol", "Variable", "Index", "Units"

   "Component mass flowrate", ":math:`M_j`", "flow_mass_phase_comp", "p, j", ":math:`\text{kg/s}`"
   "Temperature", ":math:`T`", "temperature", "None", ":math:`\text{K}`"
   "Pressure", ":math:`P`", "pressure", "None", ":math:`\text{Pa}`"

Properties
----------

.. csv-table::
   :header: "Description", "Symbol", "Variable", "Units"

   "Component mass fraction", ":math:`x_j`", "mass_frac_phase_comp", ":math:`\text{dimensionless}`"
   "Mass density", ":math:`\rho`", "dens_mass_phase", ":math:`\text{kg/}\text{m}^3`"
   "Mass density of pure water", ":math:`\rho`", "dens_mass_solvent", ":math:`\text{kg/}\text{m}^3`"
   "Phase volumetric flowrate", ":math:`Q_p`", "flow_vol_phase", ":math:`\text{m}^3\text{/s}`"
   "Volumetric flowrate", ":math:`Q`", "flow_vol", ":math:`\text{m}^3\text{/s}`"
   "Mass concentration", ":math:`C_j`", "conc_mass_phase_comp", ":math:`\text{kg/}\text{m}^3`"
   "Component mole flowrate", ":math:`N_j`", "flow_mol_phase_comp", ":math:`\text{mole/s}`"  
   "Component mole fraction", ":math:`y_j`", "mole_frac_phase_comp", ":math:`\text{dimensionless}`" 
   "Molality", ":math:`Cm_{TDS}`", "molality_phase_comp", ":math:`\text{mole/kg}`"
   "Viscosity", ":math:`\mu`", "visc_d_phase", ":math:`\text{Pa}\cdotp\text{s}`"
   "Enthalpy", ":math:`\widehat{H}`", "enth_mass_phase", ":math:`\text{J/kg}`"
   "Enthalpy flow", ":math:`H`", "enth_flow", ":math:`\text{J/s}`"
   "Vapor temperature of water", ":math:`tsat_w`", "temperature_sat_solvent", ":math:`\text{K}`"
   "Vapor temperature", ":math:`tsat`", "temperature_sat", ":math:`\text{K}`"
   "Vapor pressure of water", ":math:`psat_w`", "pressure_sat", ":math:`\text{Pa}`"
   "Specific heat capacity", ":math:`cp`", "cp_mass_phase", ":math:`\text{J/kg.K}`"   
   "Thermal conductivity", ":math:`K`", "therm_cond_phase", ":math:`\text{W/m.K}`"

Property Equations
-------------

.. csv-table::
   :header: "Description", "Equation"

   "Component mass fraction", ":math:`X_j = \\frac{M_j}{\\sum_{j} M_j}`"
   "Mass density [2]", ":math:`1145.36 + 470.84 * X_j + 1374.79 * X_j**2 - (0.333393 + 0.571749 * X_j) * T`"
   "Mass density of water [1]", ":math:`9.999e2 + (2.034e-2 * T) + (-6.162e-3 * T**2) + (2.261e-5 * T**3) + (-4.657e-8*T**4)`"
   "Phase volumetric flowrate", ":math:`Q = \\frac{\\sum_{j} M_j}{\\rho}`"
   "Volumetric flowrate", ":math:`Q = \\frac{\\sum_{j} M_j}{\\rho}`"
   "Mass concentration", ":math:`C_j = X_j \\cdot \\rho`"
   "Component mole flowrate", ":math:`N_j = \\frac{M_j}{MW_j}`"
   "Component mole fraction", ":math:`y_j = \\frac{N_j}{\\sum_{j} N_j}`"
   "Molality", ":math:`Cm_{TDS} = \\frac{x_{TDS}}{(1-x_{TDS}) \\cdot MW_{TDS}}`"
   "Viscosity [3]", ":math:`0.001*(((-494.122 + 16.3967 * X_j - 0.14511 * (X_j)**2)) + (((28606.4 - 934.568 * X_j + 8.52755 * (X_j)**2))/T) + ((70.3848 - 2.35014 * X_j + 0.0207809 * (X_j * s)**2)) * log(T))`"
   "Enthalpy", ":math:`\widehat{H} = cp \\cdot (T - 273.15)`"
   "Enthalpy flow", ":math:`H = \\sum_{j} M_j \\cdot \\widehat{H}`"
   "Vapor temperature of water", ":math:`(42.67776 - 3892.7/ (log(P / 1000000) - 9.48654))`"
   "Vapor temperature [2]", ":math:`(((0 + 16.634856 * X_j - 553.38169 * X_j ** 2 + 11228.336 * X_j ** 3 - 110283.9 * X_j ** 4 + 621094.64 * X_j ** 5 - 2111256.7 * X_j ** 6 + 4385190.1 * X_j ** 7 - 5409811.5 * X_j ** 8 + 3626674.2 * X_j ** 9 - 1015305.9 * X_j ** 10)) + (T) * ((1 - 0.068242821 * X_j + 5.873619 * X_j ** 2 - 102.78186 * X_j ** 3 + 930.32374 * X_j ** 4 - 4822.394 * X_j ** 5 + 15189.038 * X_j ** 6 - 29412.863 * X_j ** 7 + 34100.528 * X_j ** 8 - 21671.48 * X_j ** 9 + 5799.56 * X_j ** 10)))`"
   "Vapor pressure of water [3]", ":math:`(10**((7.05 - 1596.49/(Tsat) - 104095.5/(Tsat)**2)/))`"
   "Specific heat capacity [3]", ":math:`(0.0976 * (X_j)**2 - 37.512 * X_j + 3825.4)`"
   "Thermal conductivity [3]", ":math:`((-0.3081 * X_j + 0.62979)) + (((((-0.291897 * X_j + 0.59821)) - ((-0.3081 * X_j + 0.62979))) * (313 - T)/(20)))`"

Scaling
-------
This  property package includes support for scaling, such as providing default or calculating scaling factors for almost all variables. 
The component mass flowrate is the only variable without scaling factors. This should be set by the user.

The user can specify the scaling factors for component mass flowrates with the following:

.. testsetup::

   from pyomo.environ import ConcreteModel
   from idaes.core import FlowsheetBlock

.. doctest::
   
   # relevant imports
   import watertap.property_models.LiBr_prop_pack as props
   from idaes.core.util.scaling import calculate_scaling_factors

   # relevant assignments
   m = ConcreteModel()
   m.fs = FlowsheetBlock(dynamic=False)
   m.fs.properties = props.LiBrParameterBlock()

   # set scaling for component mass flowrate
   m.fs.properties.set_default_scaling('flow_mass_phase_comp', 1, index=('Liq', 'H2O'))
   m.fs.properties.set_default_scaling('flow_mass_phase_comp', 1e2, index=('Liq', 'TDS'))

   # calculate scaling factors
   calculate_scaling_factors(m.fs)

The default scaling factors are as follows:

   * 1e-2 for temperature
   * 1e-6 for pressure
   * 1e-3 for mass density
   * 1e-3 for mass density of pure water
   * 1e3 for viscosity
   * 1e-5 for enthalpy 
   * 1e-2 for vapor temperature of water
   * 1e-2 for vapor temperature
   * 1e-6 vapor pressure of water
   * 1e-3 specific heat capacity
   * 1e0 thermal conductivity

The scaling factors for other variables can be calculated based on their relationships with the other variables with the user supplied or default scaling factors.
   
References
----------

[1] Sharqawy, Mostafa H.; Lienhard, John H.; Zubair, Syed M. (2010). Thermophysical properties of seawater: a review of existing correlations and data. Desalination and Water Treatment, 16(1-3), 354-380. `DOI: 10.5004/dwt.2010.1079 <https://doi.org/10.5004/dwt.2010.1079>`_
           
[2] Hellmann, H. M., & Grossman, G. (1996). Improved property data correlations of absorption fluids for computer simulation of heat pump cycles. ASHRAE Transactions, 102(1), 980-997. OSTI ID:392525 
        
[3] G.A. Florides, S.A. Kalogirou, S.A. Tassou, L.C. Wrobel, Design and construction of a LiBr-water absorption machine, Energy Conversion & Management, 2002. `DOI: 10.1016/S0196-8904(03)00006-2 <https://doi.org/10.1016/S0196-8904(03)00006-2>`
