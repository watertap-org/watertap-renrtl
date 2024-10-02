###############################################################################
# WaterTAP Copyright (c) 2020-2024, The Regents of the University of California,
# through Lawrence Berkeley National Laboratory, Oak Ridge National Laboratory,
# National Renewable Energy Laboratory, and National Energy Technology
# Laboratory (subject to receipt of any required approvals from the U.S. Dept.
# of Energy). All rights reserved.
#
# Copyright 2024, National Technology & Engineering Solutions of Sandia,
# LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the
# U.S. Government retains certain rights in this software
#
# Copyright 2024, Pengfei Xu and Matthew D. Stuber and the University
# of Connecticut.
#
# Please see the files COPYRIGHT.md and LICENSE.md for full copyright and license
# information, respectively. These files are also available online at the URL
# "https://github.com/watertap-org/watertap/"
#
###############################################################################
"""
Sub-methods for eNRTL activity coefficient method.

Includes temperature dependence rules for alpha and tau
"""
# TODO: Missing docstrings
# pylint: disable=missing-function-docstring

from pyomo.environ import Reals, units as pyunits, Var, log
from pyomo.environ import *
from idaes.core.util.exceptions import BurntToast, ConfigurationError
import idaes.logger as idaeslog

# Set up logger
_log = idaeslog.getLogger(__name__)


class TemperatureTau(object):
    """Class for methods assuming consant tau"""

    @staticmethod
    def build_parameters(b):
        param_block = b.parent_block()

        # Get user provided values for tau (if present)
        try:
            tau_data = param_block.config.parameter_data[b.local_name + "_tau"]
        except KeyError:
            tau_data = {}

        # Check for unused parameters in tau_data
        for i, j in tau_data.keys():
            if (i, j) not in b.component_pair_set:
                raise ConfigurationError(
                    "{} eNRTL tau parameter provided for invalid "
                    "component pair {}. Please check typing and only provide "
                    "parameters for valid species pairs.".format(b.name, (i, j))
                )

        def tau_init(b, i, j):
            if (i, j) in tau_data.keys():
                v = tau_data[(i, j)]
            else:
                # Default interaction value is 0
                v = 0
            return v

        b.add_component(
            "tau",
            Var(
                b.component_pair_set,
                within=Reals,
                initialize=tau_init,
                doc="Binary interaction energy parameters",
                units=pyunits.dimensionless,
            ),
        )

    @staticmethod
    def return_expression(b, pobj, i, j, T):
        tau_data = {
            ("H2O", "Na+, Cl-"): pobj.tau["H2O", "Na+, Cl-"],
            ("Na+, Cl-", "H2O"): pobj.tau["Na+, Cl-", "H2O"],
            ("H2O", "H2O"): 0,
        }
        if (i, j) in pobj.tau:
            return tau_data[i, j]
        elif i == j:
            return 0
        else:
            raise BurntToast(
                "{} tau rule encountered unexpected index {}. Please contact"
                "the IDAES Developers with this bug.".format(b.name, (i, j))
            )
