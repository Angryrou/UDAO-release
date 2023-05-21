# Author(s): Qi FAN <qi dot fan at polytechnique dot edu>
#            Chenghao Lyu <chenghao at cs dot umass dot edu>
#
# Description: parse configurations
# Refer to the configs_parser.py in UDAO repository
#
# Created at 21/09/2022

import argparse, json
import numpy as np

from utils.parameters import VarTypes

class ConfigsParser():
    def __init__(self):
        parser = argparse.ArgumentParser(description="ConfigsParser")
        parser.add_argument("-c", "--config", required=True,
                            help="the configuration file location")
        self.parser = parser

    def parse(self):
        return self.parser.parse_args()

    def parse_details(self, option=None):
        args = self.parse()
        try:
            with open(args.config) as f:
                configs = json.load(f)
        except:
            raise Exception(f"{args.config} does not exist")

        if option is None:
            # Load the configuration information correctly
            try:
                moo_algo = configs['moo_algo']
                solver = configs['solver']
                var_types, var_ranges = self.get_vars_conf(configs['variables'])
                obj_names, opt_types, obj_types = self.get_objs_conf(configs['objectives'])
                const_types, const_names = self.get_const(configs['constraints'])
                add_params = []
                if moo_algo == "weighted_sum":
                    jobIds_path = configs['additional_params']["jobIds_path"]
                    add_params.append(jobIds_path)
                    n_probes = configs['additional_params']['n_probes']
                    add_params.append(n_probes)
                    solver_params = configs['additional_params'][
                        'solver_params']  # the number of grids/samples per variables
                    add_params.append(solver_params)
                elif moo_algo == "progressive_frontier":
                    precision_list = self.get_precision_list(configs['variables'])
                    add_params.append(precision_list)
                    pf_option = configs['additional_params']["pf_option"]
                    add_params.append(pf_option)
                    n_probes = configs['additional_params']["n_probes"]
                    add_params.append(n_probes)
                    n_grids = configs['additional_params']["n_grids"]
                    add_params.append(n_grids)
                    max_iters = configs['additional_params']["max_iters"]
                    add_params.append(max_iters)
                    jobIds_path = configs['additional_params']["jobIds_path"]
                    add_params.append(jobIds_path)
                    accurate = configs['additional_params']["accurate"]
                    add_params.append(accurate)
                    alpha = configs['additional_params']["alpha"]
                    add_params.append(alpha)
                    anchor_option = configs['additional_params']["anchor_option"]
                    add_params.append(anchor_option)
                    obj_opt_ind = configs['additional_params']["opt_obj_ind"]
                    add_params.append(obj_opt_ind)

                    mogd_params = configs['additional_params']["mogd_params"]
                    add_params.append(mogd_params)

                elif moo_algo == "evolutionary":
                    jobIds_path = configs['additional_params']["jobIds_path"]
                    add_params.append(jobIds_path)
                    inner_algo = configs['additional_params']['inner_algo']
                    add_params.append(inner_algo)
                    pop_size = configs['additional_params']['pop_size']
                    add_params.append(pop_size)
                    nfe = configs['additional_params']['nfe']
                    add_params.append(nfe)
                    flag = configs['additional_params']['fix_randomness_flag']
                    add_params.append(flag)
                    seed = configs['additional_params']['seed']
                    add_params.append(seed)
                else:
                    raise Exception(f"Algorithm {moo_algo} is not configured")

            except:
                raise Exception(f"configurations are not well specified.")

            return [moo_algo, solver, var_types, var_ranges, obj_names, opt_types, obj_types, const_types, const_names, add_params]
        else:
            model_params = configs["model"]
            return model_params

    def get_vars_conf(self, var_params):
        var_types, var_bounds = [], []

        for var in var_params:
            if var["type"] == "FLOAT":
                var_types.append(VarTypes.FLOAT)
                var_bounds.append([var["min"], var["max"]])
            elif var["type"] == "INTEGER":
                var_types.append(VarTypes.INTEGER)
                var_bounds.append([var["min"], var["max"]])
            elif var["type"] == "BINARY":
                var_types.append(VarTypes.BOOL)
                var_bounds.append([var["min"], var["max"]])
            elif var["type"] == "ENUM":
                var_types.append(VarTypes.ENUM)
                enum_values = var["values"]
                var_bounds.append(enum_values)
            else:
                error_var_type = var["type"]
                raise Exception(f"Variable type {error_var_type} is not supported!")

        return var_types, np.array(var_bounds)

    def get_objs_conf(self, obj_params):
        """
        get names and optimization types for objectives
        :param obj_params: list, each element is a dict for each constraint, including keys of "name", "optimize_trend", "type"
        :return:
                obj_names: list, objective names
                opt_types: list, optimization types (e.g. minimization or maximization)
        """
        obj_names = [obj["name"] for obj in obj_params]
        opt_types = [obj["optimize_trend"] for obj in obj_params]
        obj_types = []
        for obj in obj_params:
            if obj["type"] == "FLOAT":
                obj_types.append(VarTypes.FLOAT)
            elif obj["type"] == "INTEGER":
                obj_types.append(VarTypes.INTEGER)
            elif obj["type"] == "BINARY":
                obj_types.append(VarTypes.BOOL)
            elif obj["type"] == "ENUM":
                obj_types.append(VarTypes.ENUM)
            else:
                error_var_type = obj["type"]
                raise Exception(f"Variable type {error_var_type} is not supported!")

        return obj_names, opt_types, obj_types

    def get_const(self, const_params):
        """
        get constraint types
        :param const_params: list, each element is a dict for each constraint, including keys of "name", "type"
        :return: list, constraint types
        """
        const_types = [const["type"] for const in const_params]
        const_names = [const["name"] for const in const_params]

        return const_types, const_names

    def get_precision_list(self, vars):
        """
        get precision of each variable (only used in MOGD solver)
        :param vars: list, each element is a dict for each variable, including keys of "name", "type", "min", "max", (or "values"), "precision"
        :return: precision_list: variable precision
        """
        precision_list = [var["precision"] for var in vars]

        return precision_list
