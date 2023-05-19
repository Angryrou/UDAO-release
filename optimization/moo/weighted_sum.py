# Author(s): Qi FAN <qi dot fan at polytechnique dot edu>
#
# Description: Weighted Sum
#
# Created at 15/09/2022

from optimization.solver.grid_search import GridSearch
from optimization.solver.random_sampler import RandomSampler
import utils.optimization.moo_utils as moo_ut
from optimization.moo.base_moo import BaseMOO

import numpy as np

class WeightedSum(BaseMOO):

    def __init__(self, ws_pairs: list, inner_solver: str, solver_params: dict, n_objs: int, obj_funcs: list,
                 opt_type: list, const_funcs: list, const_types: list):
        '''
        parameters used in Weighted Sum
        :param ws_pairs: list, even weight settings for all objectives, e.g. for 2d, [[0, 1], [0.1, 0.9], ... [1, 0]]
        :param inner_solver: str, the name of the solver used in Weighted Sum
        :param solver_params: dict, parameter used in solver, e.g. in grid-search, it is the number of grids for each variable
        :param n_objs: int, the number of objectives
        :param obj_funcs: list, objective functions
        :param opt_type: list, objectives to minimize or maximize
        :param const_funcs: list, constraint functions
        :param const_types: list, constraint types ("<=", "==" or ">=", e.g. g1(x1, x2, ...) - c <= 0)
        '''
        super().__init__()
        self.inner_solver = inner_solver
        self.ws_pairs = ws_pairs
        self.n_objs = n_objs
        self.obj_funcs = obj_funcs
        self.opt_type = opt_type
        self.const_funcs = const_funcs
        self.const_types = const_types
        if self.inner_solver == "grid_search":
            self.gs = GridSearch(solver_params)
        elif self.inner_solver == "random_sampler":
            self.rs = RandomSampler(solver_params)
        else:
            raise Exception(f"WS does not support {self.inner_solver}!")

    def solve(self, wl_id, var_ranges, var_types):
        '''
        solve MOO by Weighted Sum (WS)
        :param wl_id: str, workload id
        :param var_ranges: ndarray(n_vars,), lower and upper var_ranges of variables(non-ENUM), and values of ENUM variables
        :param var_types: list, variable types (float, integer, binary, enum)
        :return: po_objs: ndarray(n_solutions, n_objs), Pareto solutions
                 po_vars: ndarray(n_solutions, n_vars), corresponding variables of Pareto solutions
        '''
        # TODO: we will compare the current WS implementation with the existing WS numerical solver in the future, and the one with better performance will be kept in the package.
        if self.inner_solver == "grid_search":
            vars = self.gs._get_input(var_ranges, var_types)
        elif self.inner_solver == "random_sampler":
            vars = self.rs._get_input(var_ranges, var_types)
        else:
            raise Exception(f"WS does not support {self.inner_solver}")

        const_violation = self._get_const_violation(wl_id, vars)

        # remove vars who lead to constraint violation
        if const_violation.size != 0:
        # if (const_violation.size != 0) & (const_violation.max() > 0):
            ## find the var index which violate the constraint
            n_const = const_violation.shape[1]
            available_indices = range(const_violation.shape[0])
            for i in range(n_const):
                if self.const_types[i] == "<=":
                    available_indice = np.where(const_violation[:, i] <= 0)
                elif self.const_types[i] == ">=":
                    available_indice = np.where(const_violation[:, i] >= 0)
                elif self.const_types[i] == "==":
                    available_indice = np.where(const_violation[:, i] == 0)
                else:
                    raise Exception(f"No feasible constraints provided! Please check constraint type settings in configurations. We do not support {self.const_types}.")
                available_indices = np.intersect1d(available_indice, available_indices)
            vars_after_const_check = vars[available_indices]
        else:
            vars_after_const_check = vars

        if vars_after_const_check.size == 0:
            print("NO feasible solutions found!")
            return None, None
        else:
            po_obj_list, po_var_list = [], []

            # get n_dim objective values
            objs = []
            for i, obj_func in enumerate(self.obj_funcs):
                if wl_id == None:
                    obj = obj_func(vars_after_const_check) * moo_ut._get_direction(self.opt_type, i)
                else:
                    obj = obj_func(wl_id, vars_after_const_check) * moo_ut._get_direction(self.opt_type, i)
                objs.append(obj.squeeze())

            # transform objs to array: (n_samples/grids * n_objs)
            objs = np.array(objs).T

            # normalization
            objs_min, objs_max = objs.min(0), objs.max(0)

            if all((objs_min - objs_max) < 0):
                objs_norm = (objs - objs_min) / (objs_max - objs_min)
                for ws in self.ws_pairs:
                    po_ind = self.get_soo_index(objs_norm, ws)
                    po_obj_list.append(objs[po_ind])
                    po_var_list.append(vars_after_const_check[po_ind])

                # only keep non-dominated solutions
                return moo_ut._summarize_ret(po_obj_list, po_var_list)
            else:
                raise Exception(f"Cannot do normalization! Lower bounds of objective values are higher than their upper bounds.")

    def get_soo_index(self, objs, ws_pairs):
        '''
        reuse code in VLDB2022
        :param objs: ndarray(n_feasible_samples/grids, 2)
        :param ws_pairs: list, one weight setting for all objectives, e.g. [0, 1]
        :return: int, index of the minimum weighted sum
        '''
        obj = np.sum(objs * ws_pairs, axis=1)
        return np.argmin(obj)

    def _get_const_violation(self, wl_id, vars):
        '''
        get violation of each constraint
        :param wl_id: str, workload id
        :param vars: ndarray(n_grids/n_samples, 2), variables
        :return: ndarray(n_samples/grids, n_const), constraint violations
        '''
        if wl_id == None:
            g_list = [const_func(vars) for const_func in self.const_funcs]
        else:
            g_list = [const_func(wl_id, vars) for const_func in self.const_funcs]

        # shape (n_samples/grids, n_const)
        return np.array(g_list).T

