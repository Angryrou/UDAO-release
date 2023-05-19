# Author(s): Qi FAN <qi dot fan at polytechnique dot edu>
#
# Description: moo entry point
#
# Created at 21/09/2022
import time
import numpy as np

from optimization.moo.weighted_sum import WeightedSum
from optimization.moo.progressive_frontier import ProgressiveFrontier
from optimization.moo.evolutionary import EVO
import utils.optimization.moo_utils as moo_ut
from utils.parameters import VarTypes

class GenericMOO:

    def __init__(self):
        pass

    def problem_setup(self, obj_names: list, obj_funcs: list, opt_types: list, const_funcs: list, const_types: list,
                      var_types: list, var_ranges: list,
                      obj_types=None, wl_list=None, wl_ranges=None, vars_constraints=None, accurate=None, std_func=None):
        '''
        setup common input paramters for MOO problems
        :param obj_names: list, objective names
        :param obj_funcs: list, objective functions
        :param opt_types: list, objectives to minimize or maximize
        :param const_funcs: list, constraint functions
        :param const_types: list, constraint types ("<=" "==" or ">=", e.g. g1(x1, x2, ...) - c <= 0)
        :param var_types: list, variable types (float, integer, binary, enum)
        :param var_ranges: ndarray(n_vars, ), lower and upper var_ranges of variables(non-ENUM), and values of ENUM variables
        :param wl_list: list, each element is a string to indicate workload id (fixme)
        :param wl_ranges: dict, each key is the workload id (fixme)
        :param vars_constraints: dict, keys are "conf_min" and "conf_max" to indicate the variable range (only used in MOGD)
        :param accurate: bool, to indicate whether the predictive model is accurate (True) (used in MOGD)
        :param std_func: function, used in in-accurate predictive models
        :return:
        '''
        self.obj_names = obj_names
        self.obj_funcs = obj_funcs
        self.opt_types = opt_types
        self.const_funcs = const_funcs
        self.const_types = const_types
        self.var_types = var_types
        self.var_ranges = var_ranges

        # used in MOGD
        self.obj_types = obj_types
        # self.wl_list = wl_list
        self.wl_ranges = wl_ranges
        self.vars_constraints = vars_constraints
        self.accurate = accurate
        self.std_func = std_func

    def solve(self, moo_algo: str, solver: str, add_params: list):
        '''
        solve MOO problems internally by different MOO algorithms
        :param moo_algo: str, the name of moo algorithm
        :param solver: str, the name of solver
        :param add_params: list, the parameters required by the specified MOO algorithm and solver
        :return: po_objs_list: list, each element is solutions (ndarray(n_solutions, n_objs)) for one job
                 po_vars_list: list, each element is solutions (ndarray(n_solutions, n_vars)) for one job, corresponding variables of MOO solutions
                 job_Ids: list, workload ids, each element is a string or None.
                 time_cost_list: list, each element is the time cost of MOO solving for one job.
        '''
        if moo_algo == "weighted_sum":
            file_path = add_params[0]
            job_ids = np.loadtxt(file_path, dtype='str', delimiter=',').tolist()
            if job_ids == "None":
                job_ids = [None]
            elif isinstance(job_ids, str):
                job_ids = [job_ids]
            elif isinstance(job_ids, list):
                if "" in job_ids:
                    raise Exception(f"job ids {job_ids} contain empty string!")
                else:
                    pass
            else:
                raise Exception(f"job ids {job_ids} are not well defined!")

            n_probes = add_params[1]
            solver_params = add_params[2]
            n_objs = len(self.opt_types)
            ws_steps = 1 / (n_probes - n_objs - 1)
            ws_pairs = moo_ut.even_weights(ws_steps, n_objs)
            ws = WeightedSum(ws_pairs, solver, solver_params, n_objs, self.obj_funcs, self.opt_types,
                             self.const_funcs, self.const_types)
            po_objs_list, po_vars_list = [], []
            time_cost_list = []
            for wl_id in job_ids:
                # fixme: to be generalized further
                if self.wl_ranges != None:
                    vars_max, vars_min = self.wl_ranges(wl_id)
                    vars_ranges = np.vstack((vars_min, vars_max)).T
                    #find indices of non_ENUM vars
                    non_enum_inds = [i for i, var_type in enumerate(self.var_types) if var_type != VarTypes.ENUM]
                    vars_ranges[non_enum_inds] = self.var_ranges[non_enum_inds]
                    self.var_ranges[non_enum_inds] = list(vars_ranges[non_enum_inds].tolist())
                else:
                    pass
                start_time = time.time()
                po_objs, po_vars = ws.solve(wl_id, self.var_ranges, self.var_types)
                time_cost = time.time() - start_time
                po_objs_list.append(po_objs)
                po_vars_list.append(po_vars)
                time_cost_list.append(time_cost)

            return po_objs_list, po_vars_list, job_ids, time_cost_list

        elif moo_algo == 'progressive_frontier':
            precision_list = add_params[0]
            pf_option = add_params[1]
            n_probes = add_params[2]
            n_grids = add_params[3]
            max_iters = add_params[4]
            file_path = add_params[5]
            accurate = add_params[6]
            alpha = add_params[7]
            anchor_option = add_params[8]
            opt_obj_ind = add_params[9]
            mogd_params = add_params[10]

            job_ids = np.loadtxt(file_path, dtype='str', delimiter=',').tolist()
            if job_ids == "None":
                job_ids = [None]
            elif isinstance(job_ids, str):
                job_ids = [job_ids]
            elif isinstance(job_ids, list):
                if "" in job_ids:
                    raise Exception(f"job ids {job_ids} contain empty string!")
                else:
                    pass
            else:
                raise Exception(f"job ids {job_ids} are not well defined!")
            self.wl_list = job_ids

            pf = ProgressiveFrontier(pf_option, solver, mogd_params, self.obj_names, self.obj_funcs, self.opt_types, self.obj_types,
                                     self.const_funcs, self.const_types, opt_obj_ind, self.wl_list, self.wl_ranges, self.vars_constraints, self.accurate, self.std_func)
            po_objs_list, po_vars_list = [], []
            time_cost_list = []
            for wl_id in job_ids:
                start_time = time.time()
                po_objs, po_vars = pf.solve(wl_id, accurate, alpha, self.var_ranges, self.var_types, precision_list, n_probes, n_grids=n_grids, max_iters=max_iters, anchor_option=anchor_option)
                time_cost = time.time() - start_time
                po_objs_list.append(po_objs)
                po_vars_list.append(po_vars.squeeze())
                time_cost_list.append(time_cost)

            return po_objs_list, po_vars_list, job_ids, time_cost_list
        elif moo_algo == 'evolutionary':
            file_path = add_params[0]
            job_ids = np.loadtxt(file_path, dtype='str', delimiter=',').tolist()
            if job_ids == "None":
                job_ids = [None]
            elif isinstance(job_ids, str):
                job_ids = [job_ids]
            elif isinstance(job_ids, list):
                if "" in job_ids:
                    raise Exception(f"job ids {job_ids} contain empty string!")
                else:
                    pass
            else:
                raise Exception(f"job ids {job_ids} are not well defined!")
            inner_algo = add_params[1]
            pop_size = add_params[2]
            # the number of function evaluations
            nfe = add_params[3]
            flag = add_params[4]
            seed = add_params[5]
            evo = EVO(inner_algo, self.obj_funcs, self.opt_types, self.const_funcs, self.const_types, pop_size, nfe,
                      fix_randomness_flag=flag, seed=seed)
            po_objs_list, po_vars_list = [], []
            time_cost_list = []
            for wl_id in job_ids:
                # fixme: to be generalized further
                if self.wl_ranges != None:
                    vars_max, vars_min = self.wl_ranges(wl_id)
                    vars_ranges = np.vstack((vars_min, vars_max)).T
                    # find indices of non_ENUM vars
                    non_enum_inds = [i for i, var_type in enumerate(self.var_types) if var_type != VarTypes.ENUM]
                    vars_ranges[non_enum_inds] = self.var_ranges[non_enum_inds]
                    self.var_ranges[non_enum_inds] = list(vars_ranges[non_enum_inds].tolist())
                else:
                    pass

                start_time = time.time()
                po_objs, po_vars = evo.solve(wl_id, self.var_ranges, self.var_types)
                time_cost = time.time() - start_time
                po_objs_list.append(po_objs)
                po_vars_list.append(po_vars)
                time_cost_list.append(time_cost)
            return po_objs_list, po_vars_list, job_ids, time_cost_list

        else:
            raise NotImplementedError

