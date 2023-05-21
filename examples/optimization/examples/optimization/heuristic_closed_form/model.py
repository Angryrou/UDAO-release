# Author(s): Chenghao Lyu <chenghao at cs dot umass dot edu>
#            Qi FAN <qi dot fan at polytechnique dot edu>
#
# Description: pre-define functions of Heuristic Closed Form (HCF).
#
# Created at 10/12/22
import numpy as np
import torch as th

from optimization.model.base_model import BaseModel

#     An example: 2D
#     https://en.wikipedia.org/wiki/Test_functions_for_optimization#Test_functions_for_multi-objective_optimization_problems
#     Binh and Korn function:
#     # minimize:
#     #          f1(x1, x2) = 4 * x_1 * x_1 + 4 * x_2 * x_2
#     #          f2(x1, x2) = (x_1 - 5) * (x_1 - 5) + (x_2 - 5) * (x_2 - 5)
#     # subject to:
#     #          g1(x_1, x_2) = (x_1 - 5) * (x_1 - 5) + x_2 * x_2 <= 25
#     #          g2(x_1, x_2) = (x_1 - 8) * (x_1 - 8) + (x_2 + 3) * (x_2 + 3) >= 7.7
#     #          x_1 in [0, 5], x_2 in [0, 3]
#     """

# """
#     An example: 3D
#     https://pymoo.org/problems/many/dtlz.html#DTLZ1
#     DTLZ1 function with 3 objectives and 3 variables:
#     # minimize:
#     #          f1(x_1, x_2, x_3) = 1/2 * x_1 * x_2 * (1 + g(x_3))
#     #          f2(x_1, x_2, x_3) = 1/2 * x_1 * (1 - x_2) * (1 + g(x_3))
#     #          f2(x_1, x_2, x_3) = 1/2 * (1 - x_1) * (1 + g(x_3))
#     # where g(x_3) = 100 * (1 + (x_3 - 0.5)^2 - cos(20 * pi * (x_3 - 0.5)))
#     # subject to:
#     #          x_i in [0, 1], i = 1, 2, 3
#     """
class HCF(BaseModel):
    def __init__(self, obj_names: list, const_names: list, var_ranges: list):
        '''
        :param obj_names: list, name of objectives
        :param const_names: list, constraint names
        :param training_vars: ndarray(n_training_samples, n_vars), input to train GPR models
        '''
        super().__init__(obj_names + const_names)
        self.n_obj = len(obj_names)
        self.initialize(var_ranges)

    def initialize(self, var_ranges):
        '''
        initialize parameters and fit GPR models for objectives separately
        :param objs: list, name of objectives
        :param training_vars: ndarray(n_training_samples, n_vars), input to train GPR models
        :return:
        '''
        self.var_ranges = var_ranges

    def fit(self):
        pass

    def normalize_config(self, config):
        """
        :param *args:
        :param real_conf: numpy.array[int]
        :return: normalized to 0-1
        """
        var_max, var_min = self.get_conf_range_for_wl()
        normalized_conf = (config - var_min) / (var_max - var_min)
        return normalized_conf

    def get_conf_range_for_wl(self, wl_id=None):
        conf_max = self.var_ranges[:, 1]
        conf_min = self.var_ranges[:, 0]
        return conf_max, conf_min

    def internal_prediction(self, name, config_norm, *args):
        '''
        get predictions
        :param name: str, a objective/cosntraint name
        :param config_norm: Tensor/ndarray(1(batch_size), n_vars), normalized input
        :param args: tuple
        :return:
                value: ndarray/Tensor
        '''
        # transfrom x from the normalized range into the original searching range
        if self.n_obj == 2:
            x1_min, x2_min = self.var_ranges[0, 0], self.var_ranges[1, 0]
            x1_range, x2_range = (self.var_ranges[0, 1] - x1_min), (self.var_ranges[1, 1] - x2_min)
            x_1 = config_norm[:, 0] * x1_range + x1_min
            x_2 = config_norm[:, 1] * x2_range + x2_min
            if name == "obj_1":
                value = 4 * x_1 * x_1 + 4 * x_2 * x_2
            elif name == "obj_2":
                value = (x_1 - 5) * (x_1 - 5) + (x_2 - 5) * (x_2 - 5)
            elif name == "g1":
                value = (x_1 - 5) * (x_1 - 5) + x_2 * x_2 - 25
            elif name == "g2":
                value = (x_1 - 8) * (x_1 - 8) + (x_2 + 3) * (x_2 + 3) - 7.7
            else:
                raise Exception(f"Objective/constraint {name} is not valid for prediction!")
        elif self.n_obj == 3:
            x1_min, x2_min, x3_min = self.var_ranges[0, 0], self.var_ranges[1, 0], self.var_ranges[2, 0]
            x1_range, x2_range, x3_range = (self.var_ranges[0, 1] - x1_min), (self.var_ranges[1, 1] - x2_min), (self.var_ranges[2, 1] - x3_min)
            x_1 = config_norm[:, 0] * x1_range + x1_min
            x_2 = config_norm[:, 1] * x2_range + x2_min
            x_3 = config_norm[:, 2] * x3_range + x3_min
            if th.is_tensor(x_3):
                g = 100 * (1 + (x_3 - 0.5) * (x_3 - 0.5) - th.cos(20 * th.pi * (x_3 - 0.5)))
            else:
                g = 100 * (1 + (x_3 - 0.5) * (x_3 - 0.5) - np.cos(20 * np.pi * (x_3 - 0.5)))
            if name == "obj_3":
                value = 0.5 * x_1 * x_2 * (1 + g)
            elif name == "obj_2":
                value = 0.5 * x_1 * (1 - x_2) * (1 + g)
            elif name == "obj_1":
                value = 0.5 * (1 - x_1) * (1 + g)
            else:
                raise Exception(f"Objective/constraint {name} is not valid for prediction!")

        else:
            raise Exception("Only support 2D and 3D optimization problem right now!")

        return value

class HCF_functions:

    def __init__(self, obj_names: list, const_names: list, var_ranges):
        '''
        initialization
        :param obj_names: list, name of objectives
        :param const_names: list, constraint names
        :param var_ranges: ndarray(n_vars, ), lower and upper var_ranges of variables(non-ENUM), and values of ENUM variables
        '''
        self.hcf = HCF(obj_names, const_names, var_ranges)

    def get_vars_range_for_wl(self, wl_id=None):
        return self.hcf.get_conf_range_for_wl(wl_id)

    def predict_obj1(self, vars, wl_id=None):
        obj = "obj_1"
        if not th.is_tensor(vars):
            value = self.hcf.predict(obj, vars)
        else:
            value = self.hcf.internal_prediction(obj, vars).reshape(-1,1)
        return value

    def predict_obj2(self, vars, wl_id=None):
        obj = "obj_2"
        if not th.is_tensor(vars):
            value = self.hcf.predict(obj, vars)
        else:
            value = self.hcf.internal_prediction(obj, vars).reshape(-1,1)
        return value

    def predict_obj3(self, vars, wl_id=None):
        obj = "obj_3"
        if not th.is_tensor(vars):
            value = self.hcf.predict(obj, vars)
        else:
            value = self.hcf.internal_prediction(obj, vars).reshape(-1,1)
        return value

    # only used for 2D example
    def const_func1(self, vars, wl_id=None):
        const = "g1"
        if not th.is_tensor(vars):
            value = self.hcf.predict(const, vars)
        else:
            value = self.hcf.internal_prediction(const, vars).reshape(-1,1)
        return value

    def const_func2(self, vars, wl_id=None):
        const = "g2"
        if not th.is_tensor(vars):
            value = self.hcf.predict(const, vars)
        else:
            value = self.hcf.internal_prediction(const, vars).reshape(-1,1)
        return value
