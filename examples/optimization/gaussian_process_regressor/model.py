# Author(s): Qi FAN <qi dot fan at polytechnique dot edu>
#            Chenghao Lyu <chenghao at cs dot umass dot edu>
#
# Description: An example of Gaussian Process Regressor (GPR) model (reuse code in ICDE paper)
#              and pre-define functions (objectives and constraints) based on GPR
#
# Created at 17/10/2022

from utils.optimization.configs_parser import ConfigsParser
import utils.optimization.solver_utils as solver_ut
from optimization.model.base_model import BaseModel

import numpy as np
import torch as th

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

class GPR(BaseModel):
    def __init__(self, obj_names: list, const_names: list, training_vars: np.ndarray, var_ranges: list):
        '''
        :param objs: list, name of objectives
        :param const_names: list, constraint names
        :param training_vars: ndarray(n_training_samples, n_vars), input to train GPR models
        '''
        super().__init__(obj_names + const_names)
        self.n_objs = len(obj_names)
        self.initialize(training_vars, var_ranges)

    def initialize(self, training_vars, var_ranges):
        '''
        initialize parameters and fit GPR models for objectives separately
        :param objs: list, name of objectives
        :param training_vars: ndarray(n_training_samples, n_vars), input to train GPR models
        :return:
        '''
        self.var_ranges = var_ranges

        self.model_params = ConfigsParser().parse_details(option="model")
        self.n_training = training_vars.shape[0]
        self.magnitude = self.model_params["magnitude"]
        self.ridge = self.model_params["ridge"]
        self.length_scale = self.model_params["length_scale"]
        self.device, self.dtype = th.device('cpu'), th.float32

        norm_training_vars = self.normalize_config(training_vars)
        self.models = self.fit(norm_training_vars)

    def gs_dist(self, x1, x2):
        '''
        calculate distances between each training sample
        :param x1: Tensor(n_training_samples, n_vars), training input java_data
        :param x2: Tensor(n_training_samples or n_x, n_vars), training input java_data or test input java_data
        :return:
                dist: Tensor(n_training_samples, n_training_samples), each element shows distance between each training sample
        '''
        # e.g.
        # x1.shape = (m, 12)
        # x2.shape = (n, 12)
        # K(x1, x2).shape = (m, n)
        assert x1.shape[1] == x2.shape[1]
        comm_dim = x1.shape[1]
        dist = th.norm(x1.reshape(-1, 1, comm_dim) - x2.reshape(1, -1, comm_dim), dim=2)
        return dist

    def get_y_dict(self, vars):
        '''
        get training java_data of true objective values with example functions
        :param vars: ndarray(n_training_samples, n_vars), input to train GPR models
        :return: y_dict: dict, key is objective names, values are training java_data of true objective values
        '''
        y_dict = {}
        # objective functions are the same as that in HCF
        for name in self.target_objs:
            # transfrom x in the original space as (
            if self.n_objs == 2:
                x1_min, x2_min = self.var_ranges[0, 0], self.var_ranges[1, 0]
                x1_range, x2_range = (self.var_ranges[0, 1] - x1_min), (self.var_ranges[1, 1] - x2_min)
                x_1 = vars[:, 0] * x1_range + x1_min
                x_2 = vars[:, 1] * x2_range + x2_min
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
            elif self.n_objs == 3:
                x1_min, x2_min, x3_min = self.var_ranges[0, 0], self.var_ranges[1, 0], self.var_ranges[2, 0]
                x1_range, x2_range, x3_range = (self.var_ranges[0, 1] - x1_min), (self.var_ranges[1, 1] - x2_min), (
                        self.var_ranges[2, 1] - x3_min)
                x_1 = vars[:, 0] * x1_range + x1_min
                x_2 = vars[:, 1] * x2_range + x2_min
                x_3 = vars[:, 2] * x3_range + x3_min
                if th.is_tensor(x_3):
                    g = 100 * (1 + (x_3 - 0.5) * (x_3 - 0.5) - th.cos(20 * th.pi * (x_3 - 0.5)))
                else:
                    g = 100 * (1 + (x_3 - 0.5) * (x_3 - 0.5) - np.cos(20 * np.pi * (x_3 - 0.5)))
                if name == "obj_1":
                    value = 0.5 * x_1 * x_2 * (1 + g)
                elif name == "obj_2":
                    value = 0.5 * x_1 * (1 - x_2) * (1 + g)
                elif name == "obj_3":
                    value = 0.5 * (1 - x_1) * (1 + g)
                else:
                    raise Exception(f"Objective/constraint {name} is not valid for prediction!")

            else:
                raise Exception("Only support 2D and 3D optimization problem right now!")
            y_dict[name] = value
        return y_dict

    def fit(self, X_train):
        '''
        fit GPR models
        :param X_train: ndarray(n_training_samples, n_vars), input to train GPR models, should be normalized
        :return:
                X_train: Tensor(n_training_samples, n_vars), input to train GPR models
                y_tensor_dict: dict, key is objective names, values are training java_data of true objective values with Tensor(n_training_samples,) format
                K_inv: Tensor(n_training_samples, n_training_samples), inversion of the covariance matrix
        '''
        y_dict = self.get_y_dict(X_train)

        if X_train.ndim != 2:
            raise Exception("x_train should have 2 dimensions! X_dim:{}"
                            .format(X_train.ndim))
        X_train = self._get_tensor(X_train)

        sample_size = self.n_training
        if np.isscalar(self.ridge):
            ridge = np.ones(sample_size) * self.ridge
        else:
            raise Exception(f"rideg is not set properly!")

        assert isinstance(ridge, np.ndarray)
        assert ridge.ndim == 1

        ridge = self._get_tensor(ridge)
        K = self.magnitude * th.exp(-self.gs_dist(X_train, X_train) / self.length_scale) + th.diag(ridge)
        K_inv = K.inverse()

        y_tensor_dict = {}
        for obj in self.target_objs:
            y_tensor_dict[obj] = self._get_tensor(y_dict[obj])
        return [X_train, y_tensor_dict, K_inv]

    def get_kernel(self, x1, x2):
        '''
        get kernel
        :param x1: Tensor(n_training_samples, n_vars), training java_data
        :param x2: Tensor(n_x, n_vars),
        :return:
        '''
        return self.magnitude * th.exp(-self.gs_dist(x1, x2) / self.length_scale)

    def objective(self, X_test, X_train, y_train, K_inv):
        '''
        call GPR model to get objective values
        :param X_test: Tensor(n_x, n_vars), input of the predictive model, where n_x shows the number of input variables
        :param X_train: Tensor(n_training_samples, n_vars), input to train GPR models
        :param y_train: Tensor(n_training_samples,), training java_data of true objective values
        :param K_inv: Tensor(n_training_samples, n_training_samples), inversion of the covariance matrix
        :return:
                yhat: Tensor(n_x,), the prediction of the objective
        '''
        if X_test.ndimension() == 1:
            X_test = X_test.reshape(1, -1)
        length_scale = self.length_scale
        K2 = self.magnitude * th.exp(-self.gs_dist(X_train, X_test) / length_scale)
        K2_trans = K2.t()
        yhat = th.matmul(K2_trans, th.matmul(K_inv, y_train))
        return yhat

    def internal_prediction(self, name, X_test, *args):
        '''
        get prediction of the objective based on the GPR predictive model
        :param obj: str, name of the objective
        :param X_test: ndarray(n_x, n_vars), normalized value, input of the predictive model, where n_x shows the number of input variables
        :return:
                yhat: Tensor(n_x,), the prediction of the objective
        '''
        return_numpy_flag = False
        if not th.is_tensor(X_test):
            X_test = solver_ut._get_tensor(X_test)
            return_numpy_flag = True
        X_train, y_dict, K_inv = self.models
        y_train = y_dict[name]
        yhat = self.objective(X_test, X_train, y_train, K_inv).view(-1, 1)
        if return_numpy_flag:
            yhat = yhat.data.numpy().squeeze()
        return yhat

    def _get_tensor(self, x, dtype=None, device=None):
        if dtype is None:
            dtype = self.dtype
        if device is None:
            device = self.device
        return th.tensor(x, dtype=dtype, device=device)

    @staticmethod
    def check_output(X):
        finite_els = np.isfinite(X)
        if not np.all(finite_els):
            raise Exception("Input contains non-finite values: {}"
                            .format(X[~finite_els]))

    def get_conf_range_for_wl(self, wl_id=None):
        conf_max = self.var_ranges[:, 1]
        conf_min = self.var_ranges[:, 0]
        return conf_max, conf_min

    def normalize_config(self, config):
        """
        :param *args:
        :param real_conf: numpy.array[int]
        :return: normalized to 0-1
        """
        var_min, var_max = self.get_conf_range_for_wl()
        normalized_conf = (config - var_min) / (var_max - var_min)
        return normalized_conf

class GPRPredictiveModels:

    def __init__(self, obj_names: list, const_names: list, training_vars: np.ndarray, var_ranges: list):
        '''
        initialization
        :param obj_names: list, name of objectives
        :param const_names: list, constraint names
        :param training_vars: ndarray(n_samples, n_vars), input used to train GPR model.
        '''
        self.training_vars = training_vars
        self.gpr = GPR(obj_names, const_names, training_vars, var_ranges)

    def get_vars_range_for_wl(self, wl_id=None):
        return self.gpr.get_conf_range_for_wl(wl_id)

    def predict_obj1(self, vars, wl_id=None):
        obj = "obj_1"
        if not th.is_tensor(vars):
            value = self.gpr.predict(obj, vars)
        else:
            value = self.gpr.internal_prediction(obj, vars)
        return value

    def predict_obj2(self, vars, wl_id=None):
        obj = "obj_2"
        if not th.is_tensor(vars):
            value = self.gpr.predict(obj, vars)
        else:
            value = self.gpr.internal_prediction(obj, vars)
        return value

    def predict_obj3(self, vars, wl_id=None):
        obj = "obj_3"
        if not th.is_tensor(vars):
            value = self.gpr.predict(obj, vars)
        else:
            value = self.gpr.internal_prediction(obj, vars)
        return value

    # only used for 2D example
    def const_func1(self, vars, wl_id=None):
        const = "g1"
        if not th.is_tensor(vars):
            value = self.gpr.predict(const, vars)
        else:
            value = self.gpr.internal_prediction(const, vars)
        return value

    def const_func2(self, vars, wl_id=None):
        const = "g2"
        if not th.is_tensor(vars):
            value = self.gpr.predict(const, vars)
        else:
            value = self.gpr.internal_prediction(const, vars)
        return value