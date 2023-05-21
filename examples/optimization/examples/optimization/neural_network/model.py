# Author(s): Qi FAN <qi dot fan at polytechnique dot edu>
#
# Description: An example of Neural Network (NN) model
#              and pre-define functions (objectives and constraints) based on NN
#
# Created at 17/10/2022

from utils.optimization.configs_parser import ConfigsParser
from optimization.model.base_model import BaseModel
import utils.optimization.solver_utils as solver_ut

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F

# an example by following: https://machinelearningmastery.com/pytorch-tutorial-develop-deep-learning-models/
class NN(BaseModel):
    def __init__(self, obj_names: list, const_names: list, training_vars: np.ndarray, var_ranges):
        super().__init__(obj_names + const_names)
        # th.manual_seed(0)
        self.initialize(training_vars, var_ranges)

    def initialize(self, training_vars, var_ranges):
        self.n_input = var_ranges.shape[0]
        self.n_out = 1
        self.n_hid = 128
        self.n_epoch = 20
        self.learning_rate = 0.01

        self.var_ranges = var_ranges
        self.n_training = training_vars.shape[0]

        self.model_params = ConfigsParser().parse_details(option="model")
        self.in_features = self.model_params["in_features"]

        norm_training_vars = self.normalize_config(training_vars)
        self.y_dict = self.get_y_dict(norm_training_vars)
        self.model_dict = self.fit(norm_training_vars)

    def fit(self, X_train):
        if not th.is_tensor(X_train):
            X_train = solver_ut._get_tensor(X_train)
        model = nn.Sequential(nn.Linear(self.n_input, self.n_hid),
                              nn.Sigmoid(),
                              nn.Linear(self.n_hid, self.n_out),
                              nn.Sigmoid()
                              )
        loss_function = nn.MSELoss()
        optimizer = th.optim.Adam(model.parameters(), lr=self.learning_rate)
        losses = []
        model_dict = {}
        for obj in self.target_objs:
            for epoch in range(self.n_epoch):
                # get forward prediction
                pred_y = model(X_train)
                loss = loss_function(pred_y, solver_ut._get_tensor(self.y_dict[obj]))
                losses.append(loss.item())
                optimizer.zero_grad()
                loss.backward()
                # for name, param in model.named_parameters():
                #     print(name, param.grad)
                optimizer.step()
            model_dict[obj] = model
        return model_dict

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
            y_dict[name] = value
        return y_dict

    def get_objs(self, vars, name):
        # transfrom normalized x into the original space
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

        return value

    def internal_prediction(self, name, X_test, *args):
        return_numpy_flag = False
        if not th.is_tensor(X_test):
            X_test = solver_ut._get_tensor(X_test)
            return_numpy_flag = True
        if isinstance(args[0], tuple):
            mode = args[0][0]
        elif isinstance(args[0], str):
            mode = args[0]
        else:
            raise Exception("Unexpected format of args!")

        if mode == "nn":
            yhat = self.model_dict[name](X_test)
        elif mode == "hcf":
            yhat = self.get_objs(X_test, name).reshape(-1,1)
        else:
            raise Exception(f"Model {mode} is not available!")

        if return_numpy_flag:
            yhat = yhat.data.numpy().squeeze()
        return yhat

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

class NNPredictiveModels:
    def __init__(self, obj_names: list, const_names: list, training_vars: np.ndarray, var_ranges):
        '''
        initialization
        :param objs: list, name of objectives
        :param training_vars: ndarray(n_samples, n_vars), input used to train NN model.
        '''
        self.training_vars = training_vars
        self.nn = NN(obj_names, const_names, training_vars, var_ranges)

    def get_vars_range_for_wl(self, wl_id=None):
        return self.nn.get_conf_range_for_wl(wl_id)

    def predict_obj1(self, vars, wl_id=None):
        obj = "obj_1"
        mode = "hcf"
        if not th.is_tensor(vars):
            value = self.nn.predict(obj, vars, mode)
        else:
            value = self.nn.internal_prediction(obj, vars, mode)
        return value

    def predict_obj2(self, vars, wl_id=None):
        obj = "obj_2"
        mode = "hcf"
        if not th.is_tensor(vars):
            value = self.nn.predict(obj, vars, mode)
        else:
            value = self.nn.internal_prediction(obj, vars, mode)
        return value

    # only used for 2D example
    def const_func1(self, vars, wl_id=None):
        const = "g1"
        mode = "hcf"
        if not th.is_tensor(vars):
            value = self.nn.predict(const, vars, mode)
        else:
            value = self.nn.internal_prediction(const, vars, mode)
        return value

    def const_func2(self, vars, wl_id=None):
        const = "g2"
        mode = "hcf"
        if not th.is_tensor(vars):
            value = self.nn.predict(const, vars, mode)
        else:
            value = self.nn.internal_prediction(const, vars, mode)
        return value
