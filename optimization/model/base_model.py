# Author(s): Qi FAN <qi dot fan at polytechnique dot edu>
# Author(s): Chenghao LYU <chenghao at cs dot umass dot edu>
#
# Description: APIs for predictive models
#
# Created at 14/10/2022

from abc import ABCMeta, abstractmethod

import numpy as np
import torch as th

import utils.optimization.solver_utils as solver_ut

class BaseModel(object, metaclass=ABCMeta):
    def __init__(self, target_objs):
        self.target_objs = target_objs

    @abstractmethod
    def initialize(self, *args):
        ...

    @abstractmethod
    def fit(self, *args):
        ...

    @abstractmethod
    def normalize_config(self, config):
        ...

    @abstractmethod
    def internal_prediction(self, obj, config_norm, *args):
        ...

    def predict(self, obj, config, *args):
        assert obj in self.target_objs
        if th.is_tensor(config):
            vars_copy = config.clone()
            config_norm = self.normalize_config(vars_copy.data.numpy())
            vars_copy.data = solver_ut._get_tensor(config_norm)
            obj_val = self.internal_prediction(obj, vars_copy, args).view(-1,1)
        elif type(config) is np.ndarray:
            config_norm = self.normalize_config(config)
            obj_val = self.internal_prediction(obj, config_norm, args)
        else:
            raise Exception(f"Configruation type {type(config)} is not supported!")
        # obj_val = self.internal_prediction(obj, config_norm, args)
        return obj_val
