# Author(s): chenghao Lyu <chenghao at cs dot umass dot edu>
#
# Description: TODO
#
# Created at 9/14/22

import torch as th
import torch.optim as optim
import numpy as np
import random
from abc import ABCMeta, abstractmethod


class BaseSolver(object, metaclass=ABCMeta):
    def __init__(self):
        pass

    # @abstractmethod
    # def solve(self, ):
    #     ...