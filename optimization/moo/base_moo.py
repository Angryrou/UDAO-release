# Author(s): Qi FAN <qi dot fan at polytechnique dot edu>
#
# Description: based methods in MOO
#
# Created at 15/09/2022

from abc import ABCMeta, abstractmethod

class BaseMOO(object, metaclass=ABCMeta):
    def __init__(self, ):
        pass

    @abstractmethod
    def solve(self, *args):
        ...