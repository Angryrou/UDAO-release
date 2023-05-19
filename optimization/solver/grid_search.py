# Author(s): Qi FAN <qi dot fan at polytechnique dot edu>
#
# Description: Grid-Search
#
# Created at 15/09/2022

import numpy as np
import itertools

from optimization.solver.base_solver import BaseSolver
from utils.parameters import VarTypes

class GridSearch(BaseSolver):
    def __init__(self, gs_params):
        '''
        :param gs_params: dict, the parameters used in grid_search
        '''
        super().__init__()
        self.n_grids_per_var = gs_params["n_grids_per_var"]

    def _get_input(self, var_ranges, var_types):
        '''
        generate grids for each variable
        :param var_ranges: ndarray (n_vars,), the lower and upper var_ranges of non-ENUM variables, and values of ENUM variables
        :param var_types: list, type of each variable
        :return: array, variables (n_grids * n_vars)
        '''

        grids_list = []
        for i, values in enumerate(var_ranges):
            # the number of points generated for each variable
            n_grids_per_var = self.n_grids_per_var[i]

            if (var_types[i] == VarTypes.FLOAT) or (var_types[i] == VarTypes.INTEGER) or (var_types[i] == VarTypes.BOOL):
                upper, lower = values[1], values[0]
                if (lower - upper) > 0:
                    raise Exception(f"ERROR: the lower bound of variable {i} is greater than its upper bound!")

                # make sure the grid point is the same with the type
                # e.g., if int x.min=0, x.max=5, n_grids_per_var=10, ONLY points[0, 1, 2, 3, 4, 5] are feasible
                if var_types[i] == VarTypes.INTEGER or var_types[i] == VarTypes.BOOL:
                    if n_grids_per_var > (upper - lower + 1):
                        n_grids_per_var = int(upper - lower + 1)

                grids_per_var = np.linspace(lower, upper, num=n_grids_per_var, endpoint=True)
            elif var_types[i] == VarTypes.ENUM:
                grids_per_var = values
            else:
                raise Exception(f"Grid-Search solver does not support variable type {var_types[i]}!")

            grids_list.append(grids_per_var)

        ## generate cartesian product of grids_list
        x = np.array([list (i) for i in itertools.product(*grids_list)])

        return x

