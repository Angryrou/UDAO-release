# Copyright (c) 2020 École Polytechnique
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
# Author(s): Qi FAN <qi dot fan at polytechnique dot edu>
#
# Description: Performance metric of MOO
#
# Created at 06/03/2023
import torch as th
import numpy as np
from botorch.utils.multi_objective.hypervolume import Hypervolume

def dominated_space(pareto_set, ref_point):
    hv = Hypervolume(ref_point=ref_point)
    volume = hv.compute(pareto_set)

    return volume

def perc_dominated_space(pareto_set, ref_point, utopia_point):
    # for 2D minimization
    init_volume = abs(ref_point[1] - utopia_point[1]) * abs(ref_point[0] - utopia_point[0])
    dominated_volume = dominated_space(-1 * th.from_numpy(pareto_set), -1 * th.from_numpy(ref_point))

    return 100 * (dominated_volume / init_volume)

if __name__ == '__main__':
    pareto_set = np.array([[1.0, 10.0], [4.0, 8.0], [6.0, 2.0]])
    ref_point = np.array([10.0, 10.0])
    # why times -1? --> by default, BoTorch solves a maximization problem.
    # The following example is based on the Pareto set of minimization optimization problem, i.e. minimize all objectives
    dominated_volume = dominated_space(-1 * th.from_numpy(pareto_set), -1 * th.from_numpy(ref_point))
    print(dominated_volume)
