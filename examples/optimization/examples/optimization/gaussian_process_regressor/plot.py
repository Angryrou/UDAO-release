# Copyright (c) 2020 École Polytechnique
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
# Author(s): Qi FAN <qi dot fan at polytechnique dot edu>
#
# Description: TODO
#
# Created at 25/11/2022
import numpy as np
import matplotlib.pyplot as plt

def read_data(file, n_obj):
    data = np.loadtxt(file, dtype='float32')
    if n_obj == 2:
        po = data[:,0:2]
    elif n_obj == 3:
        po = data[:, 0:3]
    else:
        raise Exception(f"{n_obj} objectives are not supported for now!")

    return po

def read_time(file):
    time_cost = np.loadtxt(file, dtype="float32").item()
    return time_cost

def get_data_file(root_path, algo, mode="2d"):
    if algo == "pf_ap":
        data_file = root_path + f"progressive_frontier/{mode}/mogd/jobId_None/" + "data.txt"
    elif algo == "ws_grid":
        data_file = root_path + f"weighted_sum/{mode}/grid_search/jobId_None/"  + "data.txt"
    elif algo == "ws_random":
        data_file = root_path + f"weighted_sum/{mode}/random_sampler/jobId_None/"  + "data.txt"
    elif algo == "evo":
        data_file = root_path + f"evolutionary/{mode}/jobId_None/" + "data.txt"
    else:
        raise Exception(f"Algorithm {algo} is not supported for now!")

    return data_file

if __name__ == '__main__':

    algos = ["pf_ap", "ws_grid", "ws_random", "evo"]
    root_path = "examples/optimization/gaussian_process_regressor/data/"

    fig, ax = plt.subplots()
    fig1, axs = plt.subplots(2, 2)
    for i, algo in enumerate(algos):
        data_file = get_data_file(root_path, algo, mode="2d")
        po = read_data(data_file, n_obj=2)

        ax.plot(po[:, 0], po[:, 1], marker='o', label=algo)
        axs[int(i / 2), i % 2].plot(po[:, 0], po[:, 1], marker='o')
        axs[int(i / 2), i % 2].set_xlabel('Obj_1')
        axs[int(i / 2), i % 2].set_ylabel('Obj_2')
        axs[int(i / 2), i % 2].set_title(f'{algo}')

    ax.set_xlabel('Obj_1')
    ax.set_ylabel('Obj_2')
    ax.set_title('Pareto frontiers of MOO algorithms')
    ax.legend()
    plt.tight_layout()
    plt.show()

