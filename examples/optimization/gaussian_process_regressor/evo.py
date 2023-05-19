# Author(s): Qi FAN <qi dot fan at polytechnique dot edu>
#
# Description: An example on running Weighted Sum with Grid-Search solver (with GPR predictive model)
#
# Created at 14/10/2022

from optimization.moo.generic_moo import GenericMOO
from utils.optimization.configs_parser import ConfigsParser
from examples.optimization.gaussian_process_regressor.model import GPRPredictiveModels
import utils.optimization.moo_utils as moo_ut

import numpy as np
import time

HELP = """
Format: python ws.py -c <config> -h
    - c : The configuration file location. Default is "examples/optimization/gaussian_process_regressor/configs/ws_grid_search.json"
Example:
    python examples/optimization/gaussian_process_regressor/ws.py -c examples/optimization/gaussian_process_regressor/configs/ws_grid_search.json
"""

# get input parameters
moo_algo, solver, var_types, var_ranges, obj_names, opt_types, obj_types, const_types, const_names, add_params = ConfigsParser().parse_details()

# model set up
data_file = "examples/optimization/gaussian_process_regressor/training_data/jobId_None/data.txt"
training_vars = np.loadtxt(data_file, dtype='float32')[:,len(obj_names):]
model = GPRPredictiveModels(obj_names,  const_names, training_vars, var_ranges)

# problem setup
moo = GenericMOO()

if len(obj_names) == 2:
    moo.problem_setup(obj_names=obj_names, obj_funcs=[model.predict_obj1, model.predict_obj2], opt_types=opt_types,
                  const_funcs=[model.const_func1, model.const_func2], const_types=const_types, var_types=var_types, var_ranges=var_ranges)
elif len(obj_names) == 3:
    moo.problem_setup(obj_names=obj_names, obj_funcs=[model.predict_obj1, model.predict_obj2, model.predict_obj3], opt_types=opt_types, obj_types=obj_types,
                  const_funcs=[], const_types=const_types, var_types=var_types, var_ranges=var_ranges, wl_ranges=model.get_vars_range_for_wl)
else:
    raise Exception(f"{len(obj_names)} objectives are not supported in the code repository for now!")

# solve MOO problem
po_objs_list, po_vars_list, jobIds, time_cost_list = moo.solve(moo_algo, solver, add_params)

for i, wl_id in enumerate(jobIds):
    po_objs, po_vars = po_objs_list[i], po_vars_list[i]
    time_cost = time_cost_list[i]
    print(f"Pareto solutions of wl_{wl_id}:")
    print(po_objs)
    print(f"Variables of wl_{wl_id}:")
    print(po_vars)
    print(f"Time cost of wl_{wl_id}:")
    print(time_cost)

    if len(obj_names) == 2:
        assert (np.round(po_vars, 5) == np.round(np.array(
            [[4.86125351,2.99232979],
             [4.65495805,2.94835643],
             [4.54873128,2.94835643],
             [4.16037583,2.98494115],
             [3.83646243,2.98926236],
             [3.6219659 ,2.85009242],
             [3.28134872,2.93865969],
             [2.35304965,2.94813434],
             [2.07532639,2.94715071],
             [2.01459782,2.94813434],
             [1.84258511,2.80206248],
             [1.13330739,2.98520515],
             [1.02382591,2.62601842],
             [0.89069191,2.60259714],
             [0.89069191,2.45616934],
             [0.89069191,2.35933464],
             [0.78968403,2.25705007],
             [2.1028579 ,0.95839454],
             [2.0771008 ,0.95839454],
             [2.03268929,0.92264935],
             [2.1028579 ,0.77675025],
             [1.46115117,0.90175235],
             [1.85312931,0.67520838],
             [1.186862  ,0.93618718],
             [0.84843166,0.94826915],
             [0.89069191,0.58464728],
             [0.89069191,0.53655087],
             [0.78158249,0.40276212]]), 5)).all()
        print("Test successfully!")
        # save data
        data_path = f"./examples/optimization/gaussian_process_regressor/data/{moo_algo}/{po_objs.shape[1]}d/"
        results = np.hstack([po_objs, po_vars])
        moo_ut.save_results(data_path, results, wl_id, mode="data")
        moo_ut.save_results(data_path, [time_cost], wl_id, mode="time")

        if po_objs is not None:
            moo_ut.plot_po(po_objs, n_obj=po_objs.shape[1], title="evo_gpr")
