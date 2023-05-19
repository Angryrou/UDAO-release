# Author(s): Qi FAN <qi dot fan at polytechnique dot edu>
#
# Description: An example on running Weighted Sum with Grid-Search solver (with GPR predictive model)
#
# Created at 14/10/2022

from optimization.moo.generic_moo import GenericMOO
from utils.optimization.configs_parser import ConfigsParser
from examples.optimization.heuristic_closed_form.model import HCF_functions
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

# the model is already set up in model_def
model = HCF_functions(obj_names, const_names, var_ranges)

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
            [[4.57532847,2.99736907],
             [4.57532847,2.94835643],
             [4.54873128,2.94835643],
             [4.55488339,2.35777916],
             [4.05108618,2.70649785],
             [4.22210926,2.27386321],
             [4.16037583,2.31353324],
             [4.05656379,2.31353324],
             [4.05656379,2.24382898],
             [4.54056443,1.51406057],
             [4.12394462,1.67427512],
             [4.12394462,1.54217989],
             [4.12394462,1.51634666],
             [3.58406032,1.80857956],
             [3.11403453,1.80857956],
             [3.59858766,1.13044092],
             [3.59858766,1.06023093],
             [3.59858766,0.95998119],
             [3.49265342,0.78595258],
             [2.38298477,1.75014612],
             [3.09184498,0.75151902],
             [2.91139354,0.88179067],
             [2.82988969,0.93554913],
             [2.82988969,0.81267801],
             [2.84211178,0.41017467],
             [2.75105005,0.41017467],
             [2.1028579, 0.95839454],
             [2.1028579, 0.77675025],
             [2.0367216, 0.77675025],
             [1.98890908,0.77675025],
             [1.90773962,0.4527134 ]]
        ), 5)).all()
        print("Test successfully!")
    elif len(obj_names) == 3:
        assert (np.round(po_vars, 5) == np.round(np.array(
            [[0.26791931,0.61499972,0.50280342],
             [0.25118901,0.91492858,0.40493414],
             [0.26294469,0.87027311,0.40493414],
             [0.26294469,0.90674546,0.40493414],
             [0.26294469,0.92482828,0.40493414],
             [0.28527238,0.90674546,0.40493414],
             [0.33391268,0.88321331,0.40493414],
             [0.43418336,0.90972099,0.50468686],
             [0.56767221,0.90811384,0.50468686],
             [0.56767221,0.90814917,0.50468686],
             [0.56767221,0.90975727,0.50468686],
             [0.5704818 ,0.46957565,0.50468686],
             [0.5793541 ,0.36562543,0.50468686],
             [0.5793541 ,0.4039538 ,0.50468686],
             [0.5793541 ,0.51262817,0.50468686],
             [0.58338204,0.90811289,0.50468686],
             [0.58397347,0.90811289,0.50468686],
             [0.58397347,0.90972099,0.50468686],
             [0.58398355,0.90972099,0.50468686]]
        ), 5)).all()
        print("Test successfully!")
    else:
        Exception(f"{len(obj_names)} objectives are not supported in the code repository for now!")

    # save data
    data_path = f"./examples/optimization/heuristic_closed_form/data/{moo_algo}/{po_objs.shape[1]}d/"
    results = np.hstack([po_objs, po_vars])
    moo_ut.save_results(data_path, results, wl_id, mode="data")
    moo_ut.save_results(data_path, [time_cost], wl_id, mode="time")

    if po_objs is not None:
        moo_ut.plot_po(po_objs, n_obj=po_objs.shape[1], title="evo_hcf")
