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
            [[1.36184301,1.56261991],
             [1.36184301,1.737269  ],
             [1.40918922,1.737269  ],
             [1.83173193,1.77975913],
             [1.94960149,1.80834236],
             [1.62206276,2.16873937],
             [2.20235736,1.77319633],
             [2.38298477,1.75014612],
             [2.38298477,1.84214029],
             [2.38298477,1.89763836],
             [2.38298477,2.35104292],
             [2.85654981,2.10754016],
             [3.2267594 ,2.12070031],
             [3.2267594 ,2.18792114],
             [3.20208243,2.41697009],
             [3.46884362,2.34176778],
             [3.46884362,2.34816981],
             [3.46884362,2.35935043],
             [4.0513116 ,2.12070031],
             [4.05108618,2.48638148],
             [4.05108618,2.70649785],
             [4.12857853,2.86768821],
             [4.16037583,2.94835643],
             [4.54873128,2.94835643],
             [4.57532847,2.94835643],
             [4.57532847,2.99736907],
             [4.57532847,2.99926873],
             [4.91970208,2.9993028 ]]
        ), 5)).all()
        print("Test successfully!")
    elif len(obj_names) == 3:
        assert (np.round(po_vars, 5) == np.round(np.array(
            [[0.53164114,0.8530219 ,0.49973104],
             [0.46760011,0.86701319,0.49973104],
             [0.45319151,0.78159305,0.49973104],
             [0.42268453,0.8530219 ,0.49973104],
             [0.42216693,0.64641452,0.49973104],
             [0.40645037,0.86701319,0.49973104],
             [0.40625809,0.55944023,0.49973104],
             [0.31417183,0.75710187,0.49973104],
             [0.30283298,0.8530219 ,0.49973104],
             [0.28261011,0.86701319,0.49973104],
             [0.27917286,0.8530219 ,0.49973104],
             [0.27917286,0.563307  ,0.49973104],
             [0.27843304,0.91049786,0.49973104],
             [0.2640509 ,0.8530219 ,0.49973104],
             [0.2640509 ,0.75710187,0.49973104],
             [0.2640509 ,0.75128677,0.49973104],
             [0.23762586,0.563307  ,0.49973104],
             [0.15488331,0.563307  ,0.49973104]]
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
