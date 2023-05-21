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
        if solver == "grid_search":
            assert (np.round(po_vars, 5) == np.round(np.array(
                [[5.        ,3.        ],
                 [4.29292929,3.        ],
                 [3.58585859,3.        ],
                 [2.97979798,2.96969697],
                 [2.42424242,2.39393939],
                 [1.86868687,1.84848485],
                 [1.36363636,1.36363636],
                 [0.85858586,0.87878788],
                 [0.4040404 ,0.42424242],
                 [0.        ,0.        ]]
            ), 5)).all()
        elif solver == "random_sampler":
            assert (np.round(po_vars, 5) == np.round(np.array(
                [[4.74506152e+00,2.99018859e+00],
                 [4.41689202e+00,2.99944204e+00],
                 [3.59118003e+00,2.99509568e+00],
                 [2.95615973e+00,2.97142232e+00],
                 [2.38400682e+00,2.37109084e+00],
                 [1.84686945e+00,1.84353491e+00],
                 [1.33263226e+00,1.31964954e+00],
                 [8.85083255e-01,8.52108131e-01],
                 [4.58344504e-01,4.24590966e-01],
                 [1.83671876e-03,1.56753142e-02]]
            ), 5)).all()
        else:
            raise Exception(f"Solver {solver} is not available!")
        print("Test successfully!")

    elif len(obj_names) == 3:
        if solver == "grid_search":
            assert (np.round(po_vars, 5) == np.round(np.array(
                [[0.        ,0.        ,0.        ],
                 [1.        ,0.        ,0.        ],
                 [1.        ,1.        ,0.        ],
                 [1.        ,0.88888889,0.        ],
                 [1.        ,0.11111111,0.        ],
                 [1.        ,0.66666667,0.        ]]
            ), 5)).all()
        elif solver == "random_sampler":
            assert (np.round(po_vars, 5) == np.round(np.array(
                [[5.45964897e-04,3.40604644e-01,4.97363117e-01],
                 [6.10785371e-02,9.98526578e-01,4.00510464e-01],
                 [9.76459465e-01,6.64218590e-04,4.08677397e-01],
                 [1.00014061e-01,5.15433087e-01,4.99461489e-01],
                 [6.95625446e-01,6.51214685e-02,5.01591471e-01],
                 [1.81150962e-01,3.05046698e-01,5.00537459e-01],
                 [3.41698115e-01,7.89869503e-01,5.00692099e-01],
                 [9.99808578e-01,4.78738524e-03,9.36607012e-02],
                 [5.02720761e-01,5.46868179e-01,5.00677021e-01],
                 [6.98248478e-01,6.62856479e-01,4.98639353e-01],
                 [9.94400790e-01,7.12892303e-01,4.02908445e-01]]
            ), 5)).all()
        else:
            raise Exception(f"Solver {solver} is not available!")
        print("Test successfully!")

    else:
        Exception(f"{len(obj_names)} objectives are not supported in the code repository for now!")

    # save data
    data_path = f"./examples/optimization/heuristic_closed_form/data/{moo_algo}/{po_objs.shape[1]}d/{solver}/"
    results = np.hstack([po_objs, po_vars])
    moo_ut.save_results(data_path, results, wl_id, mode="data")
    moo_ut.save_results(data_path, [time_cost], wl_id, mode="time")

    if po_objs is not None:
        moo_ut.plot_po(po_objs, n_obj=po_objs.shape[1], title=f"ws_hcf_{solver}")
