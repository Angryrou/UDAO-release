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
# Created at 28/09/2022
import pickle
import torch as th
import numpy as np
import json

# DEFAULT_DEVICE = th.device("cpu")
DEFAULT_DEVICE = th.device('cuda') if th.cuda.is_available() else th.device("cpu")
DEFAULT_DTYPE = th.float32

def load_pkl(pkl_path):
    with open(pkl_path, 'rb') as f:
        return pickle.load(f)

def get_bounded(k, lower=0.0, upper=1.0):
    k = np.maximum(k, lower)
    k = np.minimum(k, upper)
    return k

def _get_tensor(x, dtype=None, device=None, requires_grad=False):
    dtype = DEFAULT_DTYPE
    device = DEFAULT_DEVICE if device is None else device

    return th.tensor(x, dtype=dtype, device=device, requires_grad=requires_grad)

def json_parse_predict(zmesg, knob_list):
    """{"s3":10,"s4":200,"k1":32,"k2":4,"k3":4,"k4":8,"k5":48,"k6":200,"k7":1,"k8":0.6,"Objective":"latency","JobID":"13-4","s1":10000,"s2":128}"""
    FORMAT_ERROR = 'ERROR: format of zmesg is not correct'
    try:
        x = json.loads(zmesg)
        wl_id = x['JobID']
        obj = x['Objective']
        conf_raw_val = np.array([float(x[k]) for k in knob_list])
        return wl_id, conf_raw_val, obj
    except:
        print(FORMAT_ERROR + f'{zmesg}')
        return None, None, None
