# Author(s): Chenghao Lyu <chenghao at cs dot umass dot edu>
#
# Description: an example of data access
#
# Created at 23/02/2023


import os, pickle
import pandas as pd
import dgl

class PickleUtils(object):

    @staticmethod
    def load(header, file_name):
        path = f"{header}/{file_name}"
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        with open(path, "rb") as f:
            return pickle.load(f)

data_header = "data"
graph_data = PickleUtils.load(data_header, "data/graph_data.pkl")
tabular_data = PickleUtils.load(data_header, "data/tabular_data.pkl")

print(tabular_data.keys())
print(graph_data.keys())
print(tabular_data["COL_MAP"])

