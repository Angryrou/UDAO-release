# Spark-TPCH-100-Traces

We introduce our pre-processed ~100K Spark-TPCH traces that collected over a 6-node Spark Cluster. 
The trace is motivated by a task of modeling the *latency* of running a SparkSQL with a specific Spark configuration.

## Trace Collection System Setup

From the hardware perspective, the Spark cluster has 6 nodes connected with 100Gbit bit network cards, 
with 1 master node and 5 worker nodes; each node has 
  - 2x Intel(R) Xeon(R) Gold 6130 CPU @ 2.1GHz 
  - 32 cores 
  - 754G memory

From the software perspective, each node installs
- CentOS Linux 7 (core) as the operating system
- Spark 3.2.1 
- Hadoop 3.3.0 (with Yarn-client as the cluster manager)
- Hive 3.1.3 (with HiveMetastore for Spark)

For trace collection, we
  - generated ~100 queries from the 22 templates
  - generated ~100 Spark configurations over 12 selected important Spark parameters 
  - turned off Adaptive Query Execution (AQE) 

## Quick Access to the Processed Data

### Python Dependencies
- python 3.9, 
- pip install pandas, dgl, numpy

### Code example
```python
# check examples/traces/spark-tpch-100-traces/main.py
data_header = "examples/traces/spark-tpch-100-traces/data"
graph_data = PickleUtils.load(data_header, "graph_data.pkl")
tabular_data = PickleUtils.load(data_header, "tabular_data.pkl")

# sample output
print(graph_data.keys())
# dict_keys(['all_ops', 'dgl_dict'])

print(tabular_data.keys())
# dict_keys(['ALL_COLS', 'COL_MAP', 'df'])

print(tabular_data["COL_MAP"])
#  {'META_COLS': ['id', 'q_sign', 'template', 'start_timestamp', 'latency'],
#  'CH1_FEATS': ['dgl_id'],
#  'CH2_FEATS': ['input_mb', 'input_records', 'input_mb_log', 'input_records_log'],
#  'CH3_FEATS': ['m1', 'm2', 'm3', 'm4', 'm5', 'm6', 'm7', 'm8'],
#  'CH4_FEATS': ['k1', 'k2', 'k3', 'k4', 'k5', 'k6', 'k7', 'k8', 's1', 's2', 's3', 's4'],
#  'OBJS': ['latency']}
```

## Trace Description

The trace include the meta information and four channels of different input source. 
1. Query meta information:
   - `id`: the spark application id
   - `q_sign`: the indicator of a query
   - `template`: the template that generate the query 
   - `start_timestamp`: the timestamp that the query starts to run
   - `latency`: the **target objective** we want to predict.
2. (CH1) The physical query plan represented as a DAG of query operators, each operator has a feature 
  - `op_id`: identify a specific query operator (we have 13 different physical operators in this dataset)
3. (CH2) Input meta information that involves the size of input data when running a query
  - `input_mb`: the size of the input data, unit in MB
  - `input_records`: the number of records of the input data
  - `input_mb_log`: log value of `input_mb`
  - `input_records_log`: log value of `input_records`
4. (CH3) Machine system states (normalized)
  - `m1,m2,...,m8`: 8 normalized machine system states, including "cpu_utils", "mem_utils", "disk_busy", "disk_bsize/s",
               "disk_KB/s", "disk_xfers/s", "net_KB/s", "net_xfers/s"
5. (CH4) Spark Configuration knobs (converted as numerical values)
  - `k1-k8` and `s1-s4`: 12 Spark knobs


The traces are maintained by Tabular data and Graph data.

### Graph Data
Our dataset involves 42 different physical query plans that is maintained as `dgl.Graph`s (check at `graph_data["dgl_dict"]`).
Each node in a `dgl.Graph` has one attribute named `op_id` as the id of a query operator. 

The full set of the query operators are in `graph_data["all_ops"]`.

### Tabular Data
1. Tabular data is stored in a `pandas.DataFrame` in `tabular_data["df"]`. It is currently sorted by the `start_timestamp`. It includes 33 columns in total.
2. Check the total columns in `tabular_data["ALL_COLS"]`
2. Check the column names of each type of features in `tabular_data["COL_MAP"]`
3. The column name of `CH1`, is `dgl_id`, which indicates a `dgl.Graph` in `graph_data["dgl_dict"]`
