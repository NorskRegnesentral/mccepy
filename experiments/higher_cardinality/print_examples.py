import warnings
warnings.filterwarnings('ignore')

import pandas as pd
pd.set_option('display.max_columns', None)

import os
import argparse

parser = argparse.ArgumentParser(description="Fit MCCE with various datasets.")
parser.add_argument(
    "-p",
    "--path",
    required=True,
    help="Path where results are saved",
)
parser.add_argument(
    "-d",
    "--dataset",
    default='adult',
    help="Datasets for experiment",
)
parser.add_argument(
    "-n",
    "--number_of_samples",
    type=int,
    default=100,
    help="Number of instances per dataset",
)
parser.add_argument(
    "-K",
    "--K",
    type=int,
    default=10000,
    help="Number generated counterfactuals per test observation",
)
parser.add_argument(
    "-ft",
    "--force_train",
    action='store_true',  # default is False
    help="Whether to train the prediction model from scratch or not. Default will not train.",
)

args = parser.parse_args()

path = args.path
data_name = args.dataset
n_test = args.number_of_samples
K = args.K
force_train = args.force_train

results_inverse = pd.read_csv(os.path.join(path, f"adult_mcce_results_raw_data_k_{K}_n_{n_test}.csv"), index_col=0)
true_raw = pd.read_csv(os.path.join(path, f"adult_raw_data_n_{n_test}.csv"), index_col=0)

results_inverse['method'] = 'MCCE'
true_raw['method'] = 'Original'
temp = pd.concat([results_inverse, true_raw], axis=0)

cols = ['method', 'age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', \
       'hours-per-week', 'marital-status', 'native-country', \
       'occupation', 'race', 'relationship', 'sex', 'workclass']

to_write = temp[cols].loc[[1, 31, 122, 124]].sort_index()
to_write.columns = cols
# to_write.sort_values(['Method'], inplace=True, ascending=False)

print(to_write)