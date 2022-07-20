import warnings
warnings.filterwarnings('ignore')

from carla.data.catalog import OnlineCatalog
from carla.models.catalog import MLModelCatalog
from carla.models.negative_instances import predict_negative_instances

import torch
torch.manual_seed(0)

import pandas as pd
pd.set_option('display.max_columns', None)

import os
from mcce.metrics import distance, constraint_violation, feasibility, success_rate

import argparse

parser = argparse.ArgumentParser(description="Fit MCCE with various datasets.")
parser.add_argument(
    "-p",
    "--path",
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
    "-k",
    "--k",
    type=int,
    default=10000,
    help="Number generated counterfactuals per test observation",
)


args = parser.parse_args()

path = args.path
data_name = args.dataset
n_test = args.number_of_samples
k = args.k


results_inverse = pd.read_csv(os.path.join(path, f"adult_mcce_results_raw_data_k_{k}_n_{n_test}_inverse_transform.csv"), index_col=0)

print(f"L0: {results_inverse.L0.mean()}")
print(f"L2: {results_inverse.L2.mean()}")
print(f"Feasibility: {results_inverse.feasibility.mean()}")
print(f"Violations: {results_inverse.violation.mean()}")
print(f"Success: {results_inverse.success.mean()}")
print(f"N_CE: {results_inverse.shape[0]}")
print(f"Time (seconds): {results_inverse['time (seconds)'].mean()}")