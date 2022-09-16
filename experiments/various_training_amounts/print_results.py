import os
import sys
import argparse
import pandas as pd
import numpy as np
pd.set_option('display.max_columns', None)

parser = argparse.ArgumentParser(description="Print results of training MCCE with various amounts of training data.")
parser.add_argument(
    "-p",
    "--path",
    type=str,
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
    "-k",
    "--k",
    type=int,
    default=10000,
    help="Number generated counterfactuals per test observation",
)
parser.add_argument(
    "-device",
    "--device",
    type=str,
    default='cuda',
    help="Whether the CARLA methods were trained with a GPU (default) or CPU.",
)
args = parser.parse_args()

path = args.path
data_name = args.dataset
n_test = args.number_of_samples
k = args.k
if data_name == 'compas':
    k = 1000
device = args.device # cuda # cpu

# -- mcce light --

try:
    temp = pd.read_csv(os.path.join(path, f"{data_name}_mcce_results_light_k_{k}_n_{n_test}_{device}.csv"), index_col=0)
    
    CE_N = temp.groupby(['method', 'n', 'seed']).size().reset_index().rename(columns={0: 'CE_N'})
    temp = temp.merge(CE_N, on=['method', 'seed', 'n'])

    to_write = temp[['method', 'n', 'L0', 'L2', 'feasibility', 'violation', 'success', 'CE_N', 'time (seconds)']].groupby(['method', 'n']).mean()
    to_write.reset_index(inplace=True)

    to_write_sd = temp[['method', 'n', 'L0', 'L2', 'feasibility', 'violation', 'success']].groupby(['method', 'n']).std()
    to_write_sd.reset_index(inplace=True)
    to_write_sd.rename(columns={'L0': 'L0_sd', 'L2': 'L2_sd', 'feasibility': 'feasibility_sd', 'violation': 'violation_sd', 'success': 'success_sd'}, inplace=True)

    to_write = pd.concat([to_write, to_write_sd[['L0_sd', 'L2_sd', 'feasibility_sd', 'violation_sd', 'success_sd']]], axis=1)

    to_write = to_write[['method', 'n', 'L0', 'L0_sd', 'L2', 'L2_sd', 'feasibility', 'feasibility_sd', 'violation', 'violation_sd', 'success', 'CE_N', 'time (seconds)']]
    
    # Fix method names
    dct = {'mcce': 'MCCE'}
    to_write['method'] = [dct[item] for item in to_write['method']]

    to_write = to_write.round(2)

    cols = ['L0', 'L0_sd', 'L2', 'L2_sd', 'feasibility', 'feasibility_sd', 'violation', 'violation_sd', 'success', 'CE_N']
    to_write[cols] = to_write[cols].astype(str)

    # Add the standard deviations in original columns
    to_write["L0"] = to_write["L0"] + " (" + to_write["L0_sd"] + ")"
    to_write["L2"] = to_write["L2"] + " (" + to_write["L2_sd"] + ")"
    to_write["feasibility"] = to_write["feasibility"] + " (" + to_write["feasibility_sd"] + ")"
    to_write["violation"] = to_write["violation"] + " (" + to_write["violation_sd"] + ")"
    
    print("MCCE LIGHT")
    # print(to_write.round(2).to_string())
    print(to_write[['method', 'n', 'L0', 'L2', 'feasibility', 'violation', 'success', 'CE_N', 'time (seconds)']].to_latex(index=False))

except:
    print(f"No MCCE LIGHT results saved for {data_name}, k {k}, and n_test {n_test} in {path}")


# -- mcce different subsets --

try:
    temp = pd.read_csv(os.path.join(path, f"{data_name}_mcce_results_various_training_amounts_k_{k}_n_{n_test}_{device}.csv"), index_col=0)

    CE_N = temp.groupby(['method', 'n', 'seed']).size().reset_index().rename(columns={0: 'CE_N'})
    temp = temp.merge(CE_N, on=['method', 'seed', 'n'])

    to_write = temp[['method', 'n', 'L0', 'L2', 'feasibility', 'violation', 'success', 'CE_N', 'time (seconds)']].groupby(['method', 'n']).mean()
    to_write.reset_index(inplace=True)

    to_write_sd = temp[['method', 'n', 'L0', 'L2', 'feasibility', 'violation', 'success']].groupby(['method', 'n']).std()
    to_write_sd.reset_index(inplace=True)
    to_write_sd.rename(columns={'L0': 'L0_sd', 'L2': 'L2_sd', 'feasibility': 'feasibility_sd', 'violation': 'violation_sd', 'success': 'success_sd'}, inplace=True)

    to_write = pd.concat([to_write, to_write_sd[['L0_sd', 'L2_sd', 'feasibility_sd', 'violation_sd', 'success_sd']]], axis=1)

    to_write = to_write[['method', 'n', 'L0', 'L0_sd', 'L2', 'L2_sd', 'feasibility', 'feasibility_sd', 'violation', 'violation_sd', 'success', 'CE_N', 'time (seconds)']]

    # Fix method names
    dct = {'mcce': 'MCCE'}
    to_write['method'] = [dct[item] for item in to_write['method']]

    to_write = to_write.round(2)

    cols = ['L0', 'L0_sd', 'L2', 'L2_sd', 'feasibility', 'feasibility_sd', 'violation', 'violation_sd', 'success', 'CE_N']
    to_write[cols] = to_write[cols].astype(str)

    # Add the standard deviations in original columns
    to_write["L0"] = to_write["L0"] + " (" + to_write["L0_sd"] + ")"
    to_write["L2"] = to_write["L2"] + " (" + to_write["L2_sd"] + ")"
    to_write["feasibility"] = to_write["feasibility"] + " (" + to_write["feasibility_sd"] + ")"
    to_write["violation"] = to_write["violation"] + " (" + to_write["violation_sd"] + ")"

    print("\nMCCE trained with different amounts of training data")
    # print(to_write.round(2).to_string())
    print(to_write[['method', 'n', 'L0', 'L2', 'feasibility', 'violation', 'success', 'CE_N', 'time (seconds)']].to_latex(index=False))


except:
    print(f"No MCCE different subsets results saved for {data_name}, k {k}, and n_test {n_test} in {path}")
