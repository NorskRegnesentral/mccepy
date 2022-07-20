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

print("Fit predictive model")

dataset = OnlineCatalog(data_name)
ml_model = MLModelCatalog(
        dataset, 
        model_type="ann", 
        load_online=False, 
        backend="pytorch"
    )
if data_name == 'adult':
    ml_model.train(
    learning_rate=0.002,
    epochs=20,
    batch_size=1024,
    hidden_size=[18, 9, 3],
    force_train=True,
    )
elif data_name == 'give_me_some_credit':
    ml_model.train(
    learning_rate=0.002,
    epochs=20,
    batch_size=2048,
    hidden_size=[18, 9, 3],
    force_train=True,
    )
elif data_name == 'compas':
    ml_model.train(
    learning_rate=0.002,
    epochs=25,
    batch_size=25,
    hidden_size=[18, 9, 3],
    force_train=True,
    )

if data_name == 'adult':
    y = dataset.df_test['income']
elif data_name == 'give_me_some_credit':
    y = dataset.df_test['SeriousDlqin2yrs']
elif data_name == 'compas':
    y = dataset.df_test['score']

pred = ml_model.predict_proba(dataset.df_test)
pred = [row[1] for row in pred]
factuals = predict_negative_instances(ml_model, dataset.df)
test_factual = factuals.iloc[:100]

print("Compute metrics for CARLA methods")

carla_results = pd.DataFrame()
for method in ['cchvae', 'cem-vae', 'revise', 'clue', 'crud', 'face']:
    cfs = pd.read_csv(os.path.join(path, f"{data_name}_manifold_results.csv"), index_col=0)
    factuals = predict_negative_instances(ml_model, dataset.df)
    test_factual = factuals.iloc[:100]

    df_cfs = cfs[cfs['method'] == method].drop(['method',	'data'], axis=1)

    # missing values
    nan_idx = df_cfs.index[df_cfs.isnull().any(axis=1)]
    non_nan_idx = df_cfs.index[~(df_cfs.isnull()).any(axis=1)]

    output_factuals = test_factual.copy()
    output_counterfactuals = df_cfs.copy()

    factual_without_nans = output_factuals.drop(index=nan_idx)
    counterfactuals_without_nans = output_counterfactuals.drop(index=nan_idx)

    # counterfactuals
    results = pd.concat([dataset.inverse_transform(counterfactuals_without_nans)])
    results['method'] = method
    results['data'] = data_name

    # distances
    distances = pd.DataFrame(distance(counterfactuals_without_nans, factual_without_nans, ml_model))
    distances.set_index(non_nan_idx, inplace=True)
    results = pd.concat([results, distances], axis=1)

    results['feasibility'] = feasibility(counterfactuals_without_nans, factual_without_nans, \
        dataset.df.columns)
        
    # violations
    violations = []

    df_decoded_cfs = dataset.inverse_transform(counterfactuals_without_nans)
    df_factuals = dataset.inverse_transform(factual_without_nans)
    
    total_violations = constraint_violation(df_decoded_cfs, df_factuals, \
        dataset.continuous, dataset.categorical, dataset.immutables)
    for x in total_violations:
        violations.append(x[0])
    results['violation'] = violations
    
    # success
    results['success'] = success_rate(counterfactuals_without_nans, ml_model, cutoff=0.5)

    # time
    results['time (seconds)'] = df_cfs['time (seconds)'].mean() 

    carla_results = pd.concat([carla_results, results], axis=0)

print("Loading MCCE results")


results = pd.read_csv(os.path.join(path, f"{data_name}_mcce_results_k_{k}_n_{n_test}.csv"), index_col=0)
results.sort_index(inplace=True)
results['data'] = data_name
results['method'] = 'mcce'
results = dataset.inverse_transform(results)

## Concat

temp = pd.concat([carla_results, results[carla_results.columns]], axis=0)

## Write

to_write = temp[['method', 'L0', 'L2', 'feasibility', 'violation', 'success', 'time (seconds)']].groupby(['method']).mean()

to_write.reset_index(inplace=True)

CE_N = temp.groupby(['method']).size().reset_index().rename(columns={0: 'CE_N'})
to_write = pd.concat([to_write, CE_N.CE_N], axis=1)

# to_write.sort_values(['method'], inplace=True, ascending=False)
to_write = to_write[['method', 'L0', 'L2', 'feasibility', 'violation', 'success', 'CE_N', 'time (seconds)']]

print(to_write)