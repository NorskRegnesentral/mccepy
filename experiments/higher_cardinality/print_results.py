import os
import sys
import argparse
import warnings
warnings.filterwarnings('ignore')

import torch
torch.manual_seed(0)

from carla.data.catalog import CsvCatalog
from carla.models.catalog import MLModelCatalog
from carla.models.negative_instances import predict_negative_instances

import pandas as pd
pd.set_option('display.max_columns', None)

from mcce.metrics import distance, constraint_violation, feasibility, success_rate

parser = argparse.ArgumentParser(description="Calculate and print MCCE metric results.")
parser.add_argument(
    "-p",
    "--path",
    type=str,
    required=True,
    help="Path where results are saved",
)
parser.add_argument(
    "-n",
    "--number_of_samples",
    type=int,
    default=100,
    help="Number of test observations to generate counterfactuals for.",
)
parser.add_argument(
    "-k",
    "--k",
    type=int,
    default=10000,
    help="Number of observations to sample from each end node for MCCE method.",
)
parser.add_argument(
    "-ft",
    "--force_train",
    action='store_true',  # default is False
    help="Whether to train the prediction model from scratch or not. Default will not train.",
)


args = parser.parse_args()

path = args.path
n_test = args.number_of_samples
k = args.k
force_train = args.force_train

print("Read in processed data using CARLA functionality")
continuous = ["age", "fnlwgt", "education-num", "capital-gain", "hours-per-week", "capital-loss"]
categorical = ["marital-status", "native-country", "occupation", "race", "relationship", "sex", "workclass"]
immutables = ["age", "sex"]

dataset = CsvCatalog(file_path="Data/train_not_normalized_data_from_carla.csv",
                     continuous=continuous,
                     categorical=categorical,
                     immutables=immutables,
                     target='income',
                     encoding_method="OneHot_drop_first", # New!
                     )

print("Fit predictive model")
torch.manual_seed(0)
ml_model = MLModelCatalog(
        dataset, 
        model_type="ann", 
        load_online=False, 
        backend="pytorch"
    )
ml_model.train(
learning_rate=0.002,
epochs=20,
batch_size=1024,
hidden_size=[18, 9, 3],
force_train=force_train,
)

y = dataset.df_test['income']

pred = ml_model.predict_proba(dataset.df_test)
pred = [row[1] for row in pred]
factuals = predict_negative_instances(ml_model, dataset.df)
test_factual = factuals.iloc[:n_test]

print(f"Calculating results")

try:
    cfs = pd.read_csv(os.path.join(path, f"adult_mcce_results_higher_cardinality_k_{k}_n_{n_test}.csv"), index_col=0)
except:
    sys.exit(f"No MCCE results saved for k {k} and n_test {n_test} in {path}")
    

df_cfs = cfs.drop(['method', 'data'], axis=1)
df_cfs.sort_index(inplace=True)

# remove missing values
nan_idx = df_cfs.index[df_cfs.isnull().any(axis=1)]
non_nan_idx = df_cfs.index[~(df_cfs.isnull()).any(axis=1)]

output_factuals = test_factual.loc[df_cfs.index.to_list()]
output_counterfactuals = df_cfs

factual_without_nans = output_factuals.drop(index=nan_idx)
counterfactuals_without_nans = output_counterfactuals.drop(index=nan_idx)

# calculate metrics
if len(counterfactuals_without_nans) > 0:
    results = dataset.inverse_transform(counterfactuals_without_nans[factuals.columns])
    results['method'] = 'mcce'
    results['data'] = 'adult'
    
    # calculate distances
    distances = pd.DataFrame(distance(counterfactuals_without_nans, factual_without_nans, dataset, higher_card=True))
    distances.set_index(non_nan_idx, inplace=True)
    results = pd.concat([results, distances], axis=1)

    # calculate feasibility
    results['feasibility'] = feasibility(counterfactuals_without_nans, factual_without_nans, dataset.df.columns)
    
    # calculate violation
    violations = []
    df_decoded_cfs = dataset.inverse_transform(counterfactuals_without_nans)
    df_factuals = dataset.inverse_transform(factual_without_nans)
    
    # calculate violations
    total_violations = constraint_violation(df_decoded_cfs, df_factuals, dataset)
    for x in total_violations:
        violations.append(x[0])
    results['violation'] = violations
    
    # calculate success
    results['success'] = success_rate(counterfactuals_without_nans, ml_model, cutoff=0.5)

    # calculate time
    results['time (seconds)'] = df_cfs['time (seconds)'].mean() 


cols = ['method', 'L0', 'L2', 'feasibility', 'success', 'violation', 'time (seconds)']
temp = results[cols]

print("Writing results")
to_write = temp[['method', 'L0', 'L2', 'feasibility', 'violation', 'success', 'time (seconds)']].groupby(['method']).mean()
to_write.reset_index(inplace=True)

to_write_sd = temp[['method', 'L0', 'L2', 'feasibility', 'violation', 'success']].groupby(['method']).std()
to_write_sd.reset_index(inplace=True)
to_write_sd.rename(columns={'L0': 'L0_sd', 'L2': 'L2_sd', 'feasibility': 'feasibility_sd', 'violation': 'violation_sd', 'success': 'success_sd'}, inplace=True)

CE_N = temp.groupby(['method']).size().reset_index().rename(columns={0: 'CE_N'})
to_write = pd.concat([to_write, to_write_sd[['L0_sd', 'L2_sd', 'feasibility_sd', 'violation_sd', 'success_sd']], CE_N.CE_N], axis=1)
to_write = to_write[['method', 'L0', 'L0_sd', 'L2', 'L2_sd', 'feasibility', 'feasibility_sd', 'violation', 'violation_sd', 'success', 'CE_N', 'time (seconds)']]

print(to_write.round(2).to_string())
