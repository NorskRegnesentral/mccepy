import os
import sys
import argparse
import warnings
import numpy as np
warnings.filterwarnings('ignore')

from carla.data.catalog import OnlineCatalog
from carla.models.catalog import MLModelCatalog
from carla.models.negative_instances import predict_negative_instances

import torch
torch.manual_seed(0)

import pandas as pd
pd.set_option('display.max_columns', None)

from mcce.metrics import distance, constraint_violation, feasibility, success_rate

parser = argparse.ArgumentParser(description="Fit MCCE with various datasets.")
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
    type=str,
    default='adult',
    choices=["adult", "give_me_some_credit", "compas"],
    help="Datasets for experiment",
)
parser.add_argument(
    "-n",
    "--number_of_samples",
    type=int,
    default=1000,
    help="Number of instances per dataset",
)
parser.add_argument(
    "-ft",
    "--force_train",
    action='store_true',  # default is False
    help="Whether to train the prediction model from scratch or not. Default will not train.",
)
parser.add_argument(
    "-device",
    "--device",
    type=str,
    default='cpu',
    help="Whether the CARLA methods were trained with a GPU or CPU.",
)

args = parser.parse_args()

path = args.path
data_name = args.dataset
n_test = args.number_of_samples
force_train = args.force_train
device = args.device # cuda # cpu

print(f"Load {data_name} data set")

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
    force_train=force_train,
    )
elif data_name == 'give_me_some_credit':
    ml_model.train(
    learning_rate=0.002,
    epochs=20,
    batch_size=2048,
    hidden_size=[18, 9, 3],
    force_train=force_train,
    )
elif data_name == 'compas':
    ml_model.train(
    learning_rate=0.002,
    epochs=25,
    batch_size=25,
    hidden_size=[18, 9, 3],
    force_train=force_train,
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
test_factual = factuals.iloc[:n_test]

all_results = pd.DataFrame()
try:
    cfs = pd.read_csv(os.path.join(path, f"{data_name}_mcce_results_k_several_n_{n_test}_{device}.csv"), index_col=0)
except:
    sys.exit(f"No mcce results saved for n_test {n_test} in {path}")

for k in [10, 50, 100, 1000, 5000, 10000, 25000]:

    df_cfs = cfs[cfs['k'] == k].drop(['method',	'data'], axis=1)
    # In case the script was accidentally run twice, we drop the duplicate indices per method
    df_cfs = df_cfs[~df_cfs.index.duplicated(keep='first')]

    df_cfs.sort_index(inplace=True)
    if dataset.target not in df_cfs.columns:
        df_cfs = df_cfs.join(test_factual[dataset.target])
    
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
        results['data'] = data_name
        results['k'] = k
        
        # distance
        distances = pd.DataFrame(distance(counterfactuals_without_nans, factual_without_nans, dataset, higher_card=False))
        distances.set_index(non_nan_idx, inplace=True)
        results = pd.concat([results, distances], axis=1)

        # feasibility
        feas_col = dataset.df.columns.to_list()
        feas_col.remove(dataset.target)
        results['feasibility'] = feasibility(counterfactuals_without_nans, factual_without_nans, feas_col)
        
        # violation
        violations = []
        df_decoded_cfs = dataset.inverse_transform(counterfactuals_without_nans)
        df_factuals = dataset.inverse_transform(factual_without_nans)
        
        total_violations = constraint_violation(df_decoded_cfs, df_factuals, dataset)
        for x in total_violations:
            violations.append(x[0])
        results['violation'] = violations
        
        # success
        results['success'] = success_rate(counterfactuals_without_nans, ml_model, cutoff=0.5)

        # time
        results['time (seconds)'] = df_cfs['time (seconds)'].mean()
        results['fit (seconds)'] = df_cfs['fit (seconds)'].mean()
        results['generate (seconds)'] = df_cfs['generate (seconds)'].mean()
        results['postprocess (seconds)'] = df_cfs['postprocess (seconds)'].mean()
        
        all_results = pd.concat([all_results, results], axis=0)

cols = ['k', 'L0', 'L1', 'feasibility', 'success', 'violation', 'time (seconds)', 'fit (seconds)', 'generate (seconds)', 'postprocess (seconds)']
temp = all_results[cols]

print(f"Writing results for {data_name} {device}")
to_write_mean = temp[['k', 'L0', 'L1', 'feasibility', 'violation', 'success', 'time (seconds)', 'fit (seconds)', 'generate (seconds)', 'postprocess (seconds)' ]].groupby(['k']).mean()
to_write_mean.reset_index(inplace=True)

to_write_sd = temp[['k', 'L0', 'L1', 'feasibility', 'violation', 'success']].groupby(['k']).std()
to_write_sd.reset_index(inplace=True)
to_write_sd.rename(columns={'L0': 'L0_sd', 'L1': 'L1_sd', 'feasibility': 'feasibility_sd', 'violation': 'violation_sd', 'success': 'success_sd'}, inplace=True)

CE_N = temp.groupby(['k']).size().reset_index().rename(columns={0: 'CE_N'})

to_write = pd.concat([to_write_mean, to_write_sd[['L0_sd', 'L1_sd', 'feasibility_sd', 'violation_sd', 'success_sd']], CE_N.CE_N], axis=1)
to_write = to_write[['k', 'L0', 'L0_sd', 'L1', 'L1_sd', 'feasibility', 'feasibility_sd', 'violation', 'violation_sd', 'success', 'CE_N', 'time (seconds)', 'fit (seconds)', 'generate (seconds)', 'postprocess (seconds)']]

# Fix method names
to_write['method'] = 'MCCE'

# Order the methods
s1 = to_write[to_write['method'] == 'MCCE']
s2 = to_write[(to_write['method'] != 'Original') & (to_write['method'] != 'MCCE')]
to_write = pd.concat([s2.sort_values('method'), s1])

# Remove decimal point
num_feat = ['CE_N']
to_write[num_feat] = to_write[num_feat].astype(np.int64)

to_write = to_write.round(2)

cols = ['L0', 'L0_sd', 'L1', 'L1_sd', 'feasibility', 'feasibility_sd', 'violation', 'violation_sd', 'success']
to_write[cols] = to_write[cols].astype(str)

# Add the standard deviations in original columns
to_write["L0"] = to_write["L0"] + " (" + to_write["L0_sd"] + ")"
to_write["L1"] = to_write["L1"] + " (" + to_write["L1_sd"] + ")"
to_write["feasibility"] = to_write["feasibility"] + " (" + to_write["feasibility_sd"] + ")"
to_write["violation"] = to_write["violation"] + " (" + to_write["violation_sd"] + ")"

print(to_write[['k', 'L0', 'L1', 'feasibility', 'violation',  'CE_N', 'time (seconds)', 'fit (seconds)', 'generate (seconds)', 'postprocess (seconds)']].to_latex(index=False))
