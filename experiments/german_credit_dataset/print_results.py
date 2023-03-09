import os
import argparse
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

import torch
torch.manual_seed(0)

from carla.data.catalog import OnlineCatalog
from carla.models.catalog import MLModelCatalog
from carla.data.catalog import CsvCatalog
from carla.models.negative_instances import predict_negative_instances
from sklearn import preprocessing
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
    default='german_credit',
    choices=["german_credit"],
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
    "-k",
    "--k",
    type=int,
    default=1000,
    help="Number generated counterfactuals per test observation",
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
    help="Whether the CARLA methods were trained with a GPU or CPU (default).",
)

args = parser.parse_args()

path = args.path
data_name = args.dataset
n_test = args.number_of_samples
force_train = args.force_train
device = args.device # cuda # cpu
k = args.k

print(f"Load {data_name} data set")

categorical = ["Sex", "Job", "Housing", "Saving accounts", "Checking account", "Purpose"]
continuous = ["Age", "Credit amount", "Duration"]
immutable = ["Purpose", "Age", "Sex"]

encoding_method = preprocessing.OneHotEncoder(
            drop="first", sparse=False
        )
dataset = CsvCatalog(file_path="Data/german_credit_data_complete.csv",
                     continuous=continuous,
                     categorical=categorical,
                     immutables=immutable,
                     target='Risk',
                     encoding_method=encoding_method
                     )
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

factuals = predict_negative_instances(ml_model, dataset.df)
test_factual = factuals.iloc[:n_test]
# test_factual = pd.read_csv(os.path.join(path, f"german_credit_test_factuals_ann_model_k_{k}_n_{n_test}_{device}.csv"), index_col=0)


all_results = pd.DataFrame()
for method in [ 'cchvae', 'revise', 'clue', 'crud', 'face', 'mcce']: # 'cem-vae'
    print(f"Calculating results for {method}")

    if method == 'mcce':
        try:
            cfs = pd.read_csv(os.path.join(path, f"{data_name}_mcce_results_ann_model_k_{k}_n_{n_test}_{device}.csv"), index_col=0)
        except:
            print(f"No {method} results saved for n_test {n_test} in {path}")
            continue
    else:
        print(os.path.join(path, f"{data_name}_carla_results_n_{n_test}_{device}.csv"))
        
        try:
            cfs = pd.read_csv(os.path.join(path, f"{data_name}_carla_results_n_{n_test}_{device}.csv"), index_col=0)
        except:
            print(f"No {method} results saved for n_test {n_test} in {path}")
            continue
    
    df_cfs = cfs[cfs['method'] == method].drop(['method',	'data'], axis=1)
    df_cfs = df_cfs.loc[~df_cfs.index.isnull()]
    a = df_cfs.index.to_list()
    df_cfs = df_cfs.set_index([pd.Index([int(a) for a in a])])

    # In case the script was accidentally run twice, we drop the duplicate indices per method
    df_cfs = df_cfs[~df_cfs.index.duplicated(keep='first')]
    
    df_cfs.sort_index(inplace=True)
    if dataset.target not in df_cfs.columns:
        df_cfs = df_cfs.join(test_factual[dataset.target])
    lst3 = [value for value in df_cfs.index.to_list() if value in test_factual.index.to_list()]

    test_factual = test_factual.loc[lst3]
    df_cfs = df_cfs.loc[lst3]

    # remove missing values
    nan_idx = df_cfs.index[df_cfs.isnull().any(axis=1)]
    non_nan_idx = df_cfs.index[~(df_cfs.isnull()).any(axis=1)]

    output_factuals = test_factual.loc[df_cfs.index.to_list()]
    output_counterfactuals = df_cfs

    factual_without_nans = output_factuals.drop(index=nan_idx)
    counterfactuals_without_nans = output_counterfactuals.drop(index=nan_idx)
    
    factual_without_nans = output_factuals.drop(index=nan_idx)
    counterfactuals_without_nans = output_counterfactuals.drop(index=nan_idx)

    for x in dataset.continuous:
        factual_without_nans[x] = factual_without_nans[x].astype(float)
        counterfactuals_without_nans[x] = counterfactuals_without_nans[x].astype(float)

    cat_feat_encoded = dataset.encoder.get_feature_names(dataset.categorical)
    for x in cat_feat_encoded:
        factual_without_nans[x] = factual_without_nans[x].astype(float)
        counterfactuals_without_nans[x] = counterfactuals_without_nans[x].astype(float)

    factual_without_nans[dataset.target] = factual_without_nans[dataset.target].astype(float)
    counterfactuals_without_nans[dataset.target] = counterfactuals_without_nans[dataset.target].astype(float)
    
    # calculate metrics
    if len(counterfactuals_without_nans) > 0:
        results = dataset.inverse_transform(counterfactuals_without_nans) # [factuals.columns]
        results['method'] = method
        results['data'] = data_name
        
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
        results['time (seconds)'] = df_cfs['time (seconds)'].astype('float').mean() 

        all_results = pd.concat([all_results, results], axis=0)

cols = ['method', 'L0', 'L1', 'feasibility', 'success', 'violation', 'time (seconds)']
temp = all_results[cols]

print(f"Writing results for {data_name} {device}")
to_write_mean = temp[['method', 'L0', 'L1', 'feasibility', 'violation', 'success', 'time (seconds)']].groupby(['method']).mean()
to_write_mean.reset_index(inplace=True)

to_write_sd = temp[['method', 'L0', 'L1', 'feasibility', 'violation', 'success']].groupby(['method']).std()
to_write_sd.reset_index(inplace=True)
to_write_sd.rename(columns={'L0': 'L0_sd', 'L1': 'L1_sd', 'feasibility': 'feasibility_sd', 'violation': 'violation_sd', 'success': 'success_sd'}, inplace=True)

CE_N = temp.groupby(['method']).size().reset_index().rename(columns={0: 'CE_N'})
to_write = pd.concat([to_write_mean, to_write_sd[['L0_sd', 'L1_sd', 'feasibility_sd', 'violation_sd', 'success_sd']], CE_N.CE_N], axis=1)
to_write = to_write[['method', 'L0', 'L0_sd', 'L1', 'L1_sd', 'feasibility', 'feasibility_sd', 'violation', 'violation_sd', 'success', 'CE_N', 'time (seconds)']]

# Fix method names
dct = {'original': 'Original', 
       'cchvae': 'C-CHVAE',
       'cem-vae': 'CEM-VAE',
       'clue': 'CLUE',
       'crud': 'CRUDS',
       'face': 'FACE',
       'revise': 'REViSE',
       'mcce': 'MCCE'}

to_write['method'] = [dct[item] for item in to_write['method']]

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

print(to_write[['method', 'L0', 'L1', 'feasibility', 'violation', 'success', 'CE_N', 'time (seconds)']].to_latex(index=False))
