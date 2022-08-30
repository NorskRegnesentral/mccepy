import os
import argparse
import warnings
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
    "-ft",
    "--force_train",
    action='store_true',  # default is False
    help="Whether to train the prediction model from scratch or not. Default will not train.",
)

args = parser.parse_args()

path = args.path
data_name = args.dataset
n_test = args.number_of_samples
k = args.k
force_train = args.force_train

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
for method in ['cchvae', 'cem-vae', 'revise', 'clue', 'crud', 'face', 'mcce']:
    print(f"Calculating results for {method}")

    if method == 'mcce':
        cfs = pd.read_csv(os.path.join(path, f"{data_name}_mcce_results_k_{k}_n_{n_test}.csv"), index_col=0)
    else:    
        cfs = pd.read_csv(os.path.join(path, f"{data_name}_manifold_results.csv"), index_col=0)
    
    df_cfs = cfs[cfs['method'] == method].drop(['method',	'data'], axis=1)
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
        results['method'] = method
        results['data'] = data_name
        
        # distance
        distances = pd.DataFrame(distance(counterfactuals_without_nans, factual_without_nans, dataset, higher_card=False))
        distances.set_index(non_nan_idx, inplace=True)
        results = pd.concat([results, distances], axis=1)

        results['feasibility'] = feasibility(counterfactuals_without_nans, factual_without_nans, dataset.df.columns)
        
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

        all_results = pd.concat([all_results, results], axis=0)

# OLD RESULTS THAT ARE CURRENTLY IN THE PAPER
# if data_name == 'adult':
#     all_results = pd.read_csv("Final_results_Aug/adult_results_mcce_and_carla_K_10000_n_100.csv", index_col=0)
# elif data_name == 'give_me_some_credit':
#     all_results = pd.read_csv("Final_results_Aug/give_me_some_credit_results_mcce_and_carla_K_10000_n_100.csv")

# all_results.rename(columns={'violations': 'violation', 'validity': 'success'}, inplace=True)
# results = pd.read_csv(os.path.join(path, f"{data_name}_mcce_results_k_{K}_n_{n_test}.csv"), index_col=0)
# results.sort_index(inplace=True)

print("Concat all results")
cols = ['method', 'L0', 'L2', 'feasibility', 'success', 'violation', 'time (seconds)']
temp = all_results[cols]

print(f"Writing results for {data_name}")
to_write = temp[['method', 'L0', 'L2', 'feasibility', 'violation', 'success', 'time (seconds)']].groupby(['method']).mean()
to_write.reset_index(inplace=True)

to_write_sd = temp[['method', 'L0', 'L2', 'feasibility', 'violation', 'success']].groupby(['method']).std()
to_write_sd.reset_index(inplace=True)
to_write_sd.rename(columns={'L0': 'L0_sd', 'L2': 'L2_sd', 'feasibility': 'feasibility_sd', 'violation': 'violation_sd', 'success': 'success_sd'}, inplace=True)


CE_N = temp.groupby(['method']).size().reset_index().rename(columns={0: 'CE_N'})
to_write = pd.concat([to_write, to_write_sd[['L0_sd', 'L2_sd', 'feasibility_sd', 'violation_sd', 'success_sd']], CE_N.CE_N], axis=1)
to_write = to_write[['method', 'L0', 'L0_sd', 'L2', 'L2_sd', 'feasibility', 'feasibility_sd', 'violation', 'violation_sd', 'success', 'CE_N', 'time (seconds)']]

print(to_write.round(2).to_string())