import warnings
warnings.filterwarnings('ignore')

import os
import argparse
import torch
import time
import random
import numpy as np
import pandas as pd

from carla.data.catalog import OnlineCatalog
from carla.models.catalog import MLModelCatalog
from carla.models.negative_instances import predict_negative_instances, predict_label

from mcce.mcce import MCCE
from mcce.metrics import distance, constraint_violation, feasibility, success_rate

## FOR EACH DATA SET you have to adjust n below - 
## for adult and gmc, I use 100, 1000, 10000 and the size of the data set
## for compas, I use 100, 1000, 5000, and the size of the data set 

parser = argparse.ArgumentParser(description="Fit MCCE light method.")
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
    default="adult",
    help="Datasets for experiment. Options are adult, give_me_some_credit, and compas.",
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
parser.add_argument(
    "-s",
    "--s",
    type=int,
    default=5,
    help="Number of seeds to set for simulation.",
)

device = "cuda" if torch.cuda.is_available() else "cpu"

args = parser.parse_args()

n_test = args.number_of_samples
k = args.k
data_name = args.dataset
if data_name == 'compas':
    k = 1000
force_train = args.force_train
num_seeds = args.s
path = args.path
seed = 1

# Load data set from CARLA
dataset = OnlineCatalog(data_name)

# Train predictive model also from CARLA
torch.manual_seed(0)
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

# "Find unhappy customers and choose which ones to make counterfactuals for"
factuals = predict_negative_instances(ml_model, dataset.df)
test_factual = factuals.iloc[:n_test]

y_col = dataset.target
cont_feat = dataset.continuous

cat_feat = dataset.categorical
cat_feat_encoded = dataset.encoder.get_feature_names(dataset.categorical)

#  Create dtypes for MCCE()
dtypes = dict([(x, "float") for x in cont_feat])
for x in cat_feat_encoded:
    dtypes[x] = "category"
df = (dataset.df).astype(dtypes)

print("Loop through various subsets of data and train trees on the smaller subsets")
factual_indices = test_factual.index.to_list()
all_indices = dataset.df.index.to_list()
possible_train_indices = set(factual_indices) ^ set(all_indices)
if data_name == 'adult': 
    n_list = [100, 1000, 10000, len(possible_train_indices)]
elif data_name == 'give_me_some_credit':
    n_list = [100, 1000, 10000, 50000, len(possible_train_indices)]
elif data_name == 'compas':
    n_list = [100, 1000, 5000, len(possible_train_indices)]

# Fit "MCCE-light" method
all_results = pd.DataFrame()
mcce = MCCE(dataset=dataset, model=ml_model)

for n in n_list:
    print(f"Number of rows: {n}.")
    for s in range(num_seeds):
        print(f"Seed: {s}.")
        start = time.time()
        
        if n == len(possible_train_indices): # If using the whole data set
            if(s > 0): # Only need to generate counterfactuals once for full data set
                break
            
        random.seed(s)
        rows = random.sample(possible_train_indices, n)
        rows = np.sort(rows)

        positives = (df.loc[rows]).copy()
        positives["y"] = predict_label(ml_model, positives)
        positives = positives[positives["y"] == 1] # find obs with positive outcome
        positives = positives.drop("y", axis="columns")

        positives = dataset.inverse_transform(positives)
        test_factual_inverse = dataset.inverse_transform(test_factual)
        test_factual_inverse.index.name = 'test'
        
        cfs = pd.merge(test_factual_inverse.reset_index()[dataset.immutables + ['test']], 
                         positives, 
                         on=dataset.immutables).set_index(['test'])
        cfs = dataset.transform(cfs) # Go from normal to one-hot encoded
        generate_samples = time.time()

        print("Process sampled observations")
        mcce.postprocess(cfs, test_factual, cutoff=0.5)
        time_postprocess = time.time()

        mcce.results_sparse['time (seconds)'] = time_postprocess - start

        df_cfs = mcce.results_sparse
        df_cfs.sort_index(inplace=True)
        df_cfs[y_col] = test_factual[y_col]  # add back original response

        # Remove missing values
        nan_idx = df_cfs.index[df_cfs.isnull().any(axis=1)]
        non_nan_idx = df_cfs.index[~(df_cfs.isnull()).any(axis=1)]

        output_factuals = test_factual.loc[df_cfs.index.to_list()]
        output_counterfactuals = df_cfs

        factual_without_nans = output_factuals.drop(index=nan_idx)
        counterfactuals_without_nans = output_counterfactuals.drop(index=nan_idx)

        if len(counterfactuals_without_nans) > 0:
            print(f"Calculating results for subset of size {n} and seed {s}")
            results = dataset.inverse_transform(counterfactuals_without_nans[factuals.columns])
            results['method'] = 'mcce'
            results['data'] = data_name
            results['n'] = n # number of training observations to train trees on
            results['seed'] = s # seed number
            
            # calculate distances
            distances = pd.DataFrame(distance(counterfactuals_without_nans, factual_without_nans, dataset, higher_card=False))
            distances.set_index(non_nan_idx, inplace=True)
            results = pd.concat([results, distances], axis=1)

            # calculate feasibility
            results['feasibility'] = feasibility(counterfactuals_without_nans, factual_without_nans, dataset.df.columns)
            
            # calculate violation
            violations = []
            df_decoded_cfs = dataset.inverse_transform(counterfactuals_without_nans)
            df_factuals = dataset.inverse_transform(factual_without_nans)
            
            total_violations = constraint_violation(df_decoded_cfs, df_factuals, dataset)
            for x in total_violations:
                violations.append(x[0])
            results['violation'] = violations
            
            # calculate success
            results['success'] = success_rate(counterfactuals_without_nans, ml_model, cutoff=0.5)

            # calculate time
            results['time (seconds)'] = df_cfs['time (seconds)'].mean() 

            all_results = pd.concat([all_results, results], axis=0)

cols = ['data', 'method', 'n', 'seed', 'L0', 'L1', 'L2', 'feasibility', 'violation', 'success', 'time (seconds)'] + cat_feat + cont_feat + [y_col]

all_results[cols].to_csv(os.path.join(path, f"{data_name}_mcce_results_light_k_{k}_n_{n_test}_{device}.csv"))