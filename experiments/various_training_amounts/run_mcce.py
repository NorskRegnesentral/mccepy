import warnings
warnings.filterwarnings('ignore')

import os
import argparse
import torch
import time
import random 
import pandas as pd
import numpy as np

from carla.data.catalog import OnlineCatalog
from carla.models.catalog import MLModelCatalog
from carla.models.negative_instances import predict_negative_instances

from mcce.metrics import feasibility
from mcce.mcce import MCCE

## FOR EACH DATA SET you have to adjust n below - 
## for adult and gmc, I use 100, 1000, 10000 and the size of the data set
## for compas, I use 100, 1000, 5000, and the size of the data aset 

PATH = "Final_results_new"

parser = argparse.ArgumentParser(description="Fit various recourse methods from CARLA.")
parser.add_argument(
    "-d",
    "--dataset",
    nargs="*",
    default="adult",
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
    "-K",
    "--K",
    type=int,
    default=10000,
    help="Number of instances to sample for MCCE.",
)

args = parser.parse_args()

K = args.K
n_test = args.number_of_samples
seed = 1
data_name = args.dataset

dataset = OnlineCatalog(data_name)

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

# (2) Find unhappy customers and choose which ones to make counterfactuals for
factuals = predict_negative_instances(ml_model, dataset.df)
test_factual = factuals.iloc[:n_test]

y_col = dataset.target
cont_feat = dataset.continuous

cat_feat = dataset.categorical
cat_feat_encoded = dataset.encoder.get_feature_names(dataset.categorical)

if data_name == 'adult': 
    fixed_features_encoded = ['age', 'sex_Male']
    fixed_features = ['age', 'sex']
elif data_name == 'give_me_some_credit':
    fixed_features_encoded = ['age']
    fixed_features = ['age']
elif data_name == 'compas':
    fixed_features_encoded = ['age', 'sex_Male', 'race_Other']
    fixed_features = ['age', 'sex', 'race']

#  Create dtypes for MCCE()
dtypes = dict([(x, "float") for x in cont_feat])
for x in cat_feat_encoded:
    dtypes[x] = "category"
df = (dataset.df).astype(dtypes)

## Loop through various subsets of data and train trees on the smaller subsets - 48832 is the full data set

if data_name == 'adult': 
    n_list = [100, 1000, 10000, dataset.df.shape[0]]
elif data_name == 'give_me_some_credit':
    n_list = [100, 1000, 10000, 50000, dataset.df.shape[0]]
elif data_name == 'compas':
    n_list = [100, 1000, 5000, dataset.df.shape[0]]



results = []
results_pd = pd.DataFrame()
for n in n_list:

    if n == dataset.df.shape[0]:
        
        random.seed(0)
        rows = random.sample(df.index.to_list(), n)
        rows = np.sort(rows)
        df_subset = df.loc[rows]
        
        start = time.time()

        mcce = MCCE(fixed_features=fixed_features,\
                fixed_features_encoded=fixed_features_encoded,
                    continuous=dataset.continuous, categorical=dataset.categorical,\
                        model=ml_model, seed=1)

        mcce.fit(df.drop(dataset.target, axis=1), dtypes)

        synth_df = mcce.generate(test_factual.drop(dataset.target, axis=1), k=100)
        mcce.postprocess(synth=synth_df, test=test_factual, response=y_col, \
            inverse_transform=dataset.inverse_transform, cutoff=0.5)

        timing = time.time() - start

        # Feasibility 
        cols = dataset.df.columns.to_list()
        cols.remove(dataset.target)
        mcce.results_sparse['feasibility'] = feasibility(mcce.results_sparse, dataset.df, cols)

        mcce.results_sparse['time (seconds)'] = timing

        results.append([mcce.results_sparse.L0.mean(), mcce.results_sparse.L2.mean(), mcce.results_sparse.feasibility.mean(), mcce.results_sparse.violation.mean(), mcce.results_sparse.shape[0], timing, n, 0])
    else:
        for s in range(5):
            random.seed(s)
            rows = random.sample(df.index.to_list(), n)
            rows = np.sort(rows)
            df_subset = df.loc[rows]

            start = time.time()

            mcce = MCCE(fixed_features=fixed_features,\
                    fixed_features_encoded=fixed_features_encoded,
                        continuous=dataset.continuous, categorical=dataset.categorical,\
                            model=ml_model, seed=1)

            mcce.fit(df.drop(dataset.target, axis=1), dtypes)

            synth_df = mcce.generate(test_factual.drop(dataset.target, axis=1), k=100)
            mcce.postprocess(synth=synth_df, test=test_factual, response=y_col, \
                inverse_transform=dataset.inverse_transform, cutoff=0.5)

            timing = time.time() - start
            
            # Feasibility 
            cols = dataset.df.columns.to_list()
            cols.remove(dataset.target)
            mcce.results_sparse['feasibility'] = feasibility(mcce.results_sparse, dataset.df, cols)

            mcce.results_sparse['time (seconds)'] = timing

            temp = mcce.results_sparse
            temp['n'] = n

            results.append([mcce.results_sparse.L0.mean(), mcce.results_sparse.L2.mean(), mcce.results_sparse.feasibility.mean(), mcce.results_sparse.violation.mean(), mcce.results_sparse.shape[0], timing, n, s])

results2 = pd.DataFrame(results, columns=['L0', 'L2', 'feasibility', 'violation', 'NCE', 'timing', 'Ntest', 'seed'])

results2.to_csv(os.path.join(PATH, f"{data_name}_mcce_results_k_{K}_n_{n_test}_with_various_training_amounts.csv"))

