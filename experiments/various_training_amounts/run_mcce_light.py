import warnings
warnings.filterwarnings('ignore')

from carla.data.catalog import OnlineCatalog
from carla.models.catalog import MLModelCatalog
from carla.models.negative_instances import predict_negative_instances, predict_label

import os
import argparse
import torch
import time
import random
import numpy as np
import pandas as pd
            
from mcce.mcce import MCCE

PATH = "../../Results_test/"

## FOR EACH DATA SET you have to adjust n below - 
## for adult and gmc, I use 100, 1000, 10000 and the size of the data set
## for compas, I use 100, 1000, 5000, and the size of the data set 

parser = argparse.ArgumentParser(description="Fit various recourse methods from CARLA.")
parser.add_argument(
    "-d",
    "--dataset",
    nargs="*",
    default=["adult", "give_me_some_credit", "compas"],
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
    type=int,
    default=100,
    help="Number of instances to sample for MCCE.",
)

args = parser.parse_args()

K = args.K
n_test = args.n
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

## Subset full data set

factual_indices = test_factual.index.to_list()
all_indices = dataset.df.index.to_list()
possible_train_indices = set(factual_indices) ^ set(all_indices)
if data_name == 'adult': 
    n_list = [100, 1000, 10000, len(possible_train_indices)]
elif data_name == 'give_me_some_credit':
    n_list = [100, 1000, 10000, len(possible_train_indices)]
elif data_name == 'compas':
    n_list = [100, 1000, 5000, len(possible_train_indices)]

## Here we fit "MCCE-light" method

results = []
for n in n_list:
    print(n)

    if n == len(possible_train_indices): # if the whole data set

        dim = dataset.df.shape[0]

        random.seed(s)
        rows = random.sample(possible_train_indices, n)
        rows = np.sort(rows)

        positives = (df.loc[rows]).copy()
        positives["y"] = predict_label(ml_model, positives)
        positives = positives[positives["y"] == 1]
        positives = positives.drop("y", axis="columns")

        positives = dataset.inverse_transform(positives)
        test_factual_inverse = dataset.inverse_transform(test_factual)
        test_factual_inverse.index.name = 'test'

        start = time.time()

        synth = pd.merge(test_factual_inverse.reset_index()[dataset.immutables + ['test']], positives, on = dataset.immutables).set_index(['test']) # 'train',
        synth = dataset.transform(synth) # go from normal to one-hot encoded

        mcce = MCCE(fixed_features=fixed_features,\
                fixed_features_encoded=fixed_features_encoded,
                    continuous=dataset.continuous, categorical=dataset.categorical,\
                        model=ml_model, seed=1)

        mcce.fit(df.drop(dataset.target, axis=1), dtypes)

        mcce.postprocess(data=df, synth=synth, test=test_factual, response=y_col, \
            inverse_transform=dataset.inverse_transform, cutoff=0.5)

        timing = time.time() - start

        mcce.results_sparse['time (seconds)'] = timing

        results.append([mcce.results_sparse.L0.mean(), mcce.results_sparse.L2.mean(), mcce.results_sparse.feasibility.mean(), mcce.results_sparse.violation.mean(), mcce.results_sparse.shape[0], timing, n, s])
    else:
        for s in range(5):
            print(s)

            dim = dataset.df.shape[0]
            
            random.seed(s)
            rows = random.sample(possible_train_indices, n)
            rows = np.sort(rows)

            positives = (df.loc[rows]).copy()
            positives["y"] = predict_label(ml_model, positives)
            positives = positives[positives["y"] == 1]
            positives = positives.drop("y", axis="columns")

            positives = dataset.inverse_transform(positives)
            test_factual_inverse = dataset.inverse_transform(test_factual)
            test_factual_inverse.index.name = 'test'

            start = time.time()

            synth = pd.merge(test_factual_inverse.reset_index()[dataset.immutables + ['test']], positives, on = dataset.immutables).set_index(['test']) # 'train',
            synth = dataset.transform(synth) # go from normal to one-hot encoded

            mcce = MCCE(fixed_features=fixed_features,\
                fixed_features_encoded=fixed_features_encoded,
                    continuous=dataset.continuous, categorical=dataset.categorical,\
                        model=ml_model, seed=1)

            mcce.fit(df.drop(dataset.target, axis=1), dtypes)

            mcce.postprocess(data=df, synth=synth, test=test_factual, response=y_col, \
                inverse_transform=dataset.inverse_transform, cutoff=0.5)

            timing = time.time() - start

            mcce.results_sparse['time (seconds)'] = timing

            results.append([mcce.results_sparse.L0.mean(), mcce.results_sparse.L2.mean(), mcce.results_sparse.feasibility.mean(), mcce.results_sparse.violation.mean(), mcce.results_sparse.shape[0], timing, n, s])

results2 = pd.DataFrame(results, columns=['L0', 'L2', 'feasibility', 'violation', 'NCE', 'timing', 'Ntest', 'seed'])

results2.to_csv(os.path.join(PATH, f"{data_name}_mcce_light_n_{n_test}.csv"))
