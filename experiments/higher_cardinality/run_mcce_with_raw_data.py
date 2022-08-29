from sklearn.neighbors import NearestNeighbors

from carla.data.catalog import CsvCatalog
from carla.models.catalog import MLModelCatalog
from carla.models.negative_instances import predict_negative_instances

import torch
import time
import os
import argparse
import numpy as np
import pandas as pd

from mcce.mcce import MCCE

# must do pip install . in CARLA_version_2 directory

parser = argparse.ArgumentParser(description="Fit MCCE when categorical features have more than two levels.")
parser.add_argument(
    "-p",
    "--path",
    type=str,
    default="Final_results_new",
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
    "-K",
    "--K",
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

n_test = args.number_of_samples
K = args.K
force_train = args.force_train
path = args.path
seed = 1

print("Load raw adult data")
train_path = "Data/adult.data"
test_path = "Data/adult.test"
train = pd.read_csv(train_path, 
                    sep=", ", 
                    header=None, 
                    names=['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation', 'relationship', 
                    'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income'])

test = pd.read_csv(test_path, 
                   skiprows=1, 
                   sep=", ", 
                   header=None, 
                   names=['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 
                   'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income'])

df = pd.concat([train, test], axis=0, ignore_index=True)
df = df.drop(['education'], axis=1)

print("Processing data set")
mapping = {'>50K': '>50K', '>50K.': '>50K', '<=50K': '<=50K', '<=50K.': '<=50K'}

df['income'] = [mapping[item] for item in df['income']]

for feature in ["workclass", "marital-status", "occupation", "relationship", \
    "sex", "race", "native-country", "income"]:
    d = df.groupby([feature]).size().sort_values(ascending=False)
    for i, ind in enumerate(d):
        if i <= 3:
            d[i] = i
        else:
            d[i] = 3
    mapping = d.to_dict()
    df[feature] = [mapping[item] for item in df[feature]]

# Save processed data in the same folder
df.to_csv("Data/train_not_normalized_data_from_carla.csv", index=False)

print("Read in processed data using CARLA")
continuous = ["age", "fnlwgt", "education-num", "capital-gain", "hours-per-week", "capital-loss"]
categorical = ["marital-status", "native-country", "occupation", "race", "relationship", "sex", "workclass"]
immutable = ["age", "sex"]

dataset = CsvCatalog(file_path="Data/train_not_normalized_data_from_carla.csv",
                     continuous=continuous,
                     categorical=categorical,
                     immutables=immutable,
                     target='income',
                     encoding_method="OneHot_drop_first", # New!
                     )

print("Fit predictive model")
torch.manual_seed(0)
ml_model = MLModelCatalog(dataset, 
                          model_type="ann", 
                          load_online=False, 
                          backend="pytorch"
                          )
ml_model.train(learning_rate=0.002,
               epochs=20,
               batch_size=1024,
               hidden_size=[18, 9, 3],
               force_train=force_train,
)

print("Find factuals to generate counterfactuals for")
factuals = predict_negative_instances(ml_model, dataset.df)
test_factual = factuals.iloc[:n_test]

y_col = dataset.target
cont_feat = dataset.continuous
cat_feat = dataset.categorical
cat_feat_encoded = dataset.encoder.get_feature_names(dataset.categorical)

fixed_features_encoded = ['age', 'sex_1']
fixed_features = ['age', 'sex']

dtypes = dict([(x, "float") for x in cont_feat])
for x in cat_feat_encoded:
    dtypes[x] = "category"
df = (dataset.df).astype(dtypes)

print("Fit trees")
start = time.time()
mcce = MCCE(dataset=dataset,
            fixed_features=fixed_features,
            fixed_features_encoded=fixed_features_encoded,
            model=ml_model, 
            seed=1)

mcce.fit(df.drop(dataset.target, axis=1), dtypes)
time_fit = time.time()

print("Sample observations from tree nodes")
synth_df = mcce.generate(test_factual.drop(dataset.target, axis=1), k=K)
time_generate = time.time()

print("Process sampled observations")
mcce.postprocess(cfs=synth_df, fact=test_factual, cutoff=0.5, higher_cardinality=True)
time_postprocess = time.time()

print("Save results")
mcce.results_sparse['time (seconds)'] = time.time() - start
mcce.results_sparse['fit (seconds)'] = time_fit - start
mcce.results_sparse['generate (seconds)'] = time_generate - time_fit
mcce.results_sparse['postprocess (seconds)'] = time_postprocess - time_generate

results = mcce.results_sparse
results['data'] = 'adult'
results['method'] = 'mcce'
results[y_col] = test_factual[y_col]

cols = ['data', 'method'] + cat_feat_encoded.tolist() + cont_feat + [y_col] + ['time (seconds)']
results.sort_index(inplace=True)

results[cols].to_csv(os.path.join(path, f"adult_mcce_results_higher_cardinality_k_{K}_n_{n_test}.csv"))