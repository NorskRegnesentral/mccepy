import os
import argparse
import time
import torch
import re
import numpy as np
import pandas as pd

from carla.data.catalog import CsvCatalog
from carla.models.catalog import MLModelCatalog
from carla.models.negative_instances import predict_negative_instances

from sklearn import preprocessing
from mcce.mcce import MCCE

class DatasetMCCE():
    def __init__(self, 
                 immutables, 
                 target,
                 categorical,
                 categorical_encoded,
                 immutables_encoded,
                 continuous,
                 feature_order,
                 encoder,
                 scaler,
                 inverse_transform,
                 ):
        
        self.immutables = immutables
        self.target = target
        self.categorical = categorical
        self.categorical_encoded = categorical_encoded
        self.immutables_encoded = immutables_encoded
        self.continuous = continuous
        self.feature_order = feature_order
        self.encoder = encoder
        self.scaler = scaler
        self.inverse_transform = inverse_transform
    
device = "cuda" if torch.cuda.is_available() else "cpu"

parser = argparse.ArgumentParser(description="Fit MCCE when categorical features have more than two levels.")
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
    default=1000,
    help="Number of test observations to generate counterfactuals for.",
)
parser.add_argument(
    "-k",
    "--k",
    type=int,
    default=1000,
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
force_train = args.force_train
path = args.path
k = args.k
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

print("Processing raw data set")
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
df.to_csv("Data/adult_data.csv", index=False)

print("Read in processed data using CARLA")
continuous = ["age", "fnlwgt", "education-num", "capital-gain", "hours-per-week", "capital-loss"]
categorical = ["marital-status", "native-country", "occupation", "race", "relationship", "sex", "workclass"]
immutables = ["age", "sex"]

encoding_method = preprocessing.OneHotEncoder(drop="first", sparse=False)
dataset = CsvCatalog(file_path="Data/adult_data.csv",
                     continuous=continuous,
                     categorical=categorical,
                     immutables=immutables,
                     target='income',
                     encoding_method=encoding_method
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

# Define new target feature to use while training
target = dataset.target
new_target = target + '_High'

categorical = dataset.categorical + [dataset.target]
categorical_encoded = dataset.encoder.get_feature_names(dataset.categorical).tolist() + [new_target]
immutables = dataset.immutables + [dataset.target]

df = dataset.df

# Change prediction from numeric to categorical
pred = ml_model.predict(df)
pred = [row[0] for row in pred]
df[new_target] = [1 if row >= 0.5 else 0 for row in pred]

immutable_features_encoded = []
for immutable in immutables:
    if immutable in categorical:
        for new_col in categorical_encoded:
            match = re.search(immutable, new_col)
            if match:
                immutable_features_encoded.append(new_col)
    else:
        immutable_features_encoded.append(immutable)

# Create new dataset object
dataset_mcce = DatasetMCCE(immutables=immutables, 
                           target=dataset.target,
                           categorical=dataset.categorical,
                           categorical_encoded=categorical_encoded,
                           immutables_encoded=immutable_features_encoded,
                           continuous=dataset.continuous,
                           feature_order=ml_model.feature_input_order,
                           encoder=dataset.encoder,
                           scaler=dataset.scaler,
                           inverse_transform=dataset.inverse_transform
                           )

                       
#  Create dtypes for MCCE
dtypes = dict([(x, "float") for x in dataset_mcce.continuous])
for x in dataset_mcce.categorical_encoded:
    dtypes[x] = "category"
df = (df).astype(dtypes)

print("Find unhappy customers and choose which ones to make counterfactuals for")
factuals = predict_negative_instances(ml_model, df)
test_factual = factuals.iloc[:n_test]

# Define new value for predicted target in test factual
test_factual[new_target] = np.ones(test_factual.shape[0])
test_factual[new_target] = test_factual[new_target].astype("category")

print("Fit trees")
start = time.time()
mcce = MCCE(dataset=dataset_mcce,
            model=ml_model)

mcce.fit(df.drop(dataset_mcce.target, axis=1), dtypes)
time_fit = time.time()

print("Sample observations from tree nodes")
cfs = mcce.generate(test_factual.drop(dataset_mcce.target, axis=1), k=k)
time_generate = time.time()

print("Process sampled observations")
mcce.postprocess(cfs, test_factual, cutoff=0.5, higher_cardinality=False)
time_postprocess = time.time()

results = mcce.results_sparse

results['time (seconds)'] = (time_fit - start) + (time_generate - time_fit) + (time_postprocess - time_generate)
results['fit (seconds)'] = time_fit - start
results['generate (seconds)'] = time_generate - time_fit
results['postprocess (seconds)'] = time_postprocess - time_generate

results['data'] = "adult"
results['method'] = 'mcce'
results['n_test'] = n_test
results['k'] = k

cols = ['data', 'method', 'n_test', 'k'] + dataset_mcce.categorical_encoded + dataset_mcce.continuous + ['time (seconds)', 'fit (seconds)', 'generate (seconds)', 'postprocess (seconds)']
results.sort_index(inplace=True)

results[cols].to_csv(os.path.join(path, f"adult_mcce_results_higher_cardinality_k_{k}_n_{n_test}_{device}.csv"))