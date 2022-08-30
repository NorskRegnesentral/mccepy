import warnings
warnings.filterwarnings('ignore')

import pandas as pd
pd.set_option('display.max_columns', None)

import torch
torch.manual_seed(0)

from carla.data.catalog import CsvCatalog
from carla.models.catalog import MLModelCatalog
from carla.models.negative_instances import predict_negative_instances

import os
import sys
import argparse

parser = argparse.ArgumentParser(description="Print counterfactual examples generated with MCCE.")
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
immutable = ["age", "sex"]

train_path = "Data/adult.data"
test_path = "Data/adult.test"
train = pd.read_csv(train_path, sep=", ", header=None, names=['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income'])
test = pd.read_csv(test_path, skiprows=1, sep=", ", header=None, names=['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income'])
df = pd.concat([train, test], axis=0, ignore_index=True)

dataset = CsvCatalog(file_path="Data/train_not_normalized_data_from_carla.csv",
                     continuous=continuous,
                     categorical=categorical,
                     immutables=immutable,
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
test_factual = factuals.iloc[:100]

print(f"Printing examples")

try:
    cfs = pd.read_csv(os.path.join(path, f"adult_mcce_results_higher_cardinality_k_{k}_n_{n_test}.csv"), index_col=0)
except:
    sys.exit(f"No MCCE results saved for k {k} and n_test {n_test} in {path}")

df_cfs = cfs.drop(['method', 'data'], axis=1)
df_cfs.sort_index(inplace=True)

cfs = dataset.inverse_transform(cfs)
test_factual = dataset.inverse_transform(test_factual)

cfs['method'] = 'MCCE'
test_factual['method'] = 'Original'

temp = pd.concat([test_factual, cfs], axis=0)

cols = ['method', 'age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', \
       'hours-per-week', 'marital-status', 'native-country', \
       'occupation', 'race', 'relationship', 'sex', 'workclass']

to_write = temp[cols].loc[[1, 31]].sort_index() # , 122, 124
to_write.columns = cols

for feature in ["workclass", "marital-status", "occupation", "relationship", "sex", "race", "native-country"]:
    d = df.groupby([feature]).size().sort_values(ascending=False)
    for i, ind in enumerate(d):
        if i <= 3:
            d[i] = i
        else:
            d[i] = 3
    mapping = d.to_dict()
    dct = {v: k for k, v in mapping.items()}

    to_write[feature] = [dct[item] for item in to_write[feature]]

feature = 'marital-status'
dct = {'Married-civ-spouse': 'MCS', 'Never-married': 'NM', 'Divorced': 'D', 'Married-AF-spouse': 'MAFS'}
to_write[feature] = [dct[item] for item in to_write[feature]]

feature = 'native-country'
dct = {'United-States': 'US', 'Holand-Netherlands': 'HS'}
to_write[feature] = [dct[item] for item in to_write[feature]]

feature = 'occupation'
dct = {'Exec-managerial': 'EM', 'Armed-Forces': 'AF', 'Prof-specialty': 'P'}
to_write[feature] = [dct[item] for item in to_write[feature]]

feature = 'race'
dct = {'White': 'W', 'Black': 'B', 'Asian-Pac-Islander': 'API'}
to_write[feature] = [dct[item] for item in to_write[feature]]

feature = 'relationship'
dct = {'Husband': 'H', 'Own-child': 'OC'}
to_write[feature] = [dct[item] for item in to_write[feature]]

feature = 'sex'
dct = {'Male': 'M'}
to_write[feature] = [dct[item] for item in to_write[feature]]

feature = 'workclass'
dct = {'Self-emp-not-inc': 'SENI', 'Private': 'P', 'Never-worked': 'NW'}
to_write[feature] = [dct[item] for item in to_write[feature]]

print(to_write.round(0).to_string())