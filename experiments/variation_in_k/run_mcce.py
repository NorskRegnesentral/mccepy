import os
import argparse
import time
import re
import numpy as np
import warnings
warnings.filterwarnings('ignore')

import torch
torch.manual_seed(0)

from carla.data.catalog import OnlineCatalog
from carla.models.catalog import MLModelCatalog
from carla.models.negative_instances import predict_negative_instances

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
    default="adult",
    help="Datasets for experiment. Options are adult, give_me_some_credit, and compas.",
)
parser.add_argument(
    "-n",
    "--number_of_samples",
    type=int,
    default=1000,
    help="Number of test observations to generate counterfactuals for.",
)
parser.add_argument(
    "-ft",
    "--force_train",
    action='store_true',  # default is False
    help="Whether to train the prediction model from scratch or not. Default will not train.",
)
args = parser.parse_args()

n_test = args.number_of_samples
data_name = args.dataset
force_train = args.force_train
path = args.path
seed = 1

print(f"Load {data_name} data set")
dataset = OnlineCatalog(data_name)

print("Load/train predictive model")
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

for k in [10, 50, 100, 1000, 5000, 10000, 25000]:
    print(f"K: {k}")
    
    time_k_start = time.time()

    cfs = mcce.generate(test_factual.drop(dataset_mcce.target, axis=1), k=k)
    time_generate = time.time()

    mcce.postprocess(cfs, test_factual, cutoff=0.5, higher_cardinality=False)
    time_postprocess = time.time()

    results = mcce.results_sparse

    results['time (seconds)'] = (time_fit - start) + (time_generate - time_k_start) + (time_postprocess - time_generate)
    results['fit (seconds)'] = time_fit - start
    results['generate (seconds)'] = time_generate - time_k_start
    results['postprocess (seconds)'] = time_postprocess - time_generate

    results['data'] = data_name
    results['method'] = 'mcce'
    results['n_test'] = n_test
    results['k'] = k

    results_copy = results.copy()
    results_copy[ml_model.feature_input_order] = results_copy[ml_model.feature_input_order].astype(float)

    results['prediction'] = ml_model.predict(results_copy)

    cols = ['data', 'method', 'n_test', 'k'] + dataset_mcce.categorical_encoded + dataset_mcce.continuous + ['time (seconds)', 'fit (seconds)', 'generate (seconds)', 'postprocess (seconds)']
    results.sort_index(inplace=True)

    path_all = os.path.join(path, f"{data_name}_mcce_results_k_several_n_{n_test}_{device}.csv")
    if(os.path.exists(path_all)):
        results[cols].to_csv(path_all, mode='a', header=False)
    else:
        results[cols].to_csv(path_all)


