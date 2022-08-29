import os
import argparse
import time
import torch

from carla.data.catalog import OnlineCatalog
from carla.models.catalog import MLModelCatalog
from carla.models.negative_instances import predict_negative_instances

from mcce.metrics import feasibility
from mcce.mcce import MCCE

PATH = 'Final_results_new'

parser = argparse.ArgumentParser(description="Fit MCCE with various datasets.")
parser.add_argument(
    "-d",
    "--dataset",
    type=str,
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
    default=10000,  # 1000 for Compas
    help="Number generated counterfactuals per test observation",
)
parser.add_argument(
    "-ft",
    "--force_train",
    action='store_true',  # default is False
    help="Whether to train the prediction model from scratch or not. Default will not train.",
)

args = parser.parse_args()

n_test = args.number_of_samples
K = args.K # 1000 for Compas
data_name = args.dataset
force_train = args.force_train
seed = 1

print(f"Load {data_name} data set")
dataset = OnlineCatalog(data_name)

print("Load predictive model")
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

print("Find unhappy customers and choose which ones to make counterfactuals for")
factuals = predict_negative_instances(ml_model, dataset.df)
test_factual = factuals.iloc[:n_test]

y_col = dataset.target
cont_feat = dataset.continuous

cat_feat = dataset.categorical
cat_feat_encoded = dataset.encoder.get_feature_names(dataset.categorical)

if data_name == 'adult':
    fixed_features = ['age', 'sex']
    fixed_features_encoded = ['age', 'sex_Male']
elif data_name == 'give_me_some_credit':
    fixed_features = ['age']
    fixed_features_encoded = ['age']
elif data_name == 'compas':
    fixed_features = ['age', 'sex', 'race']
    fixed_features_encoded = ['age', 'sex_Male', 'race_Other']

#  Create dtypes for MCCE()
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
synth_df = mcce.generate(test_factual.drop(y_col, axis=1), k=K)
time_generate = time.time()

print("Process sampled observations")
mcce.postprocess(cfs=synth_df, fact=test_factual, cutoff=0.5)
time_postprocess = time.time()

print("Calculate timing")
mcce.results_sparse['time (seconds)'] = time.time() - start
mcce.results_sparse['fit (seconds)'] = time_fit - start
mcce.results_sparse['generate (seconds)'] = time_generate - time_fit
mcce.results_sparse['postprocess (seconds)'] = time_postprocess - time_generate

print("Save the counterfactuals")
results = mcce.results_sparse
results['data'] = data_name
results['method'] = 'mcce'
results[y_col] = test_factual[y_col]

cols = ['data', 'method'] + cat_feat_encoded.tolist() + cont_feat + [y_col] + ['time (seconds)']
results.sort_index(inplace=True)

results[cols].to_csv(os.path.join(PATH, f"{data_name}_mcce_results_k_{K}_n_{n_test}.csv"))



