import os
import time
import argparse
import warnings
warnings.filterwarnings('ignore')
import pandas as pd

import torch
import numpy as np

import os

import tensorflow as tf
from keras.layers import Dense
from keras.models import Sequential
from keras.utils import to_categorical

from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier

from carla.models.negative_instances import predict_negative_instances
from carla import MLModel
from carla.data.catalog import CsvCatalog
from carla.models.catalog import MLModelCatalog

from sklearn import preprocessing
from mcce.mcce import MCCE
# must do pip install . in CARLA_version_2 directory


device = "cuda" if torch.cuda.is_available() else "cpu"

parser = argparse.ArgumentParser(description="Fit MCCE when the underlying predictive function is a decision tree.")
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
    help="Number of test observations to generate counterfactuals for.",
)
parser.add_argument(
    "-ft",
    "--force_train",
    action='store_true',  # default is False
    help="Whether to train the prediction model from scratch or not. Default will not train.",
)

args = parser.parse_args()

data_name = 'fico'
n_test = args.number_of_samples
path = args.path
k = args.k
force_train = args.force_train

# Load data set from CARLA
print("Read in processed data using CARLA")
categorical = []
continuous = ["ExternalRiskEstimate", "MSinceOldestTradeOpen",
"MSinceMostRecentTradeOpen", "AverageMInFile", "NumSatisfactoryTrades",
"NumTrades60Ever2DerogPubRec", "NumTrades90Ever2DerogPubRec", 
"PercentTradesNeverDelq", "MSinceMostRecentDelq", "MaxDelq2PublicRecLast12M",
"MaxDelqEver", "NumTotalTrades", "NumTradesOpeninLast12M", 
"PercentInstallTrades", "MSinceMostRecentInqexcl7days", 
"NumInqLast6M", "NumInqLast6Mexcl7days", "NetFractionRevolvingBurden",
"NetFractionInstallBurden", "NumRevolvingTradesWBalance", 
"NumInstallTradesWBalance", "NumBank2NatlTradesWHighUtilization",
"PercentTradesWBalance"]
immutable = ["ExternalRiskEstimate"]

encoding_method = preprocessing.OneHotEncoder(
            drop="first", sparse=False
        )

dataset = CsvCatalog(file_path="Data/fico_data_complete.csv",
                     continuous=continuous,
                     categorical=categorical,
                     immutables=immutable,
                     target='Risk',
                     encoding_method=encoding_method
                     )

print(dataset.df)

# ml_model = AnnModel(dataset, dim_input=9, dim_hidden_layers=[20, 10, 7], num_of_classes=2, data_name="German Credit")

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
    force_train=True,
    )
pred = ml_model.predict_proba(dataset.df_test)
pred = [row[1] for row in pred]
fpr, tpr, thresholds = metrics.roc_curve(dataset.df_test[dataset.target], pred, pos_label=1)
print(f"AUC of predictive model on out-of-sample test set: {round(metrics.auc(fpr, tpr), 2)}")

print("Find factuals to generate counterfactuals for")
factuals = predict_negative_instances(ml_model, dataset.df)
print(f"Number of possible factuals: {factuals}")
test_factual = factuals.iloc[:n_test]

test_factual.to_csv(os.path.join(path, f"{data_name}_test_factuals_ann_model_k_{k}_n_{n_test}_{device}.csv"))

print("Prepare data for MCCE")
y_col = dataset.target
cont_feat = dataset.continuous

cat_feat = dataset.categorical
cat_feat_encoded = dataset.encoder.get_feature_names(dataset.categorical)

dtypes = dict([(x, "float") for x in cont_feat])
for x in cat_feat_encoded:
    dtypes[x] = "category"
df = (dataset.df).astype(dtypes)

print("Fit trees")
start = time.time()
mcce = MCCE(dataset=dataset,
            model=ml_model)

mcce.fit(df.drop(dataset.target, axis=1), dtypes)
time_fit = time.time()

print(f"K: {k}")

time_k_start = time.time()

cfs = mcce.generate(test_factual.drop(dataset.target, axis=1), k=k)
time_generate = time.time()

mcce.postprocess(cfs, test_factual, cutoff=0.5, higher_cardinality=False)
time_postprocess = time.time()

results = mcce.results_sparse

results['time (seconds)'] = (time_fit - start) + (time_generate - time_k_start) + (time_postprocess - time_generate)
results['fit (seconds)'] = time_fit - start
results['generate (seconds)'] = time_generate - time_k_start
results['postprocess (seconds)'] = time_postprocess - time_generate

# results['postprocess (predict)'] = mcce.postprocess_time['predict']
# results['postprocess (metrics)'] = mcce.postprocess_time['metrics']
# results['postprocess (loop)'] = mcce.postprocess_time['loop']

results['data'] = data_name
results['method'] = 'mcce'
results['n_test'] = n_test
results['k'] = k
# results['p'] = number_of_features

# Get the fitted tree depth for each mutable feature
# tree_depth_cols = []
# for x in dataset.feature_order:
#     if x not in dataset.immutables:
#         tree_depth_cols.append(x + "_tree_depth")
#         results[x + "_tree_depth"] = mcce.fitted_model[x].get_depth()

# results_copy = results.copy()
# results_copy[ml_model.feature_input_order] = results_copy[ml_model.feature_input_order].astype(float)
# results['prediction'] = ml_model.predict(results_copy)

cols = ['data', 'method', 'n_test', 'k'] + ['time (seconds)', 'fit (seconds)', 'generate (seconds)', 'postprocess (seconds)'] + cat_feat_encoded.tolist() + cont_feat # 'postprocess (predict)', 'postprocess (metrics)', 'postprocess (loop)'
results.sort_index(inplace=True)

path_all = os.path.join(path, f"{data_name}_mcce_results_ann_model_k_{k}_n_{n_test}_{device}.csv")

if(os.path.exists(path_all)):
    results[cols].to_csv(path_all, mode='a', header=False)
else:
    results[cols].to_csv(path_all)