import sys
import yaml
import pandas as pd

import numpy as np
from carla import DataCatalog, MLModelCatalog
from carla.recourse_methods import GrowingSpheres
from carla.models.negative_instances import predict_negative_instances
from carla.recourse_methods import *

from carla.evaluation.distances import get_distances
from carla.evaluation.nearest_neighbours import yNN
from carla.evaluation.redundancy import redundancy
from carla.evaluation.success_rate import success_rate
from carla.evaluation.violations import constraint_violation

from mcce import MCCE

n_test = 3
n_train = 1000
seed = 1
K = 1000
use_test_when_training = True

# Use CARLA to load dataset and predictive model
print("Loading data from Carla...")
data_name = "adult"
dataset = DataCatalog(data_name)
y_col = dataset.target
x_col = dataset.raw.columns.to_list()
x_col.remove(y_col)

# (1) Load predictive model and predict probabilities

model = MLModelCatalog(dataset, "ann")
gs = GrowingSpheres(model)

# This is the data used to train the model and also generate counterfactuals
df = gs.encode_normalize_order_factuals(dataset.raw, with_target=True) # includes response

# (2) Find unhappy customers and choose which ones to make counterfactuals for
 
factuals = predict_negative_instances(model, dataset)
factuals = factuals.iloc[: n_test]
factuals = factuals.reset_index(drop=True) # not normalized

enc_norm_factuals = gs.encode_normalize_order_factuals(factuals, with_target=True)

test = enc_norm_factuals.copy()

cont_feat = dataset.continous
cat_feat = [x for x in df.columns if x not in cont_feat] #  these have new names since encode_normalize_order_factuals()
fixed_features = ['age', 'sex_Male']

#  Create dtypes for MCCE()
dtypes = dict([(x, "float") for x in cont_feat])
for x in cat_feat:
    dtypes[x] = "category"
df = df.astype(dtypes)


# (3) Fit MCCE object
print("Fitting MCCE model...")
mcce = MCCE(fixed_features=fixed_features, model=model)
mcce.fit(df.drop(y_col, axis=1), dtypes)
print("Generating counterfactuals with MCCE...")
synth_df = mcce.generate(test.drop(y_col, axis=1), k=500)

# (4) Postprocess generated counterfactuals
print("Postprocessing counterfactuals with MCCE...")
mcce.postprocess(df, synth_df, test, y_col, scaler=model.scaler, cutoff=0.5)

# (5) Print results 

pd.set_option('display.max_columns', None)

results_all = mcce.results.copy()
results = mcce.results_sparse.copy()
results_all[cont_feat] = model.scaler.inverse_transform(results_all[cont_feat])
results[cont_feat] = model.scaler.inverse_transform(results[cont_feat])
# print(results)

# test[cont_feat] = model.scaler.inverse_transform(test[cont_feat])

counterfactuals = mcce.results_sparse.copy()
counterfactuals = counterfactuals[cont_feat + cat_feat]#.drop(y_col, axis=1)
print(counterfactuals)

arr_f = enc_norm_factuals.drop(y_col, axis=1).to_numpy()
arr_cf = counterfactuals.drop(y_col, axis=1).to_numpy()
distance = get_distances(arr_f, arr_cf) # n_test values per data.frame
model.use_pipeline = False # if False, will not normalize features
ynn = yNN(counterfactuals, gs, model, 5) # 1 value per data.frame
redund = redundancy(enc_norm_factuals, counterfactuals, model) # n_test values per data.frame
success = success_rate(counterfactuals) # 1 value per data.frame
violation = constraint_violation(model, counterfactuals, factuals) # n_test values per data.frame

model.use_pipeline = True
results_mcce = counterfactuals.copy()
results_mcce[cont_feat] = model.scaler.inverse_transform(results_mcce[cont_feat])

distance = pd.DataFrame(distance, columns=['L0', 'L1', 'L2', "L_inf"])
results_mcce['L0'] = distance['L0']
results_mcce['L2'] = distance['L2']
results_mcce['yNN'] = ynn
results_mcce['feasibility'] = 0
results_mcce['feasibility'] = 0
results_mcce['redundancy'] = [item for sublist in redund for item in sublist]
results_mcce['success'] = success
results_mcce['violation'] = [item for sublist in violation for item in sublist]
print(results_mcce)



# print("Original test observation...\n")
# print(test[~test.index.duplicated(keep='first')])

# print("Best counterfactuals for each test observation using MCCE...\n")
# print(results)

# ------------------ DiCE ---------------

# def load_setup():
#     with open("experimental_setup.yaml", "r") as f:
#         setup_catalog = yaml.safe_load(f)

#     return setup_catalog["recourse_methods"]

# rm = 'dice'
# setup = load_setup()
# hyperparams = setup[rm]["hyperparams"]
# recourse_method = Dice(model, hyperparams)
# model.use_pipeline = True
# counterfactuals = recourse_method.get_counterfactuals(factuals)
# # print(counterfactuals)

# print("Best counterfactuals for each test observation using DiCE and CARLA...\n")

# arr_f = enc_norm_factuals.drop(y_col, axis=1).to_numpy()
# arr_cf = counterfactuals.drop(y_col, axis=1).to_numpy()
# distance = get_distances(arr_f, arr_cf) # n_test values per data.frame

# model.use_pipeline = False # if False, will not normalize features
# ynn = yNN(counterfactuals, recourse_method, model, 5) # 1 value per data.frame
# redund = redundancy(enc_norm_factuals, counterfactuals, model) # n_test values per data.frame
# success = success_rate(counterfactuals) # 1 value per data.frame
# violation = constraint_violation(model, counterfactuals, factuals) # n_test values per data.frame

# results_dice = counterfactuals.copy()
# results_dice[cont_feat] = model.scaler.inverse_transform(results_dice[cont_feat])

# distance = pd.DataFrame(distance, columns=['L0', 'L1', 'L2', "L_inf"])
# results_dice['L0'] = distance['L0']
# results_dice['L2'] = distance['L2']
# results_dice['yNN'] = ynn
# results_dice['feasibility'] = 0
# results_dice['feasibility'] = 0
# results_dice['redundancy'] = [item for sublist in redund for item in sublist]
# results_dice['success'] = success
# results_dice['violation'] = [item for sublist in violation for item in sublist]
# print(results_dice)



# --------------------------------------





