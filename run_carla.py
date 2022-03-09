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

# Use CARLA to load dataset and predictive model
print("Loading data from Carla...")
data_name = "adult"
dataset = DataCatalog(data_name)
y_col = dataset.target

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


print("Original test observation...\n")
print(test[~test.index.duplicated(keep='first')])


# ------------------ DiCE ---------------

def load_setup():
    with open("experimental_setup.yaml", "r") as f:
        setup_catalog = yaml.safe_load(f)

    return setup_catalog["recourse_methods"]

rm = 'dice'
setup = load_setup()
hyperparams = setup[rm]["hyperparams"]
recourse_method = Dice(model, hyperparams)
model.use_pipeline = True
counterfactuals = recourse_method.get_counterfactuals(factuals)
# print(counterfactuals)

print("Best counterfactuals for each test observation using DiCE and CARLA...\n")

arr_f = enc_norm_factuals.drop(y_col, axis=1).to_numpy()
arr_cf = counterfactuals.drop(y_col, axis=1).to_numpy()
distance = get_distances(arr_f, arr_cf) # n_test values per data.frame

model.use_pipeline = False # if False, will not normalize features
ynn = yNN(counterfactuals, recourse_method, model, 5) # 1 value per data.frame
redund = redundancy(enc_norm_factuals, counterfactuals, model) # n_test values per data.frame
success = success_rate(counterfactuals) # 1 value per data.frame
violation = constraint_violation(model, counterfactuals, factuals) # n_test values per data.frame

results_dice = counterfactuals.copy()
results_dice[cont_feat] = model.scaler.inverse_transform(results_dice[cont_feat])

distance = pd.DataFrame(distance, columns=['L0', 'L1', 'L2', "L_inf"])
results_dice['L0'] = distance['L0']
results_dice['L2'] = distance['L2']
results_dice['yNN'] = ynn
results_dice['feasibility'] = 0
results_dice['feasibility'] = 0
results_dice['redundancy'] = [item for sublist in redund for item in sublist]
results_dice['success'] = success
results_dice['violation'] = [item for sublist in violation for item in sublist]
print(results_dice)



# --------------------------------------





