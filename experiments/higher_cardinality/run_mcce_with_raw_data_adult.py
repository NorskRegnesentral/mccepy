import torch
import time
import os

from sklearn.neighbors import NearestNeighbors
from mcce import MCCE
from carla.data.catalog import CsvCatalog
from carla.models.catalog import MLModelCatalog
from carla.models.negative_instances import predict_negative_instances

import numpy as np
import pandas as pd

PATH = "../../Results_test/experiment2/"

n_test = 10
K = 100

# Load raw data

train_path = "Data/adult.data"
test_path = "Data/adult.test"
train = pd.read_csv(train_path, sep=", ", header=None, names=['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income'])
test = pd.read_csv(test_path, skiprows=1, sep=", ", header=None, names=['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income'])

df = pd.concat([train, test], axis=0, ignore_index=True)

df = df.drop(['education'], axis=1)

# Processing

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

df.to_csv("Data/train_not_normalized_data_from_carla.csv", index=False)

## Read in data using CARLA package

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

## Fit predictive model

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
force_train=True, # don't forget to add this or it might load an older model from disk
)

## Find factuals to generate counterfactuals for

factuals = predict_negative_instances(ml_model, dataset.df)
test_factual = factuals.iloc[:n_test]

## Prepare data for MCCE
factuals = predict_negative_instances(ml_model, dataset.df)
test_factual = factuals.iloc[:n_test]

y_col = dataset.target
cont_feat = dataset.continuous

cat_feat = dataset.categorical
cat_feat_encoded = dataset.encoder.get_feature_names(dataset.categorical)

fixed_features_encoded = ['age', 'sex_1']
fixed_features = ['age', 'sex']

#  Create dtypes for MCCE()
dtypes = dict([(x, "float") for x in cont_feat])
for x in cat_feat_encoded:
    dtypes[x] = "category"
df = (dataset.df).astype(dtypes)


start = time.time()
mcce = MCCE(fixed_features=fixed_features,\
    fixed_features_encoded=fixed_features_encoded,
        continuous=dataset.continuous, categorical=dataset.categorical,\
            model=ml_model, seed=1)

mcce.fit(df.drop(dataset.target, axis=1), dtypes)

synth_df = mcce.generate(test_factual.drop(dataset.target, axis=1), k=100)


# ----------------------------------------------------------------
## TODO: Implement this in MCCE package

# ------------- postprocess() -----------------

data = df.copy()
synth = synth_df.copy()
test = test_factual.copy()
response = y_col
inverse_transform = dataset.inverse_transform
cutoff = 0.5

# Predict response of generated data
synth[response] = ml_model.predict(synth)
synth_positive = synth[synth[response]>=cutoff] # drop negative responses


# Duplicate original test observations N times where N is number of positive counterfactuals
n_counterfactuals = synth_positive.groupby(synth_positive.index).size()
n_counterfactuals = pd.DataFrame(n_counterfactuals, columns = ['N'])

test_repeated = test.copy()

test_repeated = test_repeated.join(n_counterfactuals)
test_repeated.dropna(inplace = True)

test_repeated = test_repeated.reindex(test_repeated.index.repeat(test_repeated.N))
test_repeated.drop(['N'], axis=1, inplace=True)

# --------------- calculate_metrics() -----------------

synth = synth_positive.copy()
test = test_repeated.copy()

features = synth.columns.to_list()
features.remove(response)

synth_inverse_transform = dataset.inverse_transform(synth)
test_inverse_transform = dataset.inverse_transform(test)
features_inverse_transform = synth_inverse_transform.columns.to_list()
features_inverse_transform.remove(response)

synth_metrics = synth.copy()
synth.sort_index(inplace=True)

cols = data.columns
cols.drop(response)

# 1) Distance: Sparsity and Euclidean distance

# factual = test_inverse_transform[features_inverse_transform].sort_index().to_numpy()
# counterfactuals = synth_inverse_transform[features_inverse_transform].sort_index().to_numpy()

# We don't transform the continuous features or the Euclidean distance will be wrong 
cfs_continuous = synth[dataset.continuous].sort_index().to_numpy()
cfs_categorical = synth_inverse_transform[dataset.categorical].sort_index().to_numpy()

factual_continuous = test[dataset.continuous].sort_index().to_numpy()
factual_categorical = test_inverse_transform[dataset.categorical].sort_index().to_numpy()


delta_cont = factual_continuous - cfs_continuous
delta_cat = factual_categorical - cfs_categorical

delta_cat = np.where(np.abs(delta_cat) > 0, 1, 0)

delta = np.concatenate((delta_cont, delta_cat), axis=1)
d1 = np.sum(np.invert(np.isclose(delta, np.zeros_like(delta), atol=1e-5)), axis=1, dtype=float).tolist() # sparsity
d2 = np.sum(np.abs(delta), axis=1, dtype=float).tolist() # manhatten distance
d3 = np.sum(np.square(np.abs(delta)), axis=1, dtype=np.float).tolist() # euclidean distance

synth_metrics['L0'] = d1
synth_metrics['L1'] = d2
synth_metrics['L2'] = d3

# 2) Feasibility 

feas_results = []
nbrs = NearestNeighbors(n_neighbors=5).fit(data[cols].values)

for i, row in synth[cols].iterrows():
    knn = nbrs.kneighbors(row.values.reshape((1, -1)), 5, return_distance=True)[0]
    
    feas_results.append(np.mean(knn))

synth_metrics['feasibility'] = feas_results

# 3) Success
synth_metrics['success'] = 1

# 4) Violation

def intersection(lst1, lst2):
    return list(set(lst1) & set(lst2))

df_decoded_cfs = dataset.inverse_transform(synth)

df_factuals = dataset.inverse_transform(test)

# check continuous using np.isclose to allow for very small numerical differences
cfs_continuous_immutable = df_decoded_cfs[
    intersection(dataset.continuous, fixed_features)
]

factual_continuous_immutable = df_factuals[
    intersection(dataset.continuous, dataset.immutables)
]

continuous_violations = np.invert(
    np.isclose(cfs_continuous_immutable, factual_continuous_immutable)
)
continuous_violations = np.sum(continuous_violations, axis=1).reshape(
    (-1, 1)
) 

# check categorical by boolean comparison
cfs_categorical_immutable = df_decoded_cfs[
    intersection(dataset.categorical, dataset.immutables)
]
# print(cfs_categorical_immutable)
factual_categorical_immutable = df_factuals[
    intersection(dataset.categorical, dataset.immutables)
]


cfs_categorical_immutable.sort_index(inplace=True)
factual_categorical_immutable.sort_index(inplace=True)
cfs_categorical_immutable.index.name = None

categorical_violations = cfs_categorical_immutable != factual_categorical_immutable
categorical_violations = np.sum(categorical_violations.values, axis=1).reshape(
    (-1, 1)
)

synth_metrics['violation'] = continuous_violations + categorical_violations

results = synth_metrics.copy()
results_sparse = pd.DataFrame(columns=results.columns)

for idx in list(set(results.index)):
    idx_df = results.loc[idx]
    if(isinstance(idx_df, pd.DataFrame)): # If you have multiple rows
        sparse = min(idx_df.L0) # 1) find least # features changed
        sparse_df = idx_df[idx_df.L0 == sparse] 
        closest = min(sparse_df.L2) # find smallest Gower distance
        close_df = sparse_df[sparse_df.L2 == closest]

        if(close_df.shape[0]>1):
            highest_feasibility = max(close_df.feasibility) #  3) find most feasible
            close_df = close_df[close_df.feasibility == highest_feasibility].head(1)

    else: # if you have only one row - return that row
        close_df = idx_df.to_frame().T
        
    results_sparse = pd.concat([results_sparse, close_df], axis=0)

timing = time.time() - start
print(f"timing: {timing}")
results_sparse['time (seconds)'] = timing

# ----------------------------------------------------------------

## Save results

results_sparse.to_csv(os.path.join(PATH, f"adult_mcce_results_raw_data_k_{K}_n_{n_test}.csv"))

## Save the original factuals

orig_preds = ml_model.predict_proba(results_sparse)
new_preds = []
for x in orig_preds:
    new_preds.append(x[1])

results_inverse = dataset.inverse_transform(results_sparse)

results_inverse['pred'] = new_preds

results_inverse.to_csv(os.path.join(PATH, f"adult_mcce_results_raw_data_k_{K}_n_{n_test}_inverse_transform.csv"))

true_raw =  dataset.inverse_transform(test_factual)

orig_preds = ml_model.predict_proba(test_factual)
new_preds = []
for x in orig_preds:
    new_preds.append(x[1])

true_raw['pred'] = new_preds

true_raw.to_csv(os.path.join(PATH, f"adult_raw_data_n_{n_test}.csv"))