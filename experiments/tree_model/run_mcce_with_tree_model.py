import time
import os
import argparse

from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier

from carla.data.catalog import OnlineCatalog
from carla.models.negative_instances import predict_negative_instances
from carla import MLModel

from mcce.metrics import feasibility
from mcce.mcce import MCCE
from mcce.rf import RandomForestModel

PATH = "Final_results_new/"
# must do pip install . in CARLA_version_2 directory

parser = argparse.ArgumentParser(description="Fit MCCE with various datasets.")
parser.add_argument(
    "-d",
    "--dataset",
    nargs="*",
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
    default=10000,
    help="Number generated counterfactuals per test observation",
)


args = parser.parse_args()

data_name = args.dataset
n_test = args.number_of_samples
K = args.K

dataset = OnlineCatalog(data_name)

# Fit predictive model
# class RandomForestModel(MLModel):
#     """The default way of implementing RandomForest from sklearn
#     https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html"""

#     def __init__(self, data):
#         super().__init__(data)

#         # get preprocessed data
#         df_train = self.data.df_train
#         df_test = self.data.df_test
        
#         encoded_features = list(self.data.encoder.get_feature_names(self.data.categorical))
        
#         x_train = df_train[self.data.continuous + encoded_features]
#         y_train = df_train[self.data.target]

#         self._feature_input_order = self.data.continuous + encoded_features

#         param = {
#             "max_depth": None,  # determines how deep the tree can go
#             "n_estimators": 200,
#             "min_samples_split": 3 # number of features to consider at each split
#         }
#         self._mymodel = RandomForestClassifier(**param)
#         self._mymodel.fit(
#                 x_train,
#                 y_train,
#             )

#     @property
#     def feature_input_order(self):
#         # List of the feature order the ml model was trained on
#         return self._feature_input_order

#     @property
#     def backend(self):
#         return "xgboost"

#     @property
#     def raw_model(self):
#         return self._mymodel

#     @property
#     def tree_iterator(self):
#         # make a copy of the trees, else feature names are not saved
#         booster_it = [booster for booster in self.raw_model.get_booster()]
#         # set the feature names
#         for booster in booster_it:
#             booster.feature_names = self.feature_input_order
#         return booster_it

#     def predict(self, x):
#         return self._mymodel.predict(self.get_ordered_features(x))

#     def predict_proba(self, x):
#         return self._mymodel.predict_proba(self.get_ordered_features(x))

ml_model = RandomForestModel(dataset)

pred = ml_model.predict_proba(dataset.df_test)
pred = [row[1] for row in pred]
fpr, tpr, thresholds = metrics.roc_curve(dataset.df_test[dataset.target], pred, pos_label=1)
print(metrics.auc(fpr, tpr))

# Find factuals to generate counterfactuals for

factuals = predict_negative_instances(ml_model, dataset.df)
test_factual = factuals.iloc[:n_test]

# Prepare data for MCCE

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

# Fit MCCE method
start = time.time()

mcce = MCCE(fixed_features=fixed_features,\
        fixed_features_encoded=fixed_features_encoded,
            continuous=dataset.continuous, categorical=dataset.categorical,\
                model=ml_model, seed=1)

mcce.fit(df.drop(dataset.target, axis=1), dtypes)
time_fit = time.time()

synth_df = mcce.generate(test_factual.drop(dataset.target, axis=1), k=K)
time_generate = time.time()

mcce.postprocess(synth=synth_df, test=test_factual, response=y_col, \
    inverse_transform=dataset.inverse_transform, cutoff=0.5)

time_postprocess = time.time()
end = time.time() - start

# Feasibility 
cols = dataset.df.columns.to_list()
cols.remove(dataset.target)
mcce.results_sparse['feasibility'] = feasibility(mcce.results_sparse, dataset.df, cols)


mcce.results_sparse['time (seconds)'] = end
mcce.results_sparse['fit (seconds)'] = time_fit - start
mcce.results_sparse['generate (seconds)'] = time_generate - time_fit
mcce.results_sparse['postprocess (seconds)'] = time_postprocess - time_generate

mcce.results_sparse['distance (seconds)'] = mcce.distance_cpu_time
mcce.results_sparse['violation (seconds)'] = mcce.violation_cpu_time

# Save results 
# mcce.results_sparse.to_csv(os.path.join(PATH, f"{data_name}_mcce_results_tree_model_results_k_{K}_n_{n_test}.csv"))

# (6) Print the counterfactuals inverted to their original feature values/ranges
results = mcce.results_sparse.copy()
results['data'] = data_name
results['method'] = 'mcce'
results['prediction'] = ml_model.predict_proba(results)[:, [1]]

results = dataset.inverse_transform(results)
results.to_csv(os.path.join(PATH, f"{data_name}_mcce_results_tree_model_k_{K}_n_{n_test}_inverse_transform.csv"))

print(dataset.inverse_transform(mcce.results_sparse).loc[31])
# Get the original factual feature values

# orig_preds = ml_model.predict_proba(test_factual)
# new_preds = []
# for x in orig_preds:
#     new_preds.append(x[1])
# test_inverse = dataset.inverse_transform(test_factual)
# test_inverse['pred'] = new_preds

# Save original factual values
# test_inverse.to_csv(os.path.join(PATH, f"{data_name}_tree_model_n_{n_test}_inverse_transform.csv"))

# orig_preds = ml_model.predict_proba(mcce.results_sparse)
# new_preds = []
# for x in orig_preds:
#     new_preds.append(x[1])

# mcce_inverse = dataset.inverse_transform(mcce.results_sparse)
# mcce_inverse['pred'] = new_preds

# mcce_inverse.to_csv(os.path.join(PATH, f"{data_name}_k_{K}_n_{n_test}_inverse_transform.csv"))