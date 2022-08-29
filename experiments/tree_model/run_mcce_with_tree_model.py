import os
import time
import argparse
import warnings
warnings.filterwarnings('ignore')

import numpy as np

from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier

from carla.data.catalog import OnlineCatalog
from carla.models.negative_instances import predict_negative_instances
from carla import MLModel

from mcce.mcce import MCCE

# must do pip install . in CARLA_version_2 directory

# Fit predictive model that takes into account MLModel (a CARLA class!)
class RandomForestModel(MLModel):
    """The default way of implementing RandomForest from sklearn
    https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html"""

    def __init__(self, data):
        super().__init__(data)

        # get preprocessed data
        df_train = self.data.df_train
        
        encoded_features = list(self.data.encoder.get_feature_names(self.data.categorical))
        
        x_train = df_train[self.data.continuous + encoded_features]
        y_train = df_train[self.data.target]

        self._feature_input_order = self.data.continuous + encoded_features

        param = {
            "max_depth": None,  # determines how deep the tree can go
            "n_estimators": 200,
            "min_samples_split": 3 # number of features to consider at each split
        }
        np.random.seed(1)  # important to use np and not random with sklearn!
        self._mymodel = RandomForestClassifier(**param)
        self._mymodel.fit(
                x_train,
                y_train,
            )

    @property
    def feature_input_order(self):
        # List of the feature order the ml model was trained on
        return self._feature_input_order

    @property
    def backend(self):
        return "xgboost"

    @property
    def raw_model(self):
        return self._mymodel

    def predict(self, x):
        return self._mymodel.predict(self.get_ordered_features(x))

    def predict_proba(self, x):
        return self._mymodel.predict_proba(self.get_ordered_features(x))


parser = argparse.ArgumentParser(description="Fit MCCE when the underlying predictive function is a decision tree.")
parser.add_argument(
    "-p",
    "--path",
    type=str,
    default="Final_results_new",
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

args = parser.parse_args()

data_name = args.dataset
n_test = args.number_of_samples
K = args.K
path = args.path

# Load data set from CARLA
dataset = OnlineCatalog(data_name)

ml_model = RandomForestModel(dataset)

pred = ml_model.predict_proba(dataset.df_test)
pred = [row[1] for row in pred]
fpr, tpr, thresholds = metrics.roc_curve(dataset.df_test[dataset.target], pred, pos_label=1)
print(f"AUC of predictive model on out-of-sample test set: {round(metrics.auc(fpr, tpr), 2)}")

print("Find factuals to generate counterfactuals for")
factuals = predict_negative_instances(ml_model, dataset.df)
test_factual = factuals.iloc[:n_test]

print("Prepare data for MCCE")
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
mcce.postprocess(cfs=synth_df, fact=test_factual, cutoff=0.5, higher_cardinality=False)
time_postprocess = time.time()
end = time.time() - start

# Timings
mcce.results_sparse['time (seconds)'] = time.time() - start
mcce.results_sparse['fit (seconds)'] = time_fit - start
mcce.results_sparse['generate (seconds)'] = time_generate - time_fit
mcce.results_sparse['postprocess (seconds)'] = time_postprocess - time_generate

results = mcce.results_sparse
results['data'] = 'adult'
results['method'] = 'mcce'
results[y_col] = test_factual[y_col]

print("Save results")
cols = ['data', 'method'] + cat_feat_encoded.tolist() + cont_feat + [y_col] + ['time (seconds)']
results.sort_index(inplace=True)

results[cols].to_csv(os.path.join(path, f"{data_name}_mcce_results_tree_model_k_{K}_n_{n_test}.csv"))
