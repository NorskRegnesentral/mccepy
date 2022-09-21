import os
import argparse
import time
import re
import numpy as np
import warnings
warnings.filterwarnings('ignore')

import torch
torch.manual_seed(0)

from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier

from carla.data.catalog import OnlineCatalog
from carla.models.negative_instances import predict_negative_instances
from carla import MLModel

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

# Fit predictive model that takes into account MLModel (a CARLA class)
class RandomForestModel(MLModel):
    """
    Trains a random forest model using the sklearn RandomForestClassifier method. 
    Takes as input the CARLA MLModel class. 
    
    Parameters
    ----------
    data : mcce.Data
        Object of class mcce.Data which contains the data to train the model on and a set
        of other attributions like which features are continuous, categorical, and fixed.
    
    Methods
    -------
    predict : 
        Predicts the response/target for a data set based on the fitted random forest.
    predict_proba :
        Outputs the predicted probability (between 0 and 1) for a data set based on the fitted random forest.
    get_ordered_features :
        Returns a pd.DataFrame where the features have the same ordering as the original data set. 
    """

    def __init__(self, data):
        super().__init__(data)

        # get preprocessed data
        df_train = self.data.df_train
        
        encoded_features = list(self.data.encoder.get_feature_names(self.data.categorical))
        
        x_train = df_train[self.data.continuous + encoded_features]
        y_train = df_train[self.data.target]

        self._feature_input_order = self.data.continuous + encoded_features

        param = {
            "max_depth": None,  # The maximum depth of the tree. If None, then nodes are expanded until 
                                # all leaves are pure or until all leaves contain less than min_samples_split samples.
            "n_estimators": 200, # The number of trees in the forest.
            "min_samples_split": 3 # The minimum number of samples required to split an internal node:
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
    "-k",
    "--k",
    type=int,
    default=1000,
    help="Number of observations to sample from each end node for MCCE method.",
)

args = parser.parse_args()

n_test = args.number_of_samples
data_name = args.dataset
path = args.path
k = args.k
seed = 1

# Load data set from CARLA
dataset = OnlineCatalog(data_name)

# Train random forest predictive model
ml_model = RandomForestModel(dataset)

pred = ml_model.predict_proba(dataset.df_test)
pred = [row[1] for row in pred]
fpr, tpr, thresholds = metrics.roc_curve(dataset.df_test[dataset.target], pred, pos_label=1)
print(f"AUC of predictive model on out-of-sample test set: {round(metrics.auc(fpr, tpr), 2)}")

# Define new target feature to use while training
target = dataset.target
new_target = target + '_High'

categorical = dataset.categorical + [dataset.target]
categorical_encoded = dataset.encoder.get_feature_names(dataset.categorical).tolist() + [new_target]
immutables = dataset.immutables + [dataset.target]

df = dataset.df

# Change prediction from numeric to categorical
pred = ml_model.predict(df)
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

print("Find factuals to generate counterfactuals for")
factuals = predict_negative_instances(ml_model, dataset.df)
test_factual = factuals.iloc[:n_test]

# Define new value for predicted target in test factual
test_factual[new_target] = np.ones(test_factual.shape[0])
test_factual[new_target] = test_factual[new_target].astype("category")

print("Fit trees")
start = time.time()
mcce = MCCE(dataset=dataset_mcce,
            model=ml_model)

mcce.fit(df.drop(dataset.target, axis=1), dtypes)
time_fit = time.time()

print("Sample observations from tree nodes")
cfs = mcce.generate(test_factual.drop(dataset.target, axis=1), k=k)
time_generate = time.time()

print("Process sampled observations")
mcce.postprocess(cfs, test_factual, cutoff=0.5, higher_cardinality=False)
time_postprocess = time.time()

results = mcce.results_sparse

results['time (seconds)'] = (time_fit - start) + (time_generate - time_fit) + (time_postprocess - time_generate)
results['fit (seconds)'] = time_fit - start
results['generate (seconds)'] = time_generate - time_fit
results['postprocess (seconds)'] = time_postprocess - time_generate

results['data'] = data_name
results['method'] = 'mcce'
results['n_test'] = n_test
results['k'] = k

cols = ['data', 'method', 'n_test', 'k'] + dataset_mcce.categorical_encoded + dataset_mcce.continuous + ['time (seconds)', 'fit (seconds)', 'generate (seconds)', 'postprocess (seconds)']
results.sort_index(inplace=True)

results[cols].to_csv(os.path.join(path, f"{data_name}_mcce_results_tree_model_k_{k}_n_{n_test}_{device}.csv"))
