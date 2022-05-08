import time
import pandas as pd

from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier

from carla.data.catalog import OnlineCatalog
from carla.models.catalog import MLModelCatalog
from carla.models.negative_instances import predict_negative_instances
import carla.recourse_methods.catalog as recourse_catalog
from carla import MLModel

from mcce import MCCE

data_name = "adult"
# data_name = 'give_me_some_credit'
K = 10000
n_test = 100
results_all = None

dataset = OnlineCatalog(data_name)

class RandomForestModel(MLModel):
    """The default way of implementing RandomForest from sklearn
    https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html"""

    def __init__(self, data):
        super().__init__(data)

        # get preprocessed data
        df_train = self.data.df_train
        df_test = self.data.df_test
        
        encoded_features = list(self.data.encoder.get_feature_names(self.data.categorical))
        
        x_train = df_train[self.data.continuous + encoded_features]
        y_train = df_train[self.data.target]
        x_test = df_test[self.data.continuous + encoded_features]
        y_test = df_test[self.data.target]

        # print(x_train)

        self._feature_input_order = self.data.continuous + encoded_features

        param = {
            "max_depth": None,  # determines how deep the tree can go
            "n_estimators": 200,
            "min_samples_split": 3 # number of features to consider at each split
        }
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
        # The ML framework the model was trained on
        return "xgboost"

    @property
    def raw_model(self):
        # The black-box model object
        return self._mymodel

    @property
    def tree_iterator(self):
        # make a copy of the trees, else feature names are not saved
        booster_it = [booster for booster in self.raw_model.get_booster()]
        # set the feature names
        for booster in booster_it:
            booster.feature_names = self.feature_input_order
        return booster_it

    # The predict function outputs
    # the continuous prediction of the model
    def predict(self, x):
        return self._mymodel.predict(self.get_ordered_features(x))

    # The predict_proba method outputs
    # the prediction as class probabilities
    def predict_proba(self, x):
        # print(self.get_ordered_features(x))
        return self._mymodel.predict_proba(self.get_ordered_features(x))

ml_model = RandomForestModel(dataset)

pred = ml_model.predict_proba(dataset.df_test)
pred = [row[1] for row in pred]
fpr, tpr, thresholds = metrics.roc_curve(dataset.df_test[dataset.target], pred, pos_label=1)
print(metrics.auc(fpr, tpr))

factuals = predict_negative_instances(ml_model, dataset.df)
test_factual = factuals.iloc[:n_test]

y_col = dataset.target
features_and_response = dataset.df.columns
cont_feat = dataset.continuous
cat_feat = [x for x in features_and_response if x not in cont_feat] #  these have new names since encode_normalize_order_factuals()

if data_name == 'adult': 
    fixed_features = ['age', 'sex_Male']
elif data_name == 'give_me_some_credit':
    fixed_features = ['age']
elif data_name == 'compas':
    fixed_features = ['age', 'sex_Male', 'race_Other']

#  Create dtypes for MCCE()
dtypes = dict([(x, "float") for x in cont_feat])
for x in cat_feat:
    dtypes[x] = "category"
df = (dataset.df).astype(dtypes)


start = time.time()

# (3) Fit MCCE object
print("Fitting MCCE model...")
mcce = MCCE(fixed_features=fixed_features, immutables=['age', 'sex'], \
    model=ml_model, seed=1, continuous=dataset.continuous, categorical=dataset.categorical)
mcce.fit(df.drop(y_col, axis=1), dtypes)

print("Generating counterfactuals with MCCE...")
synth_df = mcce.generate(test_factual.drop(y_col, axis=1), k=K)
mcce.postprocess(data=df, synth=synth_df, test=test_factual, \
    response=y_col, inverse_transform=dataset.inverse_transform, cutoff=0.5)

timing = time.time() - start
print(f"timing: {timing}")

mcce.results_sparse['time (seconds)'] = timing

orig_preds = ml_model.predict_proba(test_factual)
new_preds = []
for x in orig_preds:
    new_preds.append(x[1])

test_inverse = dataset.inverse_transform(test_factual)
test_inverse['pred'] = new_preds

test_inverse.to_csv(f"/nr/samba/user/anr/pkg/MCCE_Python/Results/{data_name}_tree_model_n_{n_test}_inverse_transform.csv")

orig_preds = ml_model.predict_proba(mcce.results_sparse)
new_preds = []
for x in orig_preds:
    new_preds.append(x[1])

mcce_inverse = dataset.inverse_transform(mcce.results_sparse)
mcce_inverse['pred'] = new_preds

mcce_inverse.to_csv(f"/nr/samba/user/anr/pkg/MCCE_Python/Results/{data_name}_mcce_results_tree_model_k_{K}_n_{n_test}_inverse_transform.csv")