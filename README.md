# mccepy
Python package to generate counterfactuals using Monte Carlo sampling of realistic counterfactual explanations.


## Installation

### From source

```bash
git clone git@github.com:NorskRegnesentral/mccepy.git
cd mccepy
pip install -r requirements.txt
python setup.py install
```

## Examples

### Use MCCE with the [CARLA](https://github.com/carla-recourse/CARLA) python package

1.  Load packages and Adult data set

```Python
from carla.data.catalog import OnlineCatalog
from carla.models.catalog import MLModelCatalog
from carla.models.negative_instances import predict_negative_instances

from mcce import MCCE

dataset = OnlineCatalog('adult')

```

2. Fit multi-layer perceptron with CARLA

```Python
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
    force_train=False, # Will not train a new model
    )

factuals = predict_negative_instances(ml_model, dataset.df)
test_factual = factuals.iloc[:5]
```

3. Specify feature names and generate counterfactuals with MCCE

```Python
y_col = dataset.target
cont_feat = dataset.continuous

cat_feat = dataset.categorical
cat_feat_encoded = dataset.encoder.get_feature_names(dataset.categorical)

fixed_features = ['age', 'sex_Male']

dtypes = dict([(x, "float") for x in cont_feat])
for x in cat_feat_encoded:
    dtypes[x] = "category"
df = (dataset.df).astype(dtypes)

```

4. Generate counterfactuals with MCCE

```Python
mcce = MCCE(fixed_features=fixed_features, continuous=dataset.continuous, categorical=dataset.categorical,\
            model=ml_model, seed=1, catalog=dataset.catalog)

mcce.fit(df.drop(y_col, axis=1), dtypes) # fit the decision trees

synth_df = mcce.generate(test_factual.drop(y_col, axis=1), k=100) # samples 100 times for each test observation

mcce.postprocess(data=df, synth=synth_df, test=test_factual, response=y_col, \
    inverse_transform=dataset.inverse_transform, cutoff=0.5) # postprocess the samples

# print best counterfactual for each test observation
print([mcce.results_sparse.L0.mean(), mcce.results_sparse.L2.mean(), mcce.results_sparse.feasibility.mean(),\
  mcce.results_sparse.violation.mean(), mcce.results_sparse.shape[0]])
    
```

### Use MCCE stand alone

In case you want to use MCCE without CARLA, use the following steps. 

### Adult dataset
Download the [US adult census dataset](https://archive.ics.uci.edu/ml/datasets/adult). Add the dataset to a local repository and save the path to the data. 


1. Initialize Data object with path to the data file, column names, feature types, response name, and a list of fixed features. 

```Python
from data import Data

feature_order = ['age', 'workclass', 'fnlwgt', 'education-num', 'marital-status', 'occupation'\
    'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'country'\
        'income']

dtypes = {"age": "float", "workclass": "category", "fnlwgt": "float", "degree": "category"\
    "education_years": "float","marital-status": "category", "occupation": "category"\
        "relationship": "category", "race": "category", \
            "sex": "category", "capital-gain": "float", "capital-loss": "float", \
                "hours": "float", "country": "category", "income": "category"}

response = 'income'

fixed_features = ['age', 'sex']
dataset = Data(path, feature_order, dtypes, response, fixed_features, "OneHot_drop_first", "MinMax")

```

2. Define a predictive model class and fit the model. 


```Python
from carla import MLModel
from sklearn.ensemble import RandomForestClassifier

class RandomForestModel(MLModel):
    """The default way of implementing RandomForest from sklearn
    https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html"""

    def __init__(self, data):
        super().__init__(data)

        # get preprocessed data
        df_train = self.data.df
        
        
        x_train = df_train[data.continuous + data.categorical_encoded]
        y_train = df_train[data.target]

        self._feature_input_order = self.data.continuous + self.data.categorical_encoded

        param = {
            "max_depth": None,  # determines how deep the tree can go
            "n_estimators": 5,
            "min_samples_split": 3 # number of features to consider at each split
        }
        self._mymodel = RandomForestClassifier(**param)
        self._mymodel.fit(
                x_train,
                y_train,
            )

    @property
    def feature_input_order(self):
        return self._feature_input_order

    @property
    def backend(self):
        return "TensorFlow"

    @property
    def raw_model(self):
        return self._mymodel

    @property
    def tree_iterator(self):
        # make a copy of the trees, else feature names are not saved
        booster_it = [booster for booster in self.raw_model.get_booster()]
        # set the feature names
        for booster in booster_it:
            booster.feature_names = self.feature_input_order
        return booster_it

    def predict(self, x):
        return self._mymodel.predict(self.get_ordered_features(x))

    def predict_proba(self, x):
        return self._mymodel.predict_proba(self.get_ordered_features(x))

ml_model = RandomForestModel(dataset)

```

3. Decide which customers to generate counterfactual explanations for

```Python
import numpy as np
preds = ml_model.predict_proba(dataset.df)[:,1]
factual_id = np.where(preds < 0.5)
factuals = dataset.df.loc[factual_id]
test_factual = factuals.iloc[:5]

```

4. Create data objects to pass to MCCE
```Python
y_col = dataset.target
cont_feat = dataset.continuous

cat_feat = dataset.categorical
cat_feat_encoded = dataset.categorical_encoded

#  Create dtypes for MCCE()
dtypes = dict([(x, "float") for x in cont_feat])
for x in cat_feat_encoded:
    dtypes[x] = "category"
df = (dataset.df).astype(dtypes)
```

4. Initialize MCCE object and generate counterfactual explanations using CART

```Python
from mcce import MCCE

from mcce import MCCE

mcce = MCCE(fixed_features=dataset.fixed_features,\
    fixed_features_encoded=dataset.fixed_features_encoded,
        continuous=dataset.continuous, categorical=dataset.categorical,\
            model=ml_model, seed=1)

mcce.fit(df.drop(dataset.target, axis=1), dtypes)

synth_df = mcce.generate(test_factual.drop(dataset.target, axis=1), k=100)

mcce.postprocess(data=df, synth=synth_df, test=test_factual, response=y_col, \
    inverse_transform=dataset.inverse_transform, cutoff=0.5)

```

5. Postprocess generated counterfactuals and inverse transform them back to their original feature ranges

```Python
results = mcce.results_sparse#[[dataset.feature_order]]
dataset.inverse_transform(results)

```

