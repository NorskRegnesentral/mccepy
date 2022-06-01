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

### A. Use MCCE with the [CARLA](https://github.com/carla-recourse/CARLA) python package

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

3. Create data objects to pass to MCCE

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

synth_df = mcce.generate(test_factual.drop(y_col, axis=1), k=100) # for each test obs

mcce.postprocess(data=df, synth=synth_df, test=test_factual, response=y_col, \
    inverse_transform=dataset.inverse_transform, cutoff=0.5) # postprocess the samples

# print average metrics across all test observations
print([mcce.results_sparse.L0.mean(), mcce.results_sparse.L2.mean(), mcce.results_sparse.feasibility.mean(),\
  mcce.results_sparse.violation.mean(), mcce.results_sparse.shape[0]])
    
```

### B. Use MCCE stand alone

In case you want to use MCCE without CARLA, use the following steps. 

Download a data set, like the [US adult census dataset](https://archive.ics.uci.edu/ml/datasets/adult). Add the dataset to a local repository and save the path to the data. 


1. Initialize Data object with path to the data file, column names, feature types, response name, a list of fixed features, and potential encoding and scaling methods. 

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
from sklearn.ensemble import RandomForestClassifier

class RandomForestModel():
    """The default way of implementing RandomForest from sklearn
    https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html"""

    def __init__(self, data):

        df_train = data.df
        
        x_train = df_train[data.continuous + data.categorical_encoded]
        y_train = df_train[data.target]

        self._feature_input_order = data.continuous + data.categorical_encoded

        param = {
            "max_depth": None,  # determines how deep the tree can go
            "n_estimators": 5, # number of trees
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
    
    def get_ordered_features(self, x):
        return x[self.feature_input_order]

ml_model = RandomForestModel(dataset)

```

3. Decide which customers to generate counterfactual explanations for.

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

dtypes = dict([(x, "float") for x in cont_feat])
for x in cat_feat_encoded:
    dtypes[x] = "category"
df = (dataset.df).astype(dtypes)
```

5. Initialize MCCE object and generate counterfactual explanations using CART

```Python
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

6. Postprocess generated counterfactuals and inverse transform them back to their original feature ranges

```Python
results = mcce.results_sparse
dataset.inverse_transform(results)

```
Original feature values for five test observations
```Python
    age  workclass    fnlwgt  education-num  marital-status  occupation  \
0  39.0          3   77516.0           13.0               1           3   
2  38.0          0  215646.0            9.0               2           3   
3  53.0          0  234721.0            7.0               0           3   
4  28.0          0  338409.0           13.0               0           0   
5  37.0          0  284582.0           14.0               0           2   

   relationship  race  sex  capital-gain  capital-loss  hours-per-week  \
0             1     0    0        2174.0           0.0            40.0   
2             1     0    0           0.0           0.0            40.0   
3             0     1    0           0.0           0.0            40.0   
4             3     1    1           0.0           0.0            40.0   
5             3     0    1           0.0           0.0            40.0   

   country  income  
0        0       0  
2        0       0  
3        0       0  
4        3       0  
5        0       0

```


Nearest counterfactuals for five test observations
```Python
    age  workclass     fnlwgt  education-num  marital-status  occupation  \
0  39.0          3  1455435.0           13.0               0           3   
2  38.0          0   248694.0           10.0               0           3   
3  53.0          0   215990.0            9.0               0           3   
4  28.0          0   132686.0           14.0               0           0   
5  37.0          0   336880.0           14.0               0           2   

   relationship  race  sex  capital-gain  capital-loss  hours-per-week  \
0             0     0    0           0.0           0.0            72.0   
2             0     0    0           0.0           0.0            40.0   
3             0     0    0           0.0           0.0            40.0   
4             0     0    1           0.0           0.0            40.0   
5             0     0    1           0.0           0.0            60.0   

   country income  
0        0      1  
2        0      1  
3        0      1  
4        0      1  
5        0      1
```

