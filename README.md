# mccepy: Monte Carlo sampling of realistic Counterfactual Explanations for tabular data
Python package to generate counterfactuals using Monte Carlo sampling of realistic counterfactual explanations.


## Installation

## Example

For other examples (e.g., how to use MCCE with the CARLA package), see examples/.

This example uses the [US adult census dataset](https://archive.ics.uci.edu/ml/datasets/adult) in the Data/ repository. Although there are some predefined data and model classes (see examples/Example_notebook.ipynb), in this Readme, we define the data/model classes from scratch.


Load data using ```pandas``` and specify column names, feature types, response name, and a list of immutable features,.

```Python
import pandas as pd

feature_order = ['age', 'workclass', 'fnlwgt', 'education-num', 'marital-status', 'occupation', 
                 'relationship', 'race', 'sex', 'hours-per-week']
                 
dtypes = {"age": "float", 
          "workclass": "category", 
          "fnlwgt": "float", 
          "education-num": "float",
          "marital-status": "category", 
          "occupation": "category", 
          "relationship": "category", 
          "race": "category",
          "sex": "category", 
          "hours-per-week": "float",
          "income": "category"}

categorical = ['workclass', 'marital-status', 'occupation', 'relationship', 'race', 'sex']
continuous = ['age', 'fnlwgt', 'education-num', 'hours-per-week']
immutable_features = ['age', 'sex']
target = ['income']
features = categorical + continuous

path = '../Data/adult_data.csv'

df = pd.read_csv(path)
df = df[features + target]

```

Using ```sklearn```, scale the continuous features between 0 and 1 and encode the categorical features using one-hot encoding (drop the first level).

```Python
from sklearn import preprocessing

encoder = preprocessing.OneHotEncoder(drop="first", sparse=False).fit(df[categorical])
df_encoded = encoder.transform(df[categorical])

scaler = preprocessing.MinMaxScaler().fit(df[continuous])
df_scaled = scaler.transform(df[continuous])

categorical_encoded = encoder.get_feature_names(categorical).tolist()
df_scaled = pd.DataFrame(df_scaled, columns=continuous)
df_encoded = pd.DataFrame(df_encoded, columns=categorical_encoded)

df = pd.concat([df_scaled, df_encoded, df[target]], axis=1)

```

Define an inverse function to go easily back to non scaled/encoded version of the features.

```Python
def inverse_transform(df, 
                      scaler, 
                      encoder, 
                      continuous,
                      categorical,
                      categorical_encoded, 
                      ):

    df_categorical = pd.DataFrame(encoder.inverse_transform(df[categorical_encoded]), columns=categorical)
    df_continuous = pd.DataFrame(scaler.inverse_transform(df[continuous]), columns=continuous)

    return pd.concat([df_categorical, df_continuous], axis=1)
```

Since the feature "sex" is a categorical feature with two levels, the encoded version of the data set now has a new feature name. Below, we find its new "encoded name":

```Python
immutable_features_encoded = []
for immutable in immutable_features:
    if immutable in categorical:
        for new_col in categorical_encoded:
            match = re.search(immutable, new_col)
            if match:
                immutable_features_encoded.append(new_col)
    else:
        immutable_features_encoded.append(immutable)

```

Create data object to pass to MCCE method

```Python
class Dataset():
    def __init__(self, 
                 immutable_features, 
                 target,
                 categorical,
                 immutable_features_encoded,
                 continuous,
                 features,
                 encoder,
                 scaler,
                 inverse_transform,
                 ):
        
        self.immutable_features = immutable_features
        self.target = target
        self.feature_order = feature_order
        self.dtypes = dtypes

        self.categorical = categorical
        self.continuous = continuous
        self.features = self.categorical + self.continuous
        self.cols = self.features + [self.target]
        self.immutable_features_encoded = immutable_features_encoded
        self.encoder = encoder
        self.scaler = scaler
        self.inverse_transform = inverse_transform
        
dataset = Dataset(immutable_features, 
                  target,
                  categorical,
                  immutable_features_encoded,
                  continuous,
                  features,
                  encoder,
                  scaler,
                  inverse_transform)

dtypes = dict([(x, "float") for x in continuous])
for x in categorical_encoded:
    dtypes[x] = "category"
df = (df).astype(dtypes)
```

Train a random forest to predict Income>=50000

```Python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

y = df[target]
X = df.drop(target, axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
clf = RandomForestClassifier(max_depth=None, random_state=0)
ml_model = clf.fit(X_train, y_train)

pred_train = ml_model.predict(X_train)
pred_test = ml_model.predict(X_test)

fpr, tpr, _ = metrics.roc_curve(y_train, pred_train, pos_label=1)
train_auc = metrics.auc(fpr, tpr)

fpr, tpr, _ = metrics.roc_curve(y_test, pred_test, pos_label=1)
test_auc = metrics.auc(fpr, tpr)

model_prediction = clf.predict(X)

```

Decide which observations to generate counterfactual explanations for.

```Python
import numpy as np

preds = ml_model.predict_proba(dataset.df)[:,1]
factual_id = np.where(preds < 0.5)
factuals = dataset.df.loc[factual_id]
test_factual = factuals.iloc[:5]

```


Use ```MCCE``` to generate counterfactual explanations

```Python
from mcce import MCCE

mcce = MCCE(dataset, ml_model)

mcce.fit(df.drop(target, axis=1), dtypes) # remove the response from training data set

cfs = mcce.generate(test_factual.drop(target, axis=1), k=100) # sample 100 times per node

mcce.postprocess(cfs, test_factual, cutoff=0.5) # predicted >= 0.5 is considered positive; < 0.5 is negative

```

Print original feature values for the five test observations

```Python
cfs = mcce.results_sparse
cfs['income'] = test_factual['income']

# invert the features to their original form
decoded_factuals = dataset.inverse_transform(test_factual,
                                             scaler, 
                                             encoder, 
                                             continuous,
                                             categorical,
                                             categorical_encoded)[feature_order]

print(decoded_factuals)

```

```Python
    age  workclass    fnlwgt  education-num  marital-status  occupation  \
0  39.0          3   77516.0           13.0               1           3   
2  38.0          0  215646.0            9.0               2           3   
3  53.0          0  234721.0            7.0               0           3   
4  28.0          0  338409.0           13.0               0           0   
5  37.0          0  284582.0           14.0               0           2   

   relationship  race  sex  
0             1     0    0      
2             1     0    0      
3             0     1    0        
4             3     1    1       
5             3     0    1 

```

Print counterfactual explanations values for five test observations

```Python
decoded_cfs = dataset.inverse_transform(cfs,
                                        scaler, 
                                        encoder, 
                                        continuous,
                                        categorical,
                                        categorical_encoded)[feature_order]
print(decoded_cfs)
```

```Python
    age  workclass    fnlwgt  education-num  marital-status  occupation  \
0  39.0          0  175232.0           13.0               1           0   
1  38.0          0   86643.0           16.0               2           0   
2  53.0          0  184176.0            9.0               0           3   
3  28.0          0  132686.0           14.0               0           0   
4  37.0          0  174150.0           14.0               0           2   

   relationship  race  sex  hours-per-week  
0             1     0    0            40.0  
1             1     0    0            45.0  
2             0     0    0            40.0  
3             0     0    1            40.0  
4             0     0    1            40.0  
```
