# mccepy
Python package to generate counterfactuals using Monte Carlo sampling of realistic counterfactual explanations.


## Installation

### From source

```bash
git clone git@github.com:NorskRegnesentral/mccepy.git
cd mccepy
pip install -r requirements.txt
```

## Examples

### A. Use MCCE stand alone

This example uses the [US adult census dataset](https://archive.ics.uci.edu/ml/datasets/adult) in the Data/ repository. Although there are some predefined data and model classes (see examples/Example_notebook.ipynb), in this Readme, we define the data/model classes from scratch.


1. Initialize the ```Data``` class with the path to the data file, column names, feature types, response name, a list of fixed features, and potential encoding and scaling methods. 

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
fixed_features = ['age', 'sex']
target = ['income']
features = categorical + continuous

path = '../Data/adult_data.csv'

df = pd.read_csv(path)
df = df[features + target]

```

Scale the continuous features between 0 and 1. Encode the categorical features using one-hot encoding

```Python
from sklearn import preprocessing, metrics

encoder = preprocessing.OneHotEncoder(drop="first", sparse=False).fit(df[categorical])
df_encoded = encoder.transform(df[categorical])

scaler = preprocessing.MinMaxScaler().fit(df[continuous])
df_scaled = scaler.transform(df[continuous])

categorical_encoded = encoder.get_feature_names(categorical).tolist()
df_scaled = pd.DataFrame(df_scaled, columns=continuous)
df_encoded = pd.DataFrame(df_encoded, columns=categorical_encoded)

df = pd.concat([df_scaled, df_encoded, df[target]], axis=1)

```

Define an inverse function to go easilby back to non scaled/encoded version

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

Find the fixed features in their encoded form

```Python
fixed_features_encoded = []
for fixed in fixed_features:
    if fixed in categorical:
        for new_col in categorical_encoded:
            match = re.search(fixed, new_col)
            if match:
                fixed_features_encoded.append(new_col)
    else:
        fixed_features_encoded.append(fixed)

```

2. Train a random forest to predict Income>=50000


```Python
y = df[target]
X = df.drop(target, axis=1)
test_size = 0.33

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
clf = RandomForestClassifier(max_depth=None, random_state=0)
ml_model = clf.fit(X_train, y_train)

pred_train = ml_model.predict(X_train)
pred_test = ml_model.predict(X_test)

fpr, tpr, _ = metrics.roc_curve(y_train, pred_train, pos_label=1)
train_auc = metrics.auc(fpr, tpr)

fpr, tpr, _ = metrics.roc_curve(y_test, pred_test, pos_label=1)
test_auc = metrics.auc(fpr, tpr)

model_prediction = clf.predict(X)

print(f"The out-of-sample AUC is {round(test_auc, 2)}")

```

3. Decide which observations to generate counterfactual explanations for.

```Python
import numpy as np

preds = ml_model.predict_proba(dataset.df)[:,1]
factual_id = np.where(preds < 0.5)
factuals = dataset.df.loc[factual_id]
test_factual = factuals.iloc[:5]

```

4. Create data objects to pass to MCCE

```Python
class Dataset():
    def __init__(self, 
                 fixed_features, 
                 target,
                 categorical,
                 fixed_features_encoded,
                 continuous,
                 features,
                 encoder,
                 scaler,
                 inverse_transform,
                 ):
        
        self.fixed_features = fixed_features
        self.target = target
        self.feature_order = feature_order
        self.dtypes = dtypes

        self.categorical = categorical
        self.continuous = continuous
        self.features = self.categorical + self.continuous
        self.cols = self.features + [self.target]
        self.fixed_features_encoded = fixed_features_encoded
        self.encoder = encoder
        self.scaler = scaler
        self.inverse_transform = inverse_transform
        
dataset = Dataset(fixed_features, 
                  target,
                  categorical,
                  fixed_features_encoded,
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

5. Initialize MCCE object and generate counterfactual explanations

```Python
from mcce import MCCE

mcce = MCCE(dataset=dataset,
            model=ml_model)

print("Fit trees")
mcce.fit(df.drop(target, axis=1), dtypes)

print("Sample observations for the specific test observations")
cfs = mcce.generate(test_factual.drop(target, axis=1), k=100)

print("Process the sampled observations")
mcce.postprocess(cfs=cfs, test_factual=test_factual, cutoff=0.5)

```

6. Print original feature values for five test observations

```Python
cfs = mcce.results_sparse
cfs['income'] = test_factual['income']

# invert the features to their original form
print("Original factuals:")
decoded_factuals = dataset.inverse_transform(test_factual,
                                             scaler, 
                                             encoder, 
                                             continuous,
                                             categorical,
                                             categorical_encoded)[feature_order]

decoded_factuals

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
decoded_cfs
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


### B. Use MCCE with the [CARLA](https://github.com/carla-recourse/CARLA) python package

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