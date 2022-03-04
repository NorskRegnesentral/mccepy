# mccepy
Python package to generate counterfactuals using Monte Carlo sampling of realistic counterfactual explanations ([MCCE])(https://arxiv.org/pdf/2111.09790.pdf).


## Installation

### pip

```bash
pip install mccepy
```

### Source

```bash
git clone git@github.com:hazy/mccepy.git
cd mccepy
pip install -r requirements.txt
python setup.py install
```

## Examples


### Adult dataset
Download the adult dataset [US adult census dataset](https://github.com/hazy/synthpop/blob/master/datasets/README.md). Add the dataset to a local repository. 


### mcce

1. Initialize data object with path to the data file, column names, feature types, response name, and a list of fixed features. 

```Python
from data import Data

data = Data(path=path, names=names, dtypes=dtypes, response=response, fixed_features=fixed_features)
```

2. Fit a predictive model e.g., random forest to predict response based on all features. For example:


```Python
X = data.df[data.cont_feat + data.cat_feat]
y = data.df[data.response]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
clf = RandomForestClassifier(max_depth=None, random_state=0)
model = clf.fit(X_train, y_train)

# Replace the response with the predicted response
data.df[data.response] = clf.predict(X)

df = data.df

```

3. Decide which customers to generated counterfactual explanations for

```Python
test_idx = df[df[data.response]==0].index # unhappy customers have response 0

# Choose first three test observations 
n_test = 3
test = data.df.loc[test_idx][0:n_test]

```

4. Fit MCCE object and generate counterfactual explanations using MCCE method and CART method

```Python
from mcce import MCCE

mcce = MCCE(fixed_features=data.fixed_features, model=model)
mcce.fit(data.df[data.features], dtypes)
synth_df = mcce.generate(test[data.features], k=500) # 10000 takes a long time -- time it?

```

5. Postprocess generated counterfactuals

```Python
mcce.postprocess(data.df, synth_df, test, data.response, scaler=data.scaler)

# print results 

results_all = mcce.results
results = mcce.results_sparse

```

