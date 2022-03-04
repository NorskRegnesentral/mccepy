# mccepy
Python package to generate counterfactuals using [Monte Carlo sampling of realistic counterfactual explanations](https://arxiv.org/pdf/2111.09790.pdf).


## Installation

### pip

```bash
pip install mccepy
```

### Source

```bash
git clone git@github.com:NorskRegnesentral/mccepy.git
cd mccepy
pip install -r requirements.txt
python setup.py install
```

## Examples


### Adult dataset
Download the [US adult census dataset](https://github.com/hazy/synthpop/blob/master/datasets/README.md). Add the dataset to a local repository. 


### mcce

1. Initialize data object with path to the data file, column names, feature types, response name, and a list of fixed features. 

```Python
from data import Data

names = ['age', 'workclass', 'fnlwgt', 'degree', 'education_years', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', \
       'hours', 'country', 'income']
dtypes = {"age": "float", "workclass": "category", "fnlwgt": "float", "degree": "category", "education_years": "float", "marital-status": "category", \
          "occupation": "category", "relationship": "category", "race": "category", "sex": "category", "capital-gain": "float", "capital-loss": "float", \
          "hours": "float", "country": "category", "income": "category"}
response = 'income'
fixed_features = ['age', 'sex']

data = Data(path=path, names=names, dtypes=dtypes, response=response, fixed_features=fixed_features)
```

2. Fit a predictive model e.g., random forest to predict response based on all features. For example


```Python
from sklearn.ensemble import RandomForestClassifier

X = data.df[data.cont_feat + data.cat_feat]
y = data.df[data.response]

clf = RandomForestClassifier(max_depth=None, random_state=0)
model = clf.fit(X, y)

# Replace the response with the predicted response
data.df[data.response] = clf.predict(X)

df = data.df

```

3. Decide which customers to generate counterfactual explanations for

```Python
test_idx = df[df[data.response]==0].index # unhappy customers have response 0

# Choose first three test observations 
n_test = 3
test = data.df.loc[test_idx][0:n_test]

```

4. Fit MCCE object and generate counterfactual explanations using CART

```Python
from mcce import MCCE

mcce = MCCE(fixed_features=data.fixed_features, model=model)

# Fit CART model based and condition on fixed_features
mcce.fit(data.df[data.features], dtypes)

# Generate 500 observations for each test observation
synth_df = mcce.generate(test[data.features], k=500)

```

5. Postprocess generated counterfactuals

```Python
# This step removes all generated observations that are not valid and computes metrics like distance, feasibility, and redundancy
mcce.postprocess(data.df, synth_df, test, data.response, scaler=data.scaler)

# print all generated rows with metrics as additional columns 
results_all = mcce.results 

# print the best result for each test observation
results = mcce.results_sparse

# Original test observation features and predicted response ('income')
   workclass  degree  marital-status  occupation  relationship  race  sex  country   age    fnlwgt  education_years  capital-gain  capital-loss  hours  income
0          0       0               1           0             1     1    1        1  50.0   83311.0             13.0           0.0           0.0   13.0       0
1          1       1               0           0             0     1    1        1  38.0  215646.0              9.0           0.0           0.0   40.0       0
2          1       0               1           0             1     0    1        1  53.0  234721.0              7.0           0.0           0.0   40.0       0
  
  
# The best counterfactual generated for the three test observations
  workclass degree marital-status occupation relationship race sex country   age    fnlwgt  education_years  capital-gain  capital-loss  hours income   L0        L2       yNN  feasibility redundancy violation
0         0      0              1          0            0    1   1       1  50.0  138370.0             13.0           0.0           0.0   60.0      1  3.0  5.328109  0.999692     1.057664          2         0
1         1      1              0          1            0    1   1       1  38.0  315640.0             10.0           0.0           0.0   40.0      1  3.0  2.336079  0.999692     1.356356          2         0
2         1      0              1          0            0    0   1       1  53.0  261584.0             13.0           0.0           0.0   40.0      1  3.0  3.586717  1.000000     1.234961          2         
```

