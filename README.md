# mccepy
Python package to generate counterfactuals using [Monte Carlo sampling of realistic counterfactual explanations](https://arxiv.org/pdf/2111.09790.pdf).


## Installation

### Source

```bash
git clone git@github.com:NorskRegnesentral/mccepy.git
cd mccepy
pip install -r requirements.txt
python setup.py install
```

## Examples


### Adult dataset
Download the [US adult census dataset](https://archive.ics.uci.edu/ml/datasets/adult). Add the dataset to a local repository and save the path to the data. 


### mcce

1. Initialize Data object with path to the data file, column names, feature types, response name, and a list of fixed features. 

```Python
from data import Data

names = ['age', 'workclass', 'fnlwgt', 'degree', 'education_years', 'marital-status', 'occupation', 'relationship', 'race', '
         'sex', 'capital-gain', 'capital-loss', 'hours', 'country', 'income']
dtypes = {"age": "float", "workclass": "category", "fnlwgt": "float", "degree": "category", "education_years": "float", '
          "marital-status": "category", "occupation": "category", "relationship": "category", "race": "category", '
          "sex": "category", "capital-gain": "float", "capital-loss": "float", \
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
data.df[data.response] = clf.predict_proba(X)[:,1]

df = data.df

```

3. Decide which customers to generate counterfactual explanations for

```Python
cutoff = 0.5
test_idx = df[df[data.response]<cutoff].index # unhappy customers have response 0

# Choose first three unhappy customers 
n_test = 3
test = data.df.loc[test_idx][0:n_test]

```

4. Initialize MCCE object and generate counterfactual explanations using CART

```Python
from mcce import MCCE

mcce = MCCE(fixed_features=data.fixed_features, model=model)

# Fit CART models iteratively while always conditioning on fixed_features
mcce.fit(data.df[data.features], dtypes)

# Generate 500 counterfactual explanations for each test observation
synth_df = mcce.generate(test[data.features], k=500)

```

5. Postprocess generated counterfactuals

```Python
# This step removes all generated explanations that are not valid and computes metrics like distance, feasibility, and redundancy
mcce.postprocess(data.df, synth_df, test, data.response, scaler=data.scaler, cutoff=cutoff)

# print all generated rows with metrics as additional columns 
results_all = mcce.results 

# print the best result for each test observation
results = mcce.results_sparse

# Original test observation features and predicted response ('income')
    workclass  degree  marital-status  occupation  relationship  race  sex  country   age    fnlwgt  education_years  capital-gain  capital-loss  hours  income
0          0       0               1           0             1     1    1        1  50.0   83311.0             13.0           0.0           0.0   13.0    0.07
1          1       1               0           0             0     1    1        1  38.0  215646.0              9.0           0.0           0.0   40.0    0.00
2          1       0               1           0             1     0    1        1  53.0  234721.0              7.0           0.0           0.0   40.0    0.00

# Best counterfactual example for all three test observations
  workclass degree marital-status occupation relationship race sex country   age    fnlwgt  education_years  capital-gain  capital-loss  hours income   L0        L2       yNN  feasibility redundancy success violation
0         0      0              1          0            0    1   1       0  50.0  120781.0             13.0           0.0           0.0   30.0      1  4.0  3.731809  0.999831     1.428615          3       1         0
1         1      1              0          1            0    1   1       1  38.0  123983.0             13.0           0.0           0.0   40.0      1  3.0  3.423253  0.999560     1.118427          2       1         0
2         1      0              1          0            0    0   1       0  53.0  304570.0             10.0           0.0           0.0   40.0      1  4.0  3.827878  0.999374     1.233153          2       1         0

```

