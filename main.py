
from sklearn import  metrics
from sklearn.ensemble import RandomForestClassifier

from mcce import MCCE
from data import Data

# Download data: https://archive.ics.uci.edu/ml/datasets/adult
# Call file Adult_income.csv
# Save in a folder and define path as pointing to data

print("Loading data...")
path = "~/pkg/MCCE/Datasets/Adult/Adult_income.csv"
names=['age', 'workclass', 'fnlwgt', 'degree', 'education_years', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours', 'country', 'income']
fixed_features = ['age', 'sex']
response = 'income'
dtypes = {"age": "float", "workclass": "category", "fnlwgt": "float", "degree": "category", "education_years": "float", "marital-status": "category", "occupation": "category", "relationship": "category", "race": "category", "sex": "category", "capital-gain": "float", "capital-loss": "float", "hours": "float", "country": "category", "income": "category"}

data = Data(path=path, names=names, dtypes=dtypes, response=response, fixed_features=fixed_features)


# (1) Fit predictive model
X = data.df[data.cont_feat + data.cat_feat]
y = data.df[data.response] # 1 = income >= 50K, 0 = income < 50K

clf = RandomForestClassifier(max_depth=None, random_state=0)
model = clf.fit(X, y)

data.df[data.response] = clf.predict_proba(X)[:,1]

df = data.df

# (2) Find unhappy customers
cutoff = 0.5
test_idx = df[df[data.response]<cutoff].index # unhappy customers

# Decide which observations deserve counterfactuals
n_test = 3
test = data.df.loc[test_idx][0:n_test]


# (3) Fit MCCE object
print("Fitting MCCE model...")
mcce = MCCE(fixed_features=data.fixed_features, model=model)
mcce.fit(data.df[data.features], dtypes)
print("Generating counterfactuals with MCCE...")
synth_df = mcce.generate(test[data.features], k=500)

# (4) Postprocess generated counterfactuals
print("Postprocessing counterfactuals with MCCE...")
mcce.postprocess(data.df, synth_df, test, data.response, scaler=data.scaler)

# (5) Print results 

results_all = mcce.results
results = mcce.results_sparse

results_all[data.cont_feat] = data.scaler.inverse_transform(results_all[data.cont_feat])
test[data.cont_feat] = data.scaler.inverse_transform(test[data.cont_feat])
results[data.cont_feat] = data.scaler.inverse_transform(results[data.cont_feat])

# Print generated data with metrics
print(results_all.head(5))

# Print original test observations
print("Original test observation...\n")
print(test[~test.index.duplicated(keep='first')])

# Print best counterfactual explanations
print("Best counterfactuals for each test observation using MCCE...\n")
print(results)
