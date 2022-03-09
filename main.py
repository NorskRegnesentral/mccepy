
from sklearn import  metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from mcce import MCCE
from data import Data

path = "~/pkg/MCCE/Datasets/Adult/Adult_income.csv"
names=['age', 'workclass', 'fnlwgt', 'degree', 'education_years', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours', 'country', 'income']
fixed_features = ['age', 'sex']
response = 'income'
dtypes = {"age": "float", "workclass": "category", "fnlwgt": "float", "degree": "category", "education_years": "float", "marital-status": "category", "occupation": "category", "relationship": "category", "race": "category", "sex": "category", "capital-gain": "float", "capital-loss": "float", "hours": "float", "country": "category", "income": "category"}

data = Data(path=path, names=names, dtypes=dtypes, response=response, fixed_features=fixed_features)


# (1) Fit predictive model
X = data.df[data.cont_feat + data.cat_feat]
y = data.df[data.response]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
clf = RandomForestClassifier(max_depth=None, random_state=0)
model = clf.fit(X_train, y_train)

pred_train = model.predict(X_train)
pred_test = model.predict(X_test)

fpr, tpr, _ = metrics.roc_curve(y_train, pred_train, pos_label=1)
train_auc = metrics.auc(fpr, tpr)

fpr, tpr, _ = metrics.roc_curve(y_test, pred_test, pos_label=1)
test_auc = metrics.auc(fpr, tpr)

data.df[data.response] = clf.predict(X)

df = data.df

# (2) Find unhappy customers
test_idx = df[df[data.response]==0].index # unhappy customers

# Decide which observations deserve counterfactuals
n_test = 3
test = data.df.loc[test_idx][0:n_test]


# (3) Fit MCCE object
mcce = MCCE(fixed_features=data.fixed_features, model=model)
# print(data.df[data.features])

mcce.fit(data.df[data.features], dtypes)
synth_df = mcce.generate(test[data.features], k=500)

# ----------------------


# (4) Postprocess generated counterfactuals
mcce.postprocess(data.df, synth_df, test, data.response, scaler=data.scaler)

# (5) Print results 

results_all = mcce.results
results = mcce.results_sparse

results_all[data.cont_feat] = data.scaler.inverse_transform(results_all[data.cont_feat])
test[data.cont_feat] = data.scaler.inverse_transform(test[data.cont_feat])
results[data.cont_feat] = data.scaler.inverse_transform(results[data.cont_feat])


print(results_all.head(5))
print(test[~test.index.duplicated(keep='first')])
print(results)
