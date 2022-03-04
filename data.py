import pandas as pd
import numpy as np
from sklearn import preprocessing, metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from mcce import MCCE


class Data():

    def __init__(self, path, names, dtypes, response, fixed_features):
        
        df = pd.read_csv(path, header=0, names=names)
        self.fixed_features = fixed_features
        self.response = response
        self.dtypes = dtypes
        
        self.cat_feat = [feat for feat in self.dtypes.keys() if dtypes[feat] == 'category']
        self.cat_feat.remove(self.response)
        self.cont_feat = [feat for feat in self.dtypes.keys() if dtypes[feat] != 'category']
        self.features = self.cat_feat + self.cont_feat
        self.cols = self.features + [self.response]
        
        df = df.astype(self.dtypes)

        # Convert response to 1 if "desirable" label; 0 otherwise
        level = df[self.response].value_counts().idxmax()
        df[self.response] = np.where(df[self.response] != level, 1, 0)
            
        # One-hot encode categorical features to most used level
        self.df_encode = self.encode_features(df[self.cat_feat])

        # Normalize continuous features
        df_norm = self.normalize_features(df[self.cont_feat])
        
        # Raw data set will only have the one-hot encoded features but not the normalized ones
        self.df_raw = pd.concat([self.df_encode, df[self.cont_feat], df[self.response]], axis = 1)

        self.df = pd.concat([self.df_encode, df_norm, df[self.response]], axis = 1)

    def encode_features(self, df):
        
        cols = df.columns

        for x in df.columns:
            level = df[x].value_counts().idxmax()
            df[x] = np.where(df[x] == level, 1, 0)
        return df[cols]

    def normalize_features(self, df):
        
        cols = df.columns

        self.scaler = preprocessing.StandardScaler().fit(df)

        X_scaled = self.scaler.transform(df)
        X_scaled = pd.DataFrame(X_scaled, columns=df.columns, index=df.index)
        return X_scaled[cols]

    def fit_model(self, X, y, test_size=0.33):
        self.test_size = test_size

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        clf = RandomForestClassifier(max_depth=None, random_state=0)
        self.model = clf.fit(X_train, y_train)
        
        pred_train = self.model.predict(X_train)
        pred_test = self.model.predict(X_test)

        fpr, tpr, _ = metrics.roc_curve(y_train, pred_train, pos_label=1)
        self.train_auc = metrics.auc(fpr, tpr)

        fpr, tpr, _ = metrics.roc_curve(y_test, pred_test, pos_label=1)
        self.test_auc = metrics.auc(fpr, tpr)

        self.model_prediction = clf.predict(X)
