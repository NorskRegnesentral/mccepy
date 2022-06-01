import pandas as pd
import numpy as np
import re

from sklearn import preprocessing, metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from mcce import MCCE


class Data():

    def __init__(self, \
        path, 
        feature_order, 
        dtypes, 
        response, 
        fixed_features,  # these are the original feature names
        encoding_method="OneHot_drop_first",
        scaling_method="MinMax"):
        
        df = pd.read_csv(path, header=0, names=feature_order) # assume no index column
        self.fixed_features = fixed_features
        self.target = response
        self.feature_order = feature_order
        self.dtypes = dtypes
        self.encoding_method = encoding_method
        self.scaling_method = scaling_method

        self.categorical = [feat for feat in self.dtypes.keys() if dtypes[feat] == 'category']
        self.categorical.remove(self.target)
        self.continuous = [feat for feat in self.dtypes.keys() if dtypes[feat] != 'category']
        self.features = self.categorical + self.continuous
        self.cols = self.features + [self.target]
        
        df = df.astype(self.dtypes)

        # Convert target to 1 if "desirable" label; 0 otherwise
        level = df[self.target].value_counts().idxmax()
        df[self.target] = np.where(df[self.target] != level, 1, 0)
        
        self.df = df
        self.df_raw = df 

        # Fit scaler and encoder
        self.scaler = self.fit_scaler(self.df[self.continuous], scaling_method)
        self.encoder = self.fit_encoder(self.df[self.categorical], encoding_method)
        self._identity_encoding = (encoding_method is None or encoding_method == "Identity")

        # Preparing pipeline components
        self._pipeline = self.__init_pipeline()
        self._inverse_pipeline = self.__init_inverse_pipeline()

        # Process the data
        self.df = self.transform(self.df)

        # Can we get the fixed feature names after the transformation?
        self.categorical_encoded = self.encoder.get_feature_names(self.categorical).tolist()
        fixed_features_encoded = []
        for fixed in fixed_features:
            if fixed in self.categorical:
                for new_col in self.categorical_encoded:
                    match = re.search(fixed, new_col)
                    if match:
                        fixed_features_encoded.append(new_col)
            else:
                fixed_features_encoded.append(fixed)

        # print(type(fixed_features_encoded))
        self.fixed_features_encoded = fixed_features_encoded
        

    def transform(self, df):
        
        output = df.copy()

        for trans_name, trans_function in self._pipeline:
            
            if trans_name == "encoder" and self._identity_encoding:
                continue
            else:
                output = trans_function(output)

        return output

    def inverse_transform(self, df):
        
        output = df.copy()

        for trans_name, trans_function in self._inverse_pipeline:
            output = trans_function(output)

        return output

    def get_pipeline_element(self, key):
        
        key_idx = list(zip(*self._pipeline))[0].index(key)  # find key in pipeline
        return self._pipeline[key_idx][1]

    def __init_pipeline(self):
        return [
            ("scaler", lambda x: self.scale(self.scaler, self.continuous, x)),
            ("encoder", lambda x: self.encode(self.encoder, self.categorical, x)),
        ]

    def __init_inverse_pipeline(self):
        return [
            ("encoder", lambda x: self.decode(self.encoder, self.categorical, x)),
            ("scaler", lambda x: self.descale(self.scaler, self.continuous, x)),
        ]
        
    def fit_encoder(self, df, encoding_method="OneHot_drop_first"):
        
        if encoding_method == "OneHot":
            encoder = preprocessing.OneHotEncoder(
                handle_unknown="error", sparse=False
            ).fit(df)
        elif encoding_method == "OneHot_drop_binary":
            encoder = preprocessing.OneHotEncoder(
                drop="if_binary", handle_unknown="error", sparse=False
            ).fit(df)
        elif encoding_method == "OneHot_drop_first":
            encoder = preprocessing.OneHotEncoder(
                drop="first", sparse=False
            ).fit(df)
        elif encoding_method is None or "Identity":
            encoder = preprocessing.FunctionTransformer(func=None, inverse_func=None)
            
        else:
            raise ValueError("Encoding Method not known")

        return encoder


    def fit_scaler(self, df, scaling_method="MinMax"):
        
        if scaling_method == "MinMax":
            scaler = preprocessing.MinMaxScaler().fit(df)
        elif scaling_method == "Standard":
            scaler = preprocessing.StandardScaler().fit(df)
        elif scaling_method is None or "Identity":
            scaler = preprocessing.FunctionTransformer(func=None, inverse_func=None)
        else:
            raise ValueError("Scaling Method not known")
        
        # X_scaled = self.scaler.transform(df)
        # X_scaled = pd.DataFrame(X_scaled, columns=df.columns, index=df.index)

        return scaler # X_scaled[cols]


    def scale(self, fitted_scaler, features, df):
        
        output = df.copy()
        output[features] = fitted_scaler.transform(output[features])

        return output


    def descale(self, fitted_scaler, features, df) -> pd.DataFrame:
        
        output = df.copy()
        output[features] = fitted_scaler.inverse_transform(output[features])

        return output


    def encode(self, fitted_encoder, features, df):
        
        output = df.copy()
        encoded_features = fitted_encoder.get_feature_names(features)
        output[encoded_features] = fitted_encoder.transform(output[features])
        output = output.drop(features, axis=1)

        return output


    def decode(self, fitted_encoder, features, df):
        
        output = df.copy()
        encoded_features = fitted_encoder.get_feature_names(features)

        # Prevent errors for datasets without categorical data
        # inverse_transform cannot handle these cases
        if len(encoded_features) == 0:
            return output

        output[features] = fitted_encoder.inverse_transform(output[encoded_features])
        output = output.drop(encoded_features, axis=1)

        return output
    
    

    

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