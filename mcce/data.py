import re
import pandas as pd
import numpy as np
from sklearn import preprocessing


class Data():
    """
    Class to make it easy to load a CSV file of data and encode and scale the features using sklearn.
    
    Parameters
    ----------
    path : str
        Path where data is saved locally. 
    feature_order : list
        List containing the correct feature order when displaying the data.
    dtypes : dict
        Dictionary containing the data types of each feature.
    response : str
        The name of the response in the dataset. 
    immutables : list
        List of features that should be fixed when generating the counterfactuals. Cannot be empty. 
        These are the original (not encoded) feature names.
    encoding_method : str, default: OneHot_drop_first
        Type of OneHotEncoding {OneHot, OneHot_drop_binary, OneHot_drop_first}. Default will drop the first level if the 
        categorical feature has more than two levels. 
        Set to "Identity" for no encoding.
    scaling_method : str, default: MinMax
        Type of sklearn scaler to use. MinMax will scale all continuous features between 0 and 1.
    
    Methods
    -------
    transform :
        Function to transform the features using the desired scaling and encoding functionality.
    inverse_transform : 
        Function to reverse the encoding/scaling to get the features back in their original form. 
    get_pipeline_element :
    fit_encoder :
        Fit the sklearn encoder to the original features. This will one-hot encode the features. 
    fit_scaler :
        Fit the sklearn scaler to the original features. This will scale (e.g., between 0 and 1) the features.
    scale :
        Scale the continuous features using the fit_scaler. 
    descale :
        Descale the continuous features (i.e., transform back to their original scales).
    encode :
        Encode the categorical features using fit_encoder. 
    decode :
        Decode the categorical features (i.e., transform back to their original levels).
    
    """

    def __init__(self,
                 path, 
                 feature_order, 
                 dtypes, 
                 response, 
                 immutables,
                 encoding_method="OneHot_drop_first",
                 scaling_method="MinMax"):
        
        df = pd.read_csv(path, header=0, names=feature_order) # assume no index column

        self.immutables = immutables
        self.target = response
        self.feature_order = feature_order
        self.dtypes = dtypes
        self.encoding_method = encoding_method
        self.scaling_method = scaling_method

        # TO DO: Check if response in dtypes
        # Check if response in feature_order
        # Check if response in df.columns

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

        # Get the new categorical feature names after encoding
        self.categorical_encoded = self.encoder.get_feature_names(self.categorical).tolist()
        
        # Get the new immutable feature names after encoding
        immutables_encoded = []
        for immutable in immutables:
            if immutable in self.categorical:
                for new_col in self.categorical_encoded:
                    match = re.search(immutable, new_col)
                    if match:
                        immutables_encoded.append(new_col)
            else:
                immutables_encoded.append(immutable)

        self.immutables_encoded = immutables_encoded
        

    def transform(self, df):
        """
        Function to transform the features using the desired scaling and encoding functionality.

        Parameters
        ----------
        df : pd.DataFrame

        Returns
        -------
        output : pd.DataFrame
                With transformed features.
        """
        
        output = df.copy()

        for trans_name, trans_function in self._pipeline:
            
            if trans_name == "encoder" and self._identity_encoding:
                continue
            else:
                output = trans_function(output)

        return output

    def inverse_transform(self, df):
        """
        Function to inverse transform the features back to their original non-scaled/non-encoded feature values.

        Parameters
        ----------
        df : pd.DataFrame

        Returns
        -------
        output : pd.DataFrame
                With original features scales/encodings.
        """
        
        output = df.copy()

        for _, trans_function in self._inverse_pipeline:
            output = trans_function(output)

        return output

    def get_pipeline_element(self, key):
        """
        Get an element from the pipeline.
        Parameters
        ----------
        key : str
            Element of the pipeline to return
        Returns
        -------
        """
        
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
        """
        Fit desired sklearn encoder.

        Parameters
        ----------
        df : pd.DataFrame
        encoding_method : str, default: OneHot_drop_first
            The type of encoding to do. The default drops the first level of a categorical
            features with more than two levels. 

        Returns
        -------
        output : sklearn encoder
    
        """
        
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
        """
        Fit desired sklearn scaler.

        Parameters
        ----------
        df : pd.DataFrame
        scaling_method : str, default: MinMax
            The type of scaling to do. The default scales the continuous features between 0 and 1.

        Returns
        -------
        output : sklearn scaler
    
        """
        
        if scaling_method == "MinMax":
            scaler = preprocessing.MinMaxScaler().fit(df)
        elif scaling_method == "Standard":
            scaler = preprocessing.StandardScaler().fit(df)
        elif scaling_method is None or "Identity":
            scaler = preprocessing.FunctionTransformer(func=None, inverse_func=None)
        else:
            raise ValueError("Scaling Method not known")
        return scaler


    def scale(self, fitted_scaler, features, df):
        """
        Apply the fit sklearn scaler.

        Parameters
        ----------
        fitted_scaler : fitted sklearn scaler
        features : list
            List of continuous features to scale. 
        df : pd.DataFrame
        
        Returns
        -------
        output : pd.DataFrame
    
        """
        
        output = df.copy()
        output[features] = fitted_scaler.transform(output[features])

        return output


    def descale(self, fitted_scaler, features, df):
        """
        Reverse the scaling applied using the fit sklearn scaler.

        Parameters
        ----------
        fitted_scaler : fitted sklearn scaler
        features : list
            List of continuous features to scale. 
        df : pd.DataFrame
        
        Returns
        -------
        output : pd.DataFrame
    
        """
        
        output = df.copy()
        output[features] = fitted_scaler.inverse_transform(output[features])

        return output


    def encode(self, fitted_encoder, features, df):
        """
        Encode the categorical features using the fit sklearn encoder.

        Parameters
        ----------
        fitted_encoder : fitted sklearn encoder
        features : list
            List of continuous features to scale. 
        df : pd.DataFrame
        
        Returns
        -------
        output : pd.DataFrame
    
        """
        output = df.copy()
        encoded_features = fitted_encoder.get_feature_names(features)
        output[encoded_features] = fitted_encoder.transform(output[features])
        output = output.drop(features, axis=1)

        return output


    def decode(self, fitted_encoder, features, df):
        """
        Reverse the feature encoding using the fit sklearn encoder.

        Parameters
        ----------
        fitted_encoder : fitted sklearn encoder
        features : list
            List of continuous features to scale. 
        df : pd.DataFrame
        
        Returns
        -------
        output : pd.DataFrame
    
        """
        output = df.copy()
        encoded_features = fitted_encoder.get_feature_names(features)

        # Prevent errors for datasets without categorical data
        # inverse_transform cannot handle these cases
        if len(encoded_features) == 0:
            return output

        output[features] = fitted_encoder.inverse_transform(output[encoded_features])
        output = output.drop(encoded_features, axis=1)

        return output
    