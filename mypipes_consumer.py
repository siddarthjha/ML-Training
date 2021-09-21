import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline,FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin


class VarSelector(BaseEstimator, TransformerMixin):

    def __init__(self,feature_names):

        self.feature_names=feature_names


    def fit(self,x,y=None):

        return self

    def transform(self,X):

        return X[self.feature_names]

    def get_feature_names(self):

        return self.feature_names


class bool_to_int(BaseEstimator, TransformerMixin):

    def __init__(self):
        self.feature_names = []

    def fit(self, x, y = None):

        self.feature_names = x.columns
        return self

    def transform(self, X):

        for col in X.columns:
            X[col] = X[col].astype(bool)
            X[col] = X[col].astype(int)
        return X

    def get_feature_names(self):

        return self.feature_names


class convert_to_numeric(BaseEstimator, TransformerMixin):

    def __init__(self):

        self.feature_names=[]

    def fit(self,x,y=None):

        self.feature_names=x.columns
        return self

    def transform(self,X):

        for col in X.columns:
            X[col]=pd.to_numeric(X[col],errors='coerce')
        return X

    def get_feature_names(self):
        return self.feature_names


class convert_to_dummy(BaseEstimator, TransformerMixin):

    def __init__(self, freq_cutoff=0):
        self.feature_names = []
        self.var_cat_dict={}
        self.freq_cutoff=freq_cutoff

    def fit(self, X, y = None):

        data_cols=X.columns

        for col in data_cols:

            k=X[col].value_counts()

            cats=k.index[k>=self.freq_cutoff]

            self.var_cat_dict[col]=cats
        
        for col in self.var_cat_dict.keys():
            for cat in self.var_cat_dict[col]:
                self.feature_names.append(col+'_'+str(cat))

        return self

    def transform(self,x,y=None):
        dummy_data=x.copy()

        for col in self.var_cat_dict.keys():
            for cat in self.var_cat_dict[col]:
                name=col+'_'+cat
                dummy_data[name]=(dummy_data[col]==cat).astype(int)

            del dummy_data[col]

        return dummy_data

    def get_feature_names(self):

        return self.feature_names


class MissingValues(BaseEstimator,TransformerMixin):

    def __init__(self):

        self.impute_dict={}
        self.feature_names=[]

    def fit(self, X, y=None):

        self.feature_names=X.columns

        for col in X.columns:
            if X[col].dtype=='O':
                self.impute_dict[col]='missing'
            else:
                self.impute_dict[col]=X[col].median()
        return self

    def transform(self, X, y=None):
        return X.fillna(self.impute_dict)

    def get_feature_names(self):

        return self.feature_names


class datetime_conversion(BaseEstimator, TransformerMixin):

    def __init__(self):
        self.feature_names = []

    def fit(self, x, y = None):

        extract = ['weekday', 'month', 'day']
        for col in x.columns:
            for val in range(len(extract)):
                self.feature_names.append(col + '_' + str(extract[val]))
        return self

    def transform(self, X):

        for col in X.columns:
            X[col] = pd.to_datetime(X[col], infer_datetime_format = True)

        for col in X.columns:
            name = col + '_weekday'
            X[name] = X[col].dt.weekday

            name = col + '_month'
            X[name] = X[col].dt.month

            name = col + '_day'
            X[name] = X[col].dt.day
                        
            del X[col]
        
        return X

    def get_feature_names(self):

        return self.feature_names


class cyclic_features(BaseEstimator, TransformerMixin):

    def __init__(self):
        self.feature_names = []

    def fit(self, x, y = None):

        cyclic = ['sin', 'cos']
        extract = ['weekday', 'month', 'day']

        for col in x.columns:
            for val1 in range(len(extract)):
                for val2 in range(len(cyclic)):
                    self.feature_names.append(col + '_' + str(extract[val1] + '_' + str(cyclic[val2])))
        
        return self

    def transform(self, X):

        for col in X.columns:
            X[col] = pd.to_datetime(X[col], infer_datetime_format = True)

        for col in X.columns:
            name = col + '_weekday_sin'
            X[name] = np.sin(2 * np.pi * X[col].dt.weekday / 7)

            name = col + '_weekday_cos'
            X[name] = np.cos(2 * np.pi * X[col].dt.weekday / 7)

            name = col + '_month_sin'
            X[name] = np.sin(2 * np.pi * X[col].dt.month / 12)

            name = col + '_month_cos'
            X[name] = np.cos(2 * np.pi * X[col].dt.month / 12)

            name = col + '_day_sin'
            X[name] = np.sin(2 * np.pi * X[col].dt.day / 31)

            name = col + '_day_cos'
            X[name] = np.cos(2 * np.pi * X[col].dt.day / 31)

            del X[col]

        return X

    def get_feature_names(self):

        return self.feature_names


class dt_difference(BaseEstimator, TransformerMixin):

    def __init__(self):
        self.feature_names = []

    def fit(self, x, y = None):

        self.feature_names = ['date_difference']
        return self

    def transform(self, X):

        for col in X.columns:
            X[col] = pd.to_datetime(X[col], infer_datetime_format = True)
            X[col] = X[col].astype('datetime64[D]')

        k = (X.iloc[:,1] - X.iloc[:,0]).dt.days
        

        return (pd.DataFrame({'date_difference': k}))

    def get_feature_names(self):

        return self.feature_names



class pdPipeline(Pipeline):

    def get_feature_names(self):

        last_step = self.steps[-1][-1]

        return last_step.get_feature_names()
