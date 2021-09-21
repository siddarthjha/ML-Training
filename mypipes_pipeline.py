import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline,FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin


class VarSelector(BaseEstimator, TransformerMixin):

    def __init__(self,feature_names):

        self.feature_names=feature_names

    def fit(self,X,y=None):

        return self

    def transform(self,X):

        return X[self.feature_names]

    def get_feature_names(self):

        return self.feature_names

class convert_to_numeric(BaseEstimator, TransformerMixin):

    def __init__(self):

        self.feature_names=[]

        
    def fit(self,X,y=None):

        self.feature_names=X.columns
        return self

    def transform(self,X):

        for col in X.columns:
            X[col]=pd.to_numeric(X[col],errors='coerce')
        return X

    def get_feature_names(self):

        return self.feature_names

class DataFrameImputer(BaseEstimator,TransformerMixin):

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


class create_dummies(BaseEstimator, TransformerMixin):

    def __init__(self,freq_cutoff):

        self.freq_cutoff=freq_cutoff
        self.var_cat_dict={}
        self.feature_names=[]

    def fit(self,X,y=None):

        data_cols=X.columns

        for col in data_cols:

            k=X[col].value_counts()

            cats=k.index[k>=self.freq_cutoff]

            self.var_cat_dict[col]=cats
        
        for col in self.var_cat_dict.keys():
            for cat in self.var_cat_dict[col]:
                self.feature_names.append(col+'_'+str(cat)) 

        return self

    def transform(self,X):

        dummy_data=X.copy()

        for col in self.var_cat_dict.keys():

            for cat in self.var_cat_dict[col]:
                name=col+'_'+str(cat)
                dummy_data[name]=(dummy_data[col]==cat).astype(int)

            del dummy_data[col]

        return dummy_data

    def get_feature_names(self):

        return self.feature_names


class remove_percent(BaseEstimator, TransformerMixin):

    def __init__(self):

        self.feature_names=[]

    def fit(self,X,y=None):

        self.feature_names=X.columns

        return self

    def transform(self,X):

        for col in X.columns:

            X[col]=X[col].str.replace('%','')

        return X

    def get_feature_names(self):

        return self.feature_names

class custom_fico(BaseEstimator, TransformerMixin):

    def __init__(self):

        self.feature_names=['fico']

    def fit(self,X,y=None):

        return self

    def transform(self,X):

        k=X['FICO.Range'].str.split('-',expand=True)[0]

        return(pd.DataFrame({'fico':k}))

    def get_feature_names(self):

        return self.feature_names

class custom_EL(BaseEstimator, TransformerMixin):

    def __init__(self):

        self.feature_names=['EL']

    def fit(self,X,y=None):

        return self

    def transform(self,X):

        k=X['Employment.Length'].replace({'10+ years':10, '< 1 year':0,
                                       '2 years':2, '3 years':3, 
                                       '5 years':5, '4 years':4,
                               '1 year':1, '6 years':6, '7 years':7, 
                                       '8 years':8, '9 years':9})

        return(pd.DataFrame({'EL':k}))

    def get_feature_names(self):

        return self.feature_names













