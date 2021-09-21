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


class wday_conversion(BaseEstimator, TransformerMixin):

    def __init__(self, ):
        self.feature_names = []

    def fit(self, x, y = None):

        self.feature_names = x.columns
        return self

    def transform(self, X):

        days = {'Sunday': 0, 'Monday': 1, 'Tuesday': 2, 'Wednesday': 3, 'Thursday': 4, 'Friday': 5, 'Saturday': 6}

        for col in X.columns:
            X[col] = X[col].apply(lambda x: days[x])
        return X

    def get_feature_names(self):

        return self.feature_names


class cyclic_features(BaseEstimator, TransformerMixin):

    def __init__(self):
        self.feature_names = []

    def fit(self, x, y = None):

        cyclic = ['sin', 'cos']

        for col in x.columns:
            for val2 in range(len(cyclic)):
                self.feature_names.append(col + '_' + str(cyclic[val2]))
        
        return self

    def transform(self, X):

        days = {'Sunday': 0, 'Monday': 1, 'Tuesday': 2, 'Wednesday': 3, 'Thursday': 4, 'Friday': 5, 'Saturday': 6}

        for col in X.columns:
            X[col] = X[col].apply(lambda x: days[x])

        for col in X.columns:
            name = col + '_sin'
            X[name] = np.sin(2 * np.pi * X[col] / 7)

            name = col + '_cos'
            X[name] = np.cos(2 * np.pi * X[col] / 7)

            del X[col]

        return X

    def get_feature_names(self):

        return self.feature_names



class dummy(BaseEstimator, TransformerMixin):

	def __init__(self,freq_cutoff=0):

		self.freq_cutoff=freq_cutoff
		self.var_cat_dict={}
		self.feature_names=[]

	def fit(self,x,y=None):

		data_cols=x.columns

		for col in data_cols:

			k=x[col].value_counts()

			if (k<=self.freq_cutoff).sum()==0:
				cats=k.index[:-1]

			else:
				cats=k.index[k>self.freq_cutoff]

			self.var_cat_dict[col]=cats

		for col in self.var_cat_dict.keys():
			for cat in self.var_cat_dict[col]:
				self.feature_names.append(col+'_'+cat)

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
	

class MissingValues(BaseEstimator,TransformerMixin):

	def __init__(self):

		self.impute_dict={}
		self.feature_names=[]

	def fit(self, X, y=None):

		self.feature_names=X.columns

		for col in X.columns:
			self.impute_dict[col]=X[col].median()
		return self

	def transform(self, X, y=None):
		return X.fillna(self.impute_dict)

	def get_feature_names(self):

		return self.feature_names


class pdPipeline(Pipeline):

    def get_feature_names(self):

        last_step = self.steps[-1][-1]

        return last_step.get_feature_names()