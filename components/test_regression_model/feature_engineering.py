'''
Additional module for helper classes and functions.
'''

import pandas as pd


from sklearn.base import BaseEstimator, TransformerMixin

class MeanTargetEncoder(BaseEstimator, TransformerMixin):
    '''
    MeanTargetEncoder class for mean encoding high-cardinality categorical feature
    encoding categories that fall within given quartile.
    '''
    
    def __init__(self, quantile = 0.9):
        self.quantile = quantile
        self.encoder = None
        self.within_quantile = None
    
    def fit(self, X, y = None):
        
        if y is None:
            raise ValueError(f'Target values should be passed to MeanTargetEncoder')
        
        df_ = pd.DataFrame(data={'neighbourhood': X, 'target': y})
        value_counts = df_['neighbourhood'].value_counts()
        within_quantile = value_counts[value_counts.cumsum() <= QUANTILE * df_.shape[0]].index.to_list()
        
        df_['neighbourhood_new'] = df_['neighbourhood'].map(lambda x: x if x in within_quantile else 'Other')
        encoder = df_.groupby('neighbourhood_new')['target'].mean().to_dict()
        self.within_quantile = within_quantile
        self.encoder = encoder
        
        return self
    
    def transform(self, X, y=None):
        if self.encoder is None or self.within_quantile is None:
            raise ValueError('MeanTargetEncoder should be fit before transform.')
            
        X_ = pd.Series(X)
        X_ = X_.map(lambda x: x if x in self.within_quantile else 'Other')
        X_ = X_.map(encoder)
        
        try:
            assert X_.isna().sum() == 0
        except AssertionError as e:
            raise ValueError('Some weird error in transforming with MeanTargetEncoder')
        
        X_ = X_.to_numpy().reshape(-1, 1)
        
        return X_

def delta_date_feature(dates):
    '''
    Given a 2d array containing dates (in any format recognized by pd.to_datetime), it returns the delta in days
    between each date and the most recent date in its column
    '''
    date_sanitized = pd.DataFrame(dates).apply(pd.to_datetime)
    return date_sanitized.apply(lambda d: (d.max() -d).dt.days, axis=0).to_numpy()
