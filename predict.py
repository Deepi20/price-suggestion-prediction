test = fread('https://s3.amazonaws.com/hackerday.datascience/109/test.tsv')
train = fread('https://s3.amazonaws.com/hackerday.datascience/109/train.tsv')


import gc
import time
import numpy as np
import pandas as pd

from scipy.sparse import csr_matrix, hstack

from sklearn.linear_model import Ridge
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split, cross_val_score

import lightgbm as lgb

NUM_BRANDS = 4000
NUM_CATEGORIES = 1000
NAME_MIN_DF = 10
MAX_FEATURES_ITEM_DESCRIPTION = 50000

def handle_missing_inplace(dataset):
    dataset['category_name'].fillna(value='missing', inplace=True)
    dataset['brand_name'].fillna(value='missing', inplace=True)
    dataset['item_description'].fillna(value='missing', inplace=True)

def cutting(dataset):
    pop_brand = dataset['brand_name'].value_counts().loc[lambda x: x.index != 'missing'].index[:NUM_BRANDS]
    dataset.loc[~dataset['brand_name'].isin(pop_brand), 'brand_name'] = 'missing'
    pop_category = dataset['category_name'].value_counts().loc[lambda x: x.index != 'missing'].index[:NUM_BRANDS]
 

def to_categorical(dataset):
    dataset['category_name'] = dataset['category_name'].astype('category')
    dataset['brand_name'] = dataset['brand_name'].astype('category')
    dataset['item_condition_id'] = dataset['item_condition_id'].astype('category')


train = pd.read_table('https://s3.amazonaws.com/hackerday.datascience/109/train.tsv', engine='c')
    test = pd.read_table('https://s3.amazonaws.com/hackerday.datascience/109/test.tsv', engine='c')
    print('[{}] Finished to load data'.format(time.time() - start_time))
    print('Train shape: ', train.shape)
    print('Test shape: ', test.shape)


nrow_train = train.shape[0]
    y = np.log1p(train["price"])
    merge: pd.DataFrame = pd.concat([train, test])
    submission: pd.DataFrame = test[['test_id']]

del train
    del test
    gc.collect()

handle_missing_inplace(merge)
    

cutting(merge)
    

to_categorical(merge)
    

cv = CountVectorizer(min_df=NAME_MIN_DF)
    X_name = cv.fit_transform(merge['name'])
    

cv = CountVectorizer()
    X_category = cv.fit_transform(merge['category_name'])
    

tv = TfidfVectorizer(max_features=MAX_FEATURES_ITEM_DESCRIPTION,
                         ngram_range=(1, 3),
                         stop_words='english')
    X_description = tv.fit_transform(merge['item_description'])
    

lb = LabelBinarizer(sparse_output=True)
    X_brand = lb.fit_transform(merge['brand_name'])
    

X_dummies = csr_matrix(pd.get_dummies(merge[['item_condition_id', 'shipping']],
                                          sparse=True).values)
    

sparse_merge = hstack((X_dummies, X_description, X_brand, X_category, X_name)).tocsr()
    

X = sparse_merge[:nrow_train]
    X_test = sparse_merge[nrow_train:]
    

    d_train = lgb.Dataset(X, label=y, max_bin=8192)
    

params = {
        'learning_rate': 0.75,
        'application': 'regression',
        'max_depth': 3,
        'num_leaves': 100,
        'verbosity': -1,
        'metric': 'RMSE',
    }

model = lgb.train(params, train_set=d_train, num_boost_round=3200,  \
    verbose_eval=100) 
    preds = 0.6*model.predict(X_test)



model = Ridge(solver="sag", fit_intercept=True, random_state=205)
    model.fit(X, y)
    

    preds += 0.4*model.predict(X=X_test)
