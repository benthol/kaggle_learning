#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 29 20:57:32 2020

@author: AEG
"""

import pandas as pd
import numpy as np
import seaborn as sns
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error as mse
from sklearn.preprocessing import LabelEncoder


# データの読み込み
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

train_x = train.drop(['y'], axis=1)
train_y = train['y']
test_x = test.copy()

# データ形式の変換
train_x['host_response_rate'] = train_x['host_response_rate'].replace('%', '', regex=True).astype(float)/100
train_x['first_review'] = pd.to_datetime(train_x['first_review'])
train_x['host_since'] = pd.to_datetime(train_x['host_since'])
train_x['last_review'] = pd.to_datetime(train_x['last_review'])
test_x['host_response_rate'] = test_x['host_response_rate'].replace('%', '', regex=True).astype(float)/100
test_x['first_review'] = pd.to_datetime(test_x['first_review'])
test_x['host_since'] = pd.to_datetime(test_x['host_since'])
test_x['last_review'] = pd.to_datetime(test_x['last_review'])

# データを可視化
#train_y.hist(bins=100)
#train_x['number_of_reviews'].hist(bins=50)
#train_x['review_scores_rating'].hist(bins=50)
#train.plot.scatter('accommodates', 'y')
#train.plot.scatter('bathrooms', 'y')
#train.plot.scatter('bedrooms', 'y')
#train.plot.scatter('beds', 'y')
#train.plot.scatter('latitude', 'y')
#train.plot.scatter('longitude', 'y')
#train.plot.scatter('longitude', 'latitude')
#train.plot.scatter('number_of_reviews', 'y')
#train.plot.scatter('review_scores_rating', 'y')

print(train['accommodates'].describe())
print(train[['bathrooms', 'bedrooms', 'beds']].describe())
print(train[['number_of_reviews', 'review_scores_rating']].describe())

# 
for c in train_x.columns:
    if train_x[c].dtype == 'object':
        lbl = LabelEncoder() 
        lbl.fit(list(train_x[c].values) + list(test_x[c].values)) 
        train_x[c] = lbl.transform(list(train_x[c].values))
        test_x[c] =  lbl.transform(list(test_x[c].values))

score_list = []
# 学習データ、バリデーションデータの分割
kf = KFold(n_splits=4, shuffle=True, random_state=1)
for i in range(4):
    tr_idx, va_idx = list(kf.split(train_x))[i]

    tr_x, va_x = train_x.iloc[tr_idx], train_x.iloc[va_idx]
    tr_y, va_y = train_y.iloc[tr_idx], train_y.iloc[va_idx]

    categorical_features = ['first_review', 'host_since', 'last_review']

    tr_x = tr_x.drop(categorical_features, axis=1)
    va_x = va_x.drop(categorical_features, axis=1)

    # 特徴量と目的変数の変換
    lgb_train = lgb.Dataset(tr_x, tr_y)
    lgb_eval = lgb.Dataset(va_x, va_y)

    # ハイパーパラメータの設定
    params = {'objective': 'regression', 'seed': 1, 'verbose': 0, 'metrics': 'rmse'}
    num_round = 200

    # 学習の実行
    """
    model = lgb.train(
        params, lgb_train, num_boost_round=num_round, 
        valid_names=['train', 'vaild'], valid_sets=[lgb_train, lgb_eval],
        early_stopping_rounds=50)
    """
    model = lgb.LGBMRegressor(objective='rmse')
    model.fit(tr_x, tr_y, eval_set=[(va_x, va_y), (tr_x, tr_y)], verbose=10)
    

    # バリデーションデータでのスコアの確認
    va_pred = model.predict(va_x)
    score = np.sqrt(mse(va_y, va_pred))
    score_list.append(score)

score_ave = np.mean(score_list)
print(f'RMSE: {score_ave:.4f}')

# 学習曲線
lgb.plot_metric(model, metric='rmse')

# 提出用データ
tr_x = train_x.drop(categorical_features, axis=1)
tr_y = train_y
model = lgb.LGBMRegressor(objective='rmse')
model.fit(tr_x, tr_y, verbose=10)

ts_x = test_x.drop(categorical_features, axis=1)
ts_pred = model.predict(ts_x)
output = pd.DataFrame(ts_pred)
output.to_csv('submit.csv', header=False, sep=',')
