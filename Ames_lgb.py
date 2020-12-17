#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 10 20:45:24 2020

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
train = pd.read_csv('train.csv', index_col=0)
test = pd.read_csv('test.csv', index_col=0)

# 外れ値の除外
train_copy = train.copy()
train_copy = train_copy[train_copy['1st Flr SF'] < 2000]
train_copy = train_copy[train_copy['SalePrice'] < 300000]
train_copy = train_copy[train_copy['SalePrice'] > 100000]

# 外れ値のクリッピング
"""
num_cols = train.select_dtypes(include=float).columns
p01 = train_copy[num_cols].quantile(0.01)
p99 = train_copy[num_cols].quantile(0.99)
train_copy[num_cols] = train_copy[num_cols].clip(p01, p99, axis=1)
test_copy[num_cols] = test_copy[num_cols].clip(p01, p99, axis=1)
"""

# 特徴量と目的変数の分離
train_x = train_copy.drop(['SalePrice'], axis=1)
train_y = train_copy['SalePrice']
test_x = test.copy()

# 不要な特徴量を落とす
drop_feature = ['MS SubClass']
train_x = train_x.drop(drop_feature, axis=1)
test_x = test_x.drop(drop_feature, axis=1)

# ラベルエンコード
for c in train_x.columns:
    if train_x[c].dtype == 'object':
        lbl = LabelEncoder() 
        lbl.fit(list(train_x[c].values) + list(test_x[c].values)) 
        train_x[c] = lbl.transform(list(train_x[c].values))
        test_x[c] =  lbl.transform(list(test_x[c].values))

score_list = []

# 学習データ、バリデーションデータの分割
kf = KFold(n_splits=4, shuffle=True, random_state=1)
for i, (tr_idx, va_idx) in enumerate(kf.split(train_x)):

    tr_x, va_x = train_x.iloc[tr_idx].copy(), train_x.iloc[va_idx].copy()
    tr_y, va_y = train_y.iloc[tr_idx], train_y.iloc[va_idx]
    """
    # 変数をループしてtarget encoding
    for c in tr_x.columns:
        if tr_x[c].dtype == 'object':
            # 学習データ全体で各カテゴリにおけるtargetの平均を計算
            data_tmp = pd.DataFrame({c: tr_x[c], 'target': tr_y})
            target_mean = data_tmp.groupby(c)['target'].mean()
            # バリデーションデータのカテゴリを置換
            va_x.loc[:, c] = va_x[c].map(target_mean)
            
            # 学習データの変換後の値を格納する配列を準備
            tmp = np.repeat(np.nan, tr_x.shape[0])
            kf_encoding = KFold(n_splits=4, shuffle=True, random_state=1)
            for idx_1, idx_2 in kf_encoding.split(tr_x):
                target_mean = data_tmp.iloc[idx_1].groupby(c)['target'].mean()
                tmp[idx_2] = tr_x[c].iloc[idx_2].map(target_mean)
            
            tr_x.loc[:, c] = tmp
    """
    
    # 特徴量と目的変数の変換
    lgb_train = lgb.Dataset(tr_x, tr_y)
    lgb_eval = lgb.Dataset(va_x, va_y)

    # 学習の実行
    model = lgb.LGBMRegressor(objective='rmse', early_stopping_rounds=50)
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
tr_x = train_x
tr_y = train_y
ts_x = test_x

"""
# 変数をループしてtarget encoding
for c in tr_x.columns:
    if tr_x[c].dtype == 'object':
        # 学習データ全体で各カテゴリにおけるtargetの平均を計算
        data_tmp = pd.DataFrame({c: tr_x[c], 'target': tr_y})
        target_mean = data_tmp.groupby(c)['target'].mean()
        # バリデーションデータのカテゴリを置換
        ts_x.loc[:, c] = ts_x[c].map(target_mean)
        
        # 学習データの変換後の値を格納する配列を準備
        tmp = np.repeat(np.nan, tr_x.shape[0])
        kf_encoding = KFold(n_splits=4, shuffle=True, random_state=1)
        for idx_1, idx_2 in kf_encoding.split(tr_x):
            target_mean = data_tmp.iloc[idx_1].groupby(c)['target'].mean()
            tmp[idx_2] = tr_x[c].iloc[idx_2].map(target_mean)
            
        tr_x.loc[:, c] = tmp
"""

model = lgb.LGBMRegressor(objective='rmse')
model.fit(tr_x, tr_y, verbose=10)

ts_pred = model.predict(ts_x)
output = pd.DataFrame(ts_pred, index=ts_x.index)
output.to_csv('submit.csv', header=False, sep=',')
