#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 17 20:17:22 2020

@author: hirokawaeiji
"""

import pandas as pd
import sweetviz as sv

# データの読み込み
train = pd.read_csv('train.csv', index_col=0)
test = pd.read_csv('test.csv', index_col=0)

my_report = sv.compare([train, "Training Data"], [test, "Test Data"], "SalePrice")
my_report.show_html()