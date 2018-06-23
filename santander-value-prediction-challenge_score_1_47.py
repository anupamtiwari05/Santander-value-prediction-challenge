# Importing required modules
import numpy as np
import pandas as pd 
import gc
from lightgbm import LGBMRegressor
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# importing data files
train_df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')
print("Train samples: {}, test samples: {}".format(len(train_df), len(test_df)))

#Checking columns with constant values
unique_df = train_df.nunique().reset_index()
unique_df.columns = ["col_name", "unique_count"]
constant_df = unique_df[unique_df["unique_count"]==1]

# Getting train/test data after removing not required fetures/columns
train_X = train_df.drop(constant_df.col_name.tolist() + ["ID", "target"], axis=1)
test_X = test_df.drop(constant_df.col_name.tolist() + ["ID"], axis=1)
train_y = np.log1p(train_df["target"].values)

# Splitting train data into training and validation set
Tr_X, val_X, Tr_y, val_y = train_test_split(train_X, train_y, test_size = 0.2, random_state = 42)

# Starting building model 
model = LGBMRegressor(boosting_type='gbdt', 
	num_leaves=50, 
	max_depth=-8, 
	learning_rate=0.02, 
	n_estimators=4000, 
	max_bin=255, 
	subsample_for_bin=50000, 
	objective='regression', 
	min_split_gain=0.0222415, 
	min_child_weight=30,
	min_child_samples=10, 
	subsample=0.8715623, 
	subsample_freq=1, 
	colsample_bytree=0.7, 
	reg_alpha=0.1, 
	reg_lambda=0, 
	seed=17,
	silent=False, 
	nthread=4)

model.fit(Tr_X, Tr_y, eval_set=[(val_X, val_y)], eval_metric= 'rmse', verbose= 100, early_stopping_rounds= 200)
print(model.best_iteration_)
preds = model.predict(test_X, num_iteration=model.best_iteration_)

# Write submission file and plot feature importance
test_df['target'] = np.expm1(preds)
test_df[['ID', 'target']].to_csv("submission_1.csv", index= False)