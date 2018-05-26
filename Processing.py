#!/usr/bin/python
#-*-coding:utf-8-*-
'''@author:duncan'''

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing


train_consumer_A = pd.read_csv("./train/scene_A/train_consumer_A.csv")
train_behavior_A = pd.read_csv('./train/scene_A/train_behavior_A.csv')
train_ccx_A = pd.read_csv("./train/scene_A/train_ccx_A.csv")
train_consumer_B = pd.read_csv("./train/scene_B/train_consumer_B.csv")
train_behavior_B = pd.read_csv('./train/scene_B/train_behavior_B.csv')

test_consumer_A = pd.read_csv("./test/scene_A/test_consumer_A.csv")
test_behavior_A = pd.read_csv('./test/scene_A/test_behavior_A.csv')
test_ccx_A = pd.read_csv("./test/scene_A/test_ccx_A.csv")
test_consumer_B = pd.read_csv("./test/scene_B/test_consumer_B.csv")
test_behavior_B = pd.read_csv('./test/scene_B/test_behavior_B.csv')
# read labels
Y = pd.read_csv('./train/scene_A/train_target_A.csv')


def RemoveUnique(X):
    to_remove = []
    cols = X.columns
    for col in cols:
        if len(X[col].unique()) == 1:
            to_remove.append(col)
    X = X.drop(to_remove,axis=1)
    print ("remove %d columns" % len(to_remove))
    return X

def RemoveNAN(X):
    to_remove = []
    total = len(X)
    for col in X.columns:
        # remove columns which has more than 90% nan
        if X[col].count() * 1.0 / total < 0.01:
            to_remove.append(col)
    X = X.drop(to_remove,axis=1)
    print ("remove %d columns" % len(to_remove))
    return X

# remove only two vals(NAN & other val)
def RemoveTwoVals(X):
    to_remove = []
    for col in X.columns:
        if len(X[col].unique()) == 2:
            if np.nan in set(X[col].unique()):
                to_remove.append(col)
    print("containing %d columns" % len(to_remove))
    X = X.drop(to_remove,axis=1)
    return X

# process behaivor
def GetBehavior(basic_info):
    basic_info = RemoveUnique(basic_info)
    basic_info = RemoveNAN(basic_info)
    # basic_info = RemoveTwoVals(basic_info)
    return basic_info


# process consuming
# remove duplicated rows and keep the last
def DeleteComplicate(consuming):
    compare_cols = [col for col in consuming.columns if col != 'V_11']
    consuming = consuming[compare_cols].drop_duplicates(keep='last')
    return consuming

# calc the total cost of each user
def GenerateCostFeatures(consuming):
    cost = pd.DataFrame(columns=['ccx_id'])
    cost['ccx_id'] = consuming['ccx_id'].unique()
    consuming['cost'] = consuming['V_12'] * consuming['V_13']
    cost['ccx_id'] = consuming['ccx_id'].unique()
    group = consuming.groupby('ccx_id')['cost'].sum().reset_index()
    # group['ccx_id'] = group.index
    cost = pd.merge(group,cost)
    times = consuming.groupby('ccx_id')['V_1'].count().reset_index()
    cost = pd.merge(times,cost)
    cost.rename(columns={'V_1':'times'},inplace=True)
    return cost

# calc the times of each value(V_1,V_2,V_8,V_14)
def GenerateCategoricalFeatures(consuming):
    res = pd.DataFrame(columns=['ccx_id'])
    res['ccx_id'] = consuming['ccx_id'].unique()
    categorical_cols = ['V_1','V_2','V_8','V_14']
    for col in categorical_cols:
        temp = consuming[['ccx_id',col]]
        temp['count'] = 1
        temp = temp.groupby(["ccx_id",col])["count"].sum().reset_index().pivot_table(index='ccx_id',columns=col).fillna(0)
        cols = [temp.columns[i][1] for i in range(len(temp.columns))]
        temp = pd.DataFrame(temp.values,columns=cols)
        uid = [temp.index[i] for i in range(len(temp.index))]
        temp = pd.DataFrame(temp.values,columns=cols)
        temp['ccx_id'] = uid
        res = pd.merge(res,temp,on='ccx_id',how='left').fillna(0)
    return res

# process consuming
def GetConsuming(consuming_info):
    consuming_info = DeleteComplicate(consuming_info)
    cost = GenerateCostFeatures(consuming_info)
    res = pd.DataFrame(GenerateCategoricalFeatures(consuming_info))
    consuming_info = pd.merge(cost,res)
    return consuming_info

# process data
def PreProcess(X):
    le = LabelEncoder()
    # convert 'object'
    object_cols = X.columns[X.dtypes == object]

    # delete object_cols
    # X = X.drop(object_cols,axis=1)

    for col in object_cols:
        # fill nan
        X[col] = X[col].fillna("-1")
        le.fit(X[col].astype(str))
        X[col] = le.transform(X[col].astype(str))

    # fill nan
    X = X.fillna(-1)

    # standardlization
    min_max_scaler = preprocessing.MinMaxScaler()
    X = min_max_scaler.fit_transform(X)
    return X

def Train(regression,X,Y):
    X = PreProcess(X)
    regression = regression.fit(X,Y)
    return regression

xgb_params = {
    'eta': 0.05,
    'max_depth': 5,
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'silent': 1
}

# metric on sklearn regression
def Metric(reg,X,Y,n):
    auc = 0
    # cross validation
    kf = KFold(n_splits=n)
    kf.get_n_splits(X)
    for train_index, test_index in kf.split(X):
        X_train,X_test = X.iloc[train_index], X.iloc[test_index]
        Y_train,Y_test = Y.iloc[train_index], Y.iloc[test_index]
        reg = Train(reg,X_train,Y_train)
        pred = reg.predict_proba(PreProcess(X_test))[:,0]
        print(pred)
        auc += roc_auc_score(Y_test,pred)
    # compute average auc
    return auc / n

'''
# metric on xgboost
def MetricOnXgboost(X,Y,n):
    auc = 0
    # cross validation
    kf = KFold(n_splits=n)
    kf.get_n_splits(X)
    for train_index, test_index in kf.split(X):
        X_train,X_test = X.iloc[train_index], X.iloc[test_index]
        Y_train,Y_test = Y.iloc[train_index], Y.iloc[test_index]
        dtrain = xgb.DMatrix(PreProcess(X_train),label=Y_train)
        dtest = xgb.DMatrix(PreProcess(X_test))
        # evallist  = [(dtest,'eval'), (dtrain,'train')]
        bst = xgb.train(xgb_params,dtrain,500)
        pred = bst.predict(dtest)
        print(pred)
        auc += roc_auc_score(Y_test,pred)
    # compute average auc
    return auc / n
'''

# Run
def Test(test_consumer_A,test_behavior_A,test_ccx_A,test_consumer_B,test_behavior_B):
    # read behaivor
    basic_info = GetBehavior(test_behavior_A)
    # read consuming
    consuming_info = GetConsuming(test_consumer_A)
    reg = LogisticRegression(max_iter=1000)
    info = pd.merge(basic_info,consuming_info)
    # info = basic_info
    features = [col for col in info.columns if col != 'ccx_id']
    res = Metric(reg,info[features],Y['target'],5)

    # res = MetricOnXgboost(info[features],Y['target'],5)
    print(res)

# Genrate Results
def Run(test_consumer_A,test_behavior_A,test_ccx_A,test_consumer_B,test_behavior_B):
    # read behaivor
    basic_info = GetBehavior(train_behavior_A)
    # read consuming
    consuming_info = GetConsuming(train_consumer_A)
    reg = LogisticRegression()
    info = pd.merge(basic_info,consuming_info)
    features = [col for col in info.columns if col != 'ccx_id']
    reg = Train(reg,info[features],Y['target'])

    test_basic_info = GetBehavior(test_behavior_A)
    test_consuming_info = GetConsuming(test_consumer_A)
    test_info = pd.merge(test_basic_info,test_consuming_info)
    # fill features
    lost_features = set(features) - set(test_info.columns)
    for col in lost_features:
        test_info[col] = 0
    predict_result_A = pd.DataFrame(reg.predict_proba(PreProcess(test_info[features]))[:,0])
    predict_result_A.to_csv('./predict_result_A.csv',encoding='utf-8',index=False)
    predict_result_A.to_csv('./predict_result_B.csv',encoding='utf-8',index=False)
    # pred_A = reg.predict()
# Test(test_consumer_A,test_behavior_A,test_ccx_A,test_consumer_B,test_behavior_B)
Run(test_consumer_A,test_behavior_A,test_ccx_A,test_consumer_B,test_behavior_B)