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
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import OneHotEncoder
# import xgboost as xgb
import lightgbm as lgb

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
    # print ("remove %d columns" % len(to_remove))
    return X

def RemoveNAN(X,threshold):
    to_remove = []
    total = len(X)
    for col in X.columns:
        # remove columns which has more than 90% nan
        if X[col].count() * 1.0 / total < threshold:
            to_remove.append(col)
    X = X.drop(to_remove,axis=1)
    # print ("remove %d columns" % len(to_remove))
    return X

# remove only two vals(NAN & other val)
def RemoveTwoVals(X):
    to_remove = []
    for col in X.columns:
        if len(X[col].unique()) == 2:
            if np.nan in set(X[col].unique()):
                to_remove.append(col)
    # print("remove %d columns" % len(to_remove))
    X = X.drop(to_remove,axis=1)
    return X

# delete year columns
def RemoveYearColumns(basic_info):
    cols = [col for col in basic_info.columns if basic_info[col].dtypes != object and len(basic_info[basic_info[col] > 2000]) * 1.0 / basic_info[col].count() > 0.9 and col != 'ccx_id']
    # print("remove %d columns" % len(cols))
    basic_info = basic_info.drop(cols,axis=1)
    return basic_info

# process behaivor
def GetBehavior(basic_info):
    basic_info = RemoveUnique(basic_info)
    basic_info = RemoveNAN(basic_info,0.2)
    basic_info = RemoveTwoVals(basic_info)
    basic_info = RemoveYearColumns(basic_info)
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
    cost.rename(columns={'V_1':'cost_times'},inplace=True)
    return cost

# get each user cost each times(the same date)
def GetEachUserCostEachDate(consuming):
    consuming['cost'] = consuming['V_12'] * consuming['V_13']
    temp = consuming.groupby(['ccx_id','V_7'])['cost'].sum().reset_index()
    total_cost = temp.groupby('ccx_id')['cost'].sum().reset_index()
    times = temp.groupby('ccx_id')['cost'].count().reset_index()
    times = times.rename(columns={'cost':'times'})
    cost_each_time = pd.DataFrame(columns=['ccx_id','cost_each_time'])
    cost_each_time['ccx_id'] = total_cost['ccx_id'].unique()
    cost_each_time['cost_each_time'] = total_cost['cost'] / times['times']
    return cost_each_time

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

'''
# convert to categorical of consuming
def ConvertToCategorical(X,threshold):
    has = ['ccx_id','times','cost','cost_each_time']
    cols = [col for col in X.columns if col not in has]
    to_categorical_cols = [col for col in cols if len(X[col].unique()) < threshold]
    if len(to_categorical_cols) == 0:
        return X
    enc = OneHotEncoder(sparse=False)
    temp = pd.DataFrame(enc.fit_transform(X[to_categorical_cols]))
    temp['ccx_id'] = X['ccx_id']
    X = X.drop(columns=to_categorical_cols,axis=1)
    X = pd.merge(X,temp)
    return X
'''

# remove 0 columns from consuming
def RemoveZero(X,threshold):
    to_remove = []
    total = len(X)
    for col in X.columns:
        # remove columns which has more than 90% nan
        if len(X[X[col] != 0]) * 1.0 / total < threshold:
            to_remove.append(col)
    X = X.drop(to_remove,axis=1)
    # print("remove %d columns" % len(to_remove))
    return X

# process consuming
def GetConsuming(consuming_info):
    consuming = DeleteComplicate(consuming_info)
    # consuming
    # max_pay = pd.DataFrame(consuming.groupby(['ccx_id'])['V_5'].max().reset_index())
    # max_pay = max_pay.rename(columns={'V_5':'max_pay'})
    # min_pay = pd.DataFrame(consuming.groupby(['ccx_id'])['V_5'].min().reset_index())
    # min_pay = min_pay.rename(columns={'V_5':'min_pay'})
    # pay = pd.merge(max_pay,min_pay)
    # pay = max_pay
    # pay['difference'] = pay['max_pay'] - pay['min_pay']
    cost = GenerateCostFeatures(consuming)
    res = pd.DataFrame(GenerateCategoricalFeatures(consuming))
    cost_each_date = GetEachUserCostEachDate(consuming)
    res = pd.merge(cost,res)
    res = pd.merge(res,cost_each_date)
    consuming_info = RemoveZero(res,0.2)
    # consuming_info = RemoveZero(pd.merge(res,pay),0.2)
    return consuming_info

# process query
# generate query data
def GetQueryFeatures(query_info,uids):
    res = pd.DataFrame(columns=['ccx_id'])
    res['ccx_id'] = uids
    query_info['count'] = 1
    categorical_cols = ['var_01','var_02','var_03','var_04','var_05']
    for col in categorical_cols:
        temp = query_info.groupby(['ccx_id',col])['count'].sum().reset_index().pivot_table(index='ccx_id',columns=col).fillna(0)
        uid = temp.index
        temp = pd.DataFrame(temp.values,columns=[temp.columns[i][1] for i in range(len(temp.columns))]).fillna(0)
        temp['ccx_id'] = uid
        res = pd.merge(res,temp,how='left')
    # query times of each user
    temp = query_info.groupby('ccx_id')['count'].sum().reset_index()
    temp.rename(columns={'count':'query_times'},inplace=True)
    res = pd.merge(res,temp,how='left').fillna(0)
    # remove 0 columns from query
    # res = RemoveZero(res,0.01)
    return res

# process data
def PreProcess(X,flag=True):
    '''

    :param X:
    :param flag: normalize data (true or flase)
    :return:
    '''
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

    if flag:
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
    # train & test 0.65 0.35
    # train_x,test_x,train_y,test_y = train_test_split(X,Y,test_size=0.35,random_state=0)
    # reg = Train(reg,train_x,train_y)
    # pred = reg.predict_proba(PreProcess(test_x))[:,1]
    # return roc_auc_score(test_y,pred)

    # cross validation
    kf = KFold(n_splits=n)
    kf.get_n_splits(X)
    for train_index, test_index in kf.split(X):
        X_train,X_test = X.iloc[train_index], X.iloc[test_index]
        Y_train,Y_test = Y.iloc[train_index], Y.iloc[test_index]
        # logitics regression
        # regeression = Train(reg,X_train,Y_train)
        # pred = regeression.predict_proba(PreProcess(X_test))[:,1]
        # gbdt
        est = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=0, loss='ls').fit(PreProcess(X_train,False), Y_train)
        pred = est.predict(PreProcess(X_test,False))
        # print(pred)
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
def Test(train_consumer_A,train_behavior_A,train_ccx_A,train_consumer_B,train_behavior_B):
    reg = LogisticRegression(max_iter=1000)
    # read behaivor
    basic_info = GetBehavior(train_behavior_A)
    print("behavior has %d features" % len(basic_info.columns))
    # read consuming
    consuming_info = GetConsuming(train_consumer_A)
    print("consuming has %d features" % len(consuming_info.columns))
    info = pd.merge(basic_info,consuming_info,how='left')
    # # read query
    uids = basic_info['ccx_id'].unique()
    query_info = GetQueryFeatures(train_ccx_A,uids)
    print("query has %d features" % len(query_info.columns))
    info = pd.merge(info,query_info,how='left')
    # info = basic_info
    info = pd.merge(info,Y,how="outer")
    label = info['target']
    features = [col for col in info.columns if col != 'ccx_id' and col != 'target']
    # print(len(features))

    # lightgbm
    # split train_data and test_data

    param = {'num_leaves':31, 'num_trees':100, 'objective':'binary','metric':'auc'}
    train_data = lgb.Dataset(PreProcess(info[features],False),label=label)
    print(np.mean((lgb.cv(param, train_data, 100, nfold=5))['auc-mean']))

    # res = Metric(reg,info[features],label,5)
    # res = MetricOnXgboost(info[features],label,5)
    # print(res)

# Genrate Results
def Run(test_consumer_A,test_behavior_A,test_ccx_A,test_consumer_B,test_behavior_B):
    # read behaivor

    train_A_index = len(train_behavior_A)
    test_A_index = len(test_behavior_A)

    behavior_info = pd.concat([train_behavior_A,test_behavior_A,test_behavior_B])
    basic_info = GetBehavior(behavior_info)
    # read consuming
    consuming_info = pd.concat([train_consumer_A,test_consumer_A,test_consumer_B])
    consuming_info = GetConsuming(consuming_info)

    # read ccx_A
    ccx_A = pd.concat([train_ccx_A,test_ccx_A])
    uids = basic_info['ccx_id'].unique()
    query_info = GetQueryFeatures(ccx_A,uids)

    info = pd.merge(basic_info,consuming_info,how='left')
    info = pd.merge(info,query_info,how='left')
    info = pd.merge(info,Y,how="outer")
    label = info.iloc[:train_A_index]['target']
    features = [col for col in info.columns if col != 'target']

    # lightgbm
    param = {'num_leaves':31, 'num_trees':100, 'objective':'binary','metric':'auc'}
    train_data = lgb.Dataset(PreProcess(info.iloc[:train_A_index][features],False),label=label)
    bst = lgb.train(param,train_data,100)

    predict_result_A = pd.DataFrame(columns=['ccx_id','prob'])
    predict_result_A['ccx_id'] = info.iloc[train_A_index:train_A_index+test_A_index]['ccx_id'].unique()


    # lightgbm

    predict_result_A['prob'] = bst.predict(PreProcess(info.iloc[train_A_index:train_A_index+test_A_index][features],False))

    predict_result_A.to_csv('./predict_result_A.csv',encoding='utf-8',index=False)

    predict_result_B = pd.DataFrame(columns=['ccx_id','prob'])
    predict_result_B['ccx_id'] = info.iloc[train_A_index+test_A_index:]['ccx_id'].unique()

    # lightgbm
    # retrain
    features_B = [col for col in features if col != 'ccx_id']
    train_data = lgb.Dataset(PreProcess(info.iloc[:train_A_index][features_B],False),label=label)
    bst = lgb.train(param,train_data,100)

    predict_result_B['prob'] = bst.predict(PreProcess(info.iloc[train_A_index+test_A_index:][features_B],False))

    predict_result_B.to_csv('./predict_result_B.csv',encoding='utf-8',index=False)
    # pred_A = reg.predict()
Test(train_consumer_A,train_behavior_A,train_ccx_A,train_consumer_B,train_behavior_B)
# Run(test_consumer_A,test_behavior_A,test_ccx_A,test_consumer_B,test_behavior_B)