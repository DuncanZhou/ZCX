<<<<<<< HEAD
#!/usr/bin/python
#-*-coding:utf-8-*-
'''@author:duncan'''

import pandas as pd
import numpy as np
#from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing
#from sklearn.cross_validation import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
#from sklearn.preprocessing import OneHotEncoder
# import xgboost as xgb
import lightgbm as lgb
#from pyfm import pylibfm
from sklearn.feature_extraction import DictVectorizer
from catboost import CatBoostRegressor

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


params = {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': {'auc'},
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': 1,
}

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

# count the bad data
def CountBadCount(df):
    c = 'V_11'
    temp = df[["ccx_id", c]]
    temp = temp[pd.isnull(temp[c]) | (temp[c] == "0000-00-00 00:00:00")]
    newc = c + "_bad_count"
    temp = temp.groupby("ccx_id")[c].count().reset_index().rename(columns={c: newc})
    return temp

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
    # count the bad data
    # temp = CountBadCount(consuming_info)
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
    # consuming_info = pd.merge(consuming_info,temp,how="left").fillna(0)
    # consuming_info = RemoveZero(pd.merge(res,pay),0.2)
    return consuming_info

# process query
# count query time in each month
def CountOneMonth(data,month):
    cur = pd.DataFrame(columns={'ccx_id'})
    cur['ccx_id'] = data['ccx_id'].unique()
    res = data['2017-%d' % month:'2017-%d' % (month+1)].groupby('ccx_id')['count'].sum().reset_index().rename(columns={'count':'month%d'%month})
    cur = pd.merge(cur,res,how='left').fillna(0)
    return cur

# total month
def CountQueryTimesEachMonth(data):
    cur = pd.DataFrame(columns={'ccx_id'})
    cur['ccx_id'] = data['ccx_id'].unique()
    months = [1,2,3,4,5]
    for m in months:
        temp = CountOneMonth(data,m)
        cur = pd.merge(cur,temp,how='left')
    return cur

# count query time
def QueryTimes(data):
    date = pd.to_datetime(data['var_06'])
    temp = data
    temp['query_date'] = date
    temp.set_index('query_date',inplace=True)
    return CountQueryTimesEachMonth(temp)

# generate query data
def GetQueryFeatures(query_info,uids):
    res = pd.DataFrame(columns=['ccx_id'])
    res['ccx_id'] = uids
    query_info['count'] = 1
    categorical_cols = ['var_01','var_02','var_03','var_04','var_05']
    # categorical_cols = ['var_01','var_02']
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
    res = RemoveZero(res,0.01)
    # query times each month ([1...5])
    query_times_each_month = QueryTimes(query_info)
    res = pd.merge(res,query_times_each_month,how='left')
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

# xgb_params = {
#     'eta': 0.05,
#     'max_depth': 5,
#     'subsample': 0.7,
#     'colsample_bytree': 0.7,
#     'objective': 'binary:logistic',
#     'eval_metric': 'auc',
#     'silent': 1
# }

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
    for train_index, test_index in kf.split(X):
        X_train,X_test = X.iloc[train_index], X.iloc[test_index]
        Y_train,Y_test = Y.iloc[train_index], Y.iloc[test_index]
        # logitics regression
        # regeression = Train(reg,X_train,Y_train)
        # pred = regeression.predict_proba(PreProcess(X_test))[:,1]
        # gbdt
        est = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=0, loss='ls').fit(PreProcess(X_train,False), Y_train)
        pred = est.predict(PreProcess(X_test,False))

        # lightgbm
        # merge X_train and B_info


        # fm
        # 需要转换成字典格式
        # v = DictVectorizer()
        # # print(PreProcess(X_train,False).to_dict("records"))
        # train = v.fit_transform(PreProcess(X_train,False).to_dict("records"))
        # reg.fit(train,np.asarray(Y_train.values))
        # pred = reg.predict(v.transform(PreProcess(X_test,False).to_dict("records")))
        # print(pred)
        auc += roc_auc_score(Y_test,pred)
    # compute average auc
    return auc / n

def MetricLGBWithB_data(info,n,B_info=None):
    auc = 0
    features = [col for col in info.columns if col != 'ccx_id' and col != 'target']
    X = info[features]
    Y = info['target']

    # cross validation
    kf = KFold(n_splits=n,shuffle=True)
    for train_index, test_index in kf.split(X):
        X_train,X_test = X.iloc[train_index], X.iloc[test_index]
        Y_train,Y_test = Y.iloc[train_index], Y.iloc[test_index]

        # lightgbm
        # merge X_train and B_info

        # train_data = X_train
        # label = Y_train
        train_data = pd.concat([X_train,B_info[features]])
        label = pd.concat([Y_train,B_info['target']])

        train_data = lgb.Dataset(PreProcess(train_data,False),label=label)
        bst = lgb.train(params,train_data,num_boost_round=150)
        pred = bst.predict(PreProcess(X_test,False),num_iteration=bst.best_iteration)
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

# process the format of data (fm)
def FormatData(data):
    train = data.to_dict('records')
    # 需要转换成字典格式
    v = DictVectorizer()
    return v.fit_transform(train)

# label B
def LabelB(train_consumer_A,train_behavior_A,train_behavior_B,train_consumer_B,target,Max_Iteration,test_behavior_A,test_consumer_A,test_behavior_B=None,test_consumer_B=None):
    # process behaivor_info
    # test_behavior = pd.concat([test_behavior_A,test_behavior_B])
    test_behavior = test_behavior_A
    behavior_info = pd.concat([train_behavior_A,train_behavior_B,test_behavior])
    behavior_info = GetBehavior(behavior_info)

    # process consuming_info
    # test_consumer = pd.concat([test_consumer_A,test_consumer_B])
    test_consumer = test_consumer_A
    consuming_info = pd.concat([train_consumer_A,train_consumer_B,test_consumer])
    consuming_info = GetConsuming(consuming_info)

    info = pd.merge(behavior_info,consuming_info,how = 'left')
    # add the label of A
    info = pd.merge(info,target,how='left',on='ccx_id')
    features = [col for col in info.columns if col != 'target' and col != 'ccx_id']

    A_info = info[info.ccx_id.isin(train_behavior_A.ccx_id)]
    B_info = info[info.ccx_id.isin(train_behavior_B.ccx_id)]
    # iteration Max_iteration times
    # lightgbm

    iteration = 0
    train_info = A_info
    leave = B_info
    while iteration < Max_Iteration:
        train_data = lgb.Dataset(PreProcess(train_info[features],False),label=train_info['target'])
        bst = lgb.train(params,train_data,num_boost_round=150)
        # predict B
        pred = bst.predict(PreProcess(leave[features],False),num_iteration=bst.best_iteration)
        leave['predict'] = pred
        leave.loc[leave['predict'] >= 0.7,'target'] = leave.loc[leave['predict'] >= 0.7,'predict']
        leave.loc[leave['predict'] <= 0.1,'target'] = leave.loc[leave['predict'] <= 0.1,'predict']
        # add B(has labels) into traindata
        to_add = leave[~leave['target'].isnull()]
        leave = leave[leave['target'].isnull()]
        train_info = pd.concat([train_info,to_add])
        iteration += 1
        if(len(to_add) == 0):
            break
    print("iteration %d times" % iteration)
    train_info = train_info.drop(['predict'],axis=1)
    # set ccx_id of B is 0
    train_info.loc[train_info.ccx_id.isin(train_behavior_B.ccx_id),'ccx_id'] = 0
    # read test data
    test_info_A = info[info.ccx_id.isin(test_behavior_A.ccx_id)]
    # test_info_B = info[info.ccx_id.isin(test_behavior_B.ccx_id)]
    return train_info,test_info_A


# Run
def Test(train_consumer_A,train_behavior_A,train_ccx_A,train_consumer_B,train_behavior_B):

    # read behaivor
    basic_info = GetBehavior(train_behavior_A)
    print("behavior has %d features" % len(basic_info.columns))
    # read consuming
    consuming_info = GetConsuming(train_consumer_A)
    # fei_features = pd.read_csv("feats_0601.csv")

    print("consuming has %d features" % len(consuming_info.columns))
    info = pd.merge(basic_info,consuming_info,how='left')
    # # read query
    uids = basic_info['ccx_id'].unique()
    query_info = GetQueryFeatures(train_ccx_A,uids)
    print("query has %d features" % len(query_info.columns))
    # info = pd.merge(info,query_info,how='left')
    # info = pd.merge(info,fei_features,how="left")
    # info = basic_info
    info = pd.merge(info,Y,how="outer")
    label = info['target']
    features = [col for col in info.columns if col != 'target' and col != 'ccx_id']
    # print(len(features))

    # lightgbm
    # train_data = lgb.Dataset(PreProcess(info[features],False),label=label)
    # print("without B data auc is %.3f(using bahavior and consuming data)" % np.mean((lgb.cv(params, train_data, 150, nfold=5))['auc-mean']))
    # res = MetricLGBWithB_data(info,5,None)
    # print(res)


    # add B
    train_data_with_B,test = LabelB(train_consumer_A,train_behavior_A,train_behavior_B,train_consumer_B,10)
    train_behavior_B = pd.merge(train_data_with_B,query_info,how='left')
    # split A and B
    A_info = train_data_with_B[train_data_with_B['ccx_id'].isin(train_behavior_A['ccx_id'])]
    B_info = train_data_with_B[train_data_with_B.ccx_id.isin(train_behavior_B['ccx_id'])]
    res = MetricLGBWithB_data(A_info,5,B_info)
    print("5 Fold CV in A with B data is %.4f" % res)


    # features_with_B = [col for col in train_data_with_B.columns if col != 'target' and col != 'ccx_id']
    # train_data_with_B = pd.merge(train_data_with_B,query_info,how='left')
    # train_data_with_B = lgb.Dataset(PreProcess(train_data_with_B[features_with_B],False),label=train_data_with_B['target'])
    # print("with B data auc is %.3f(using bahavior and consuming data)" % np.mean((lgb.cv(params, train_data_with_B, 150, nfold=5))['auc-mean']))


    #
    # fm
    # fm = pylibfm.FM(num_factors=10, num_iter=100, verbose=True, task="regression", initial_learning_rate=0.001, learning_rate_schedule="optimal")
    # res = Metric(fm,info[features],label,5)
    # print(res)

    # res = Metric(reg,info[features],label,5)
    # res = MetricOnXgboost(info[features],label,5)
    # print(res)


def run_ffg():
    def preprocess_ccx_cons(ccx, cons):
        cons['V_7'] = cons['V_7'].str.replace('/', '-')
        ccx['var_06'] = ccx['var_06'].str.replace('/', '-')
        cons['year'] = cons['V_7'].apply(lambda x: int(x.split()[0].split('-')[0]))
        cons['month'] = cons['V_7'].apply(lambda x: int(x.split()[0].split('-')[1]))
        cons['date'] = cons['V_7'].apply(lambda x: x.split()[0])
        cons['time'] = cons['V_7'].apply(lambda x: x.split()[1])
        cons['month_from_now'] = cons.apply(lambda x: (2017 - x['year']) * 12 + (6 - x['month']), axis = 1)
        cons['day_from_now'] = (pd.to_datetime('2017-06-01') - pd.to_datetime(cons['date'])).apply(lambda x:x.days)
        cons['hour'] = cons['time'].apply(lambda x: int(x.split(':')[0]))
        ccx['year'] = ccx['var_06'].apply(lambda x: int(x.split('-')[0]))
        ccx['month'] = ccx['var_06'].apply(lambda x: int(x.split('-')[1]))
        ccx['month_from_now'] = ccx.apply(lambda x: (2017 - x['year']) * 12 + (6 - x['month']), axis = 1)
        ccx['day_from_now'] = (pd.to_datetime('2017-06-01') - pd.to_datetime(ccx['var_06'])).apply(lambda x:x.days)
        for cons_var in ['V_1', 'V_2', 'V_3', 'V_8', 'V_14']:
            _a = cons.groupby(cons_var)[cons_var].count() / (1.0 * len(cons))
            cons[cons_var + '_float'] = cons[cons_var].map(_a).fillna(0.0)
        for ccx_var in ['var_02', 'var_03', 'var_04', 'var_05']:
            _a = ccx.groupby(ccx_var)[ccx_var].count() / (1.0 * len(ccx))
            ccx[ccx_var + '_float'] = ccx[ccx_var].map(_a).fillna(0.0)

    def feats_b_behavior(b_a, b_b, b_a_test):
        feats = b_a.iloc[:,20:-1].columns.values
        for feat in feats:
            for b in [b_a, b_b, b_a_test]:
                b[feat] = b[feat].fillna(0)
        return feats

        # time = 'day_from_now' or 'month_from_now'
    def feats_stat_cons_time(cons, b_a, b_b, b_a_test, time):
        feats = []
        for agg_method in ['mean', 'std', 'min', 'max']:
            _a = cons.groupby('ccx_id')[time].agg(agg_method)
            feat = 'trade_{}_{}'.format(agg_method, time)
            for b in [b_a, b_b, b_a_test]:
                b[feat] = b['ccx_id'].map(_a).fillna(9999)
            feats.append(feat)
        return feats


    def feats_stat_cons_cato(cons, b_a, b_b, b_a_test):
        feats = []
        for agg_method in ['mean', 'std', 'min', 'max']:
            for feat in ['V_4', 'V_5', 'V_6', 'V_9', 'V_10', 'V_12', 'V_13']:
                _a = cons.groupby('ccx_id')[feat].agg(agg_method)
                feat = 'cons_{}_{}'.format(agg_method, feat)
                for b in [b_a, b_b, b_a_test]:
                    b[feat] = b['ccx_id'].map(_a)
                feats.append(feat)
        return feats

    def feats_trade_cons(cons,b_a, b_b, b_a_test):
        feats = []
        # 近1、3、6、12、24、36个月网购交易总额
        for m in [1, 3, 6, 12, 24, 36]:
            _c = cons[cons.month_from_now <= m]
            _a = _c.groupby('ccx_id')['V_6'].sum()
            feat = 'total_pay_last_{}_month'.format(m)
            for b in [b_a, b_b, b_a_test]:
                b[feat] = b['ccx_id'].map(_a).fillna(0)
            feats.append(feat)

        # 近1、3、6、12、24、36个月网购交易总笔数
        for m in [1, 3, 6, 12, 24, 36]:
            _c = cons[cons.month_from_now <= m]
            _a = _c.groupby('ccx_id')['V_6'].count()
            feat = 'total_pay_times_last_{}_month'.format(m)
            for b in [b_a, b_b, b_a_test]:
                b[feat] = b['ccx_id'].map(_a).fillna(0)
            feats.append(feat)

        # 近3、6、12、24、36个月月均交易额
        for m in [1, 3, 6, 12, 24, 36]:
            feat = 'mean_pay_last_{}_month'.format(m)
            for b in [b_a, b_b, b_a_test]:
                b[feat] = b['total_pay_last_{}_month'.format(m)]/m
            feats.append(feat)

        # 近3、6、12、24、36个月月均交易笔数
        for m in [1, 3, 6, 12, 24, 36]:
            feat = 'mean_pay_times_last_{}_month'.format(m)
            for b in [b_a, b_b, b_a_test]:
                b[feat] = b['total_pay_times_last_{}_month'.format(m)]/m
            feats.append(feat)

        # 近1个月交易笔数比上近3、6、12、24、36个月平均交易笔数
        for m in [3, 6, 12, 24, 36]:
            feat = 'mean_pay_times_last_1divide_{}_month'.format(m)
            for b in [b_a, b_b, b_a_test]:
                b[feat] = b['mean_pay_times_last_1_month'] / (0.001 + b['mean_pay_times_last_{}_month'.format(m)])
            feats.append(feat)

        # 近3个月交易笔数比上近6、12、24、36个月平均交易笔数
        for m in [6, 12, 24, 36]:
            feat = 'mean_pay_times_last_3divide_{}_month'.format(m)
            for b in [b_a, b_b, b_a_test]:
                b[feat] = b['mean_pay_times_last_3_month'] / (0.001 + b['mean_pay_times_last_{}_month'.format(m)])
            feats.append(feat)

        # 近1个月交易额比上近3、6、12、24、36个月平均交易额
        for m in [3, 6, 12, 24, 36]:
            feat = 'mean_pay_last_1divide_{}_month'.format(m)
            for b in [b_a, b_b, b_a_test]:
                b[feat] = b['mean_pay_last_1_month'] / (0.001 + b['mean_pay_last_{}_month'.format(m)])
            feats.append(feat)

        # 近3个月交易额比上近6、12、24、36个月平均交易额
        for m in [6, 12, 24, 36]:
            feat = 'mean_pay_last_3divide_{}_month'.format(m)
            for b in [b_a, b_b, b_a_test]:
                b['mean_pay_last_3divide_{}_month'.format(m)] = b['mean_pay_last_3_month'] / (0.001 + b['mean_pay_last_{}_month'.format(m)])
            feats.append(feat)

        for m in [1, 3, 6, 12, 24, 36]:
            feat = 'max_pay_last_{}_month'.format(m)
            _c = cons[cons.month_from_now <= m]
            _a = _c.groupby('ccx_id')['V_6'].max()
            for b in [b_a, b_b, b_a_test]:
                b[feat] = b['ccx_id'].map(_a).fillna(0)
            feats.append(feat)

        for m in [1, 3, 6, 12, 24, 36]:
            feat = 'min_pay_last_{}_month'.format(m)
            _c = cons[cons.month_from_now <= m]
            _a = _c.groupby('ccx_id')['V_6'].min()
            for b in [b_a, b_b, b_a_test]:
                b[feat] = b['ccx_id'].map(_a).fillna(0)
            feats.append(feat)
        return feats

    def feats_encoding_behavior_cato(b_a, b_b, b_a_test):
        feats = []
        for var in ['var3','var4','var5','var6','var11','var12','var13','var14','var15','var16','var17','var18']:
            feat = var + '_float'
            _a = b_a.groupby(var)[var].count() / (1.0 * len(b_a))
            for b in [b_a, b_b, b_a_test]:
                b[var] = b[var].fillna('na')
                b[feat] = b[var].map(_a).fillna(0.0)
            feats.append(feat)
        return feats

    def feats_encoding_mean_cons(cons,b_a,b_b,b_a_test):
        feats = []
        for var in ['V_1', 'V_2', 'V_3', 'V_8', 'V_14']:
            feat = 'trade_{}_float_mean'.format(var)
            _a = cons.groupby('ccx_id')['{}_float'.format(var)].mean()
            b_a[feat] = b_a['ccx_id'].map(_a).fillna(0.0)
            b_b[feat] = b_b['ccx_id'].map(_a).fillna(0.0)
            b_a_test[feat] = b_a_test['ccx_id'].map(_a).fillna(0.0)
            feats.append(feat)
        return feats

    def feats_encoding_mean_ccx(ccx, b_a, b_a_test):
        feats = []
        for var in ['var_02', 'var_03', 'var_04', 'var_05']:
            feat = 'query_{}_float_mean'.format(var)
            _a = ccx.groupby('ccx_id')['{}_float'.format(var)].mean()
            b_a[feat] = b_a['ccx_id'].map(_a).fillna(0.0)
            b_a_test[feat] = b_a_test['ccx_id'].map(_a).fillna(0.0)
            feats.append(feat)
        return feats

    def feats_counts_ccx(ccx, b_a, b_a_test):
        feats = [
            'query_var1_c2_count',
            'query_var1_c3_count',
            'query_var2_T1_count'
        ]
        for d, f in zip([
            ccx[ccx.var_01 == 'C2'],
            ccx[ccx.var_01 == 'C3'],
            ccx[ccx.var_02 == 'T1']
        ], feats):
            _a = d.groupby('ccx_id')['ccx_id'].count()
            b_a[f] = b_a['ccx_id'].map(_a).fillna(0)
            b_a_test[f] = b_a_test['ccx_id'].map(_a).fillna(0)
        return feats

    def feats_counts_cons(cons, b_a,b_b, b_a_test):
        feats = [
            'trade_var2_c60_count',
            'trade_var2_c31_count',
            'trade_var3_a4_count',
            'trade_var8_pl1_count',
        ]
        for d, f in zip([cons[cons.V_2 == 'C60'], cons[cons.V_2 == 'C31'], cons[cons.V_3 == 'a4'], cons[cons.V_3 == 'R0'], cons[cons.V_8 == 'PL1']], feats):
            _a = d.groupby('ccx_id')['ccx_id'].count()
            b_a[f] = b_a['ccx_id'].map(_a).fillna(0)
            b_b[f] = b_b['ccx_id'].map(_a).fillna(0)
            b_a_test[f] = b_a_test['ccx_id'].map(_a).fillna(0)
        return feats

        # time = 'day_from_now' or 'month_from_now'
    def feats_stat_ccx_time(ccx, b_a, b_a_test, time):
        feats = []
        for agg_method in ['mean', 'std', 'min', 'max']:
            _a = ccx.groupby('ccx_id')[time].agg(agg_method)
            feat = 'query_{}_{}'.format(agg_method, time)
            for b in [b_a, b_a_test]:
                b[feat] = b['ccx_id'].map(_a).fillna(9999)
            feats.append(feat)
        return feats

    def feats_query_cons(ccx, b_a, b_a_test):
        feats = []

        # 近1、3、6、12、24、36个月网购交易总笔数
        for m in [1, 3, 6, 12, 24, 36]:
            _c = ccx[ccx.month_from_now <= m]
            _a = _c.groupby('ccx_id')['var_01'].count()
            feat = 'total_query_times_last_{}_month'.format(m)
            for b in [b_a, b_a_test]:
                b[feat] = b['ccx_id'].map(_a).fillna(0)
            feats.append(feat)

        # 近3、6、12、24、36个月月均交易笔数
        for m in [1, 3, 6, 12, 24, 36]:
            feat = 'mean_query_times_last_{}_month'.format(m)
            for b in [b_a, b_a_test]:
                b[feat] = b['total_query_times_last_{}_month'.format(m)]/m
            feats.append(feat)

        # 近1个月交易笔数比上近3、6、12、24、36个月平均交易笔数
        for m in [3, 6, 12, 24, 36]:
            feat = 'mean_query_times_last_1divide_{}_month'.format(m)
            for b in [b_a, b_a_test]:
                b[feat] = b['mean_query_times_last_1_month'] / (0.001 + b['mean_query_times_last_{}_month'.format(m)])
            feats.append(feat)

        # 近3个月交易笔数比上近6、12、24、36个月平均交易笔数
        for m in [6, 12, 24, 36]:
            feat = 'mean_query_times_last_3divide_{}_month'.format(m)
            for b in [b_a, b_a_test]:
                b[feat] = b['mean_query_times_last_3_month'] / (0.001 + b['mean_query_times_last_{}_month'.format(m)])
            feats.append(feat)
        return feats

    def get_model_a(X_train, y, feats_all):
        train_data = lgb.Dataset(X_train[feats_all], label=y, free_raw_data=True)
        params = {
            'boosting_type': 'gbdt',
            'objective': 'binary',
            'metric': 'auc',
            'learning_rate': 0.05,
            'num_threads': 16,
            'verbose': -1,
            'num_leaves': 8,
        }
        num_round = 100
        return lgb.train(params, train_data, num_round)

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
    Y = pd.read_csv('./train/scene_A/train_target_A.csv')

    behavior = pd.merge(train_behavior_A, Y)
    y = behavior['target']
    b_a = behavior
    b_b = test_behavior_B
    b_a_test = test_behavior_A
    cons = pd.concat([train_consumer_A, test_consumer_A, test_consumer_B], axis=0, ignore_index=True)
    ccx = pd.concat([train_ccx_A, test_ccx_A], axis=0, ignore_index=True)
    preprocess_ccx_cons(ccx, cons)
    feats1 = list(feats_b_behavior(b_a, b_b, b_a_test))
    feats2 = feats_encoding_behavior_cato(b_a, b_b, b_a_test)
    feats3 = feats_stat_cons_time(cons, b_a, b_b, b_a_test, 'day_from_now')
    feats4 = feats_stat_cons_time(cons, b_a, b_b, b_a_test, 'month_from_now')
    feats5 = feats_stat_cons_cato(cons, b_a, b_b, b_a_test)
    feats6 = feats_trade_cons(cons,b_a, b_b, b_a_test)
    feats7 = feats_counts_cons(cons, b_a, b_a_test, b_b)
    feats14 = feats_encoding_mean_cons(cons,b_a,b_b,b_a_test)
    feats8= feats_encoding_mean_ccx(ccx, b_a, b_a_test)
    feats9= feats_counts_ccx(ccx, b_a, b_a_test)
    feats10= feats_stat_ccx_time(ccx, b_a, b_a_test, 'day_from_now')
    feats11= feats_stat_ccx_time(ccx, b_a, b_a_test, 'month_from_now')
    feats12 = feats_query_cons(ccx, b_a, b_a_test)
    feats_b = feats1 + feats2 + feats3+ feats4+ feats5 + feats6+ feats7 + ['ccx_id', 'var1', 'var2', 'var7', 'var8', 'var9']
    feats_a = feats_b + feats8 + feats9 + feats10 + feats11 + feats12 + feats14
    feats_all_b = list(set(feats_b) - set(['target']))
    feats_all_a = list(set(feats_a) - set(['target']))
    clf = get_model_a(b_a, y, feats_all_a)
    pred = clf.predict(b_a_test[feats_all_a].values)
    return pred


# Genrate Results
def Run(test_consumer_A,test_behavior_A,test_ccx_A,test_consumer_B,test_behavior_B,use_B=False):
    # if use_B == False:
    # without B data
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
    print("without B data %d features" % len(features))
    # lightgbm
    train_data = lgb.Dataset(PreProcess(info.iloc[:train_A_index][features],False),label=label)
    bst = lgb.train(params,train_data,num_boost_round=150)

    # catboost
    # train_data = info.iloc[:train_A_index]
    # categorical_features = np.where(train_data[features].dtypes != np.float)[0]
    #
    # model=CatBoostRegressor(iterations=150, depth=3, learning_rate=0.1, loss_function='RMSE')
    # model.fit(train_data[features].fillna(-1),train_data['target'],cat_features=categorical_features)


    yiming = bst.predict(PreProcess(info.iloc[train_A_index:train_A_index+test_A_index][features],False),num_iteration=bst.best_iteration)

    # predict_result_A['prob'] = model.predict(info.iloc[train_A_index:train_A_index+test_A_index][features].fillna(-1))

    # else:
    # add B data
    train_behavior_consume,test_behavior_consume_A = LabelB(train_consumer_A,train_behavior_A,train_behavior_B,train_consumer_B,Y,10,test_behavior_A,test_consumer_A)
    train_ccx = pd.concat([train_ccx_A,test_ccx_A])
    ccx = GetQueryFeatures(train_ccx,pd.concat([train_behavior_A.ccx_id,test_behavior_A.ccx_id]))
    train_ccx = ccx[ccx.ccx_id.isin(train_behavior_A.ccx_id)]
    test_ccx = ccx[ccx.ccx_id.isin(test_behavior_A.ccx_id)]

    train_data = pd.merge(train_behavior_consume,train_ccx,how='left')
    test_data = pd.merge(test_behavior_consume_A,test_ccx,how='left')

    label = train_data['target']
    features = [col for col in train_data.columns if col != 'target']
    print("with B data %d features" % len(features))
    # set the weight
    train_data['weight'] = 1
    train_data.loc[train_data.ccx_id.isin(train_behavior_B.ccx_id),'weight'] = 0.5
    weight = train_data['weight']

    train_data = lgb.Dataset(PreProcess(train_data[features],False),label=label,weight=weight)

    bst = lgb.train(params,train_data,num_boost_round=150)

    predict_result_A = pd.DataFrame(columns=['ccx_id','prob'])
    predict_result_A['ccx_id'] = test_data['ccx_id'].unique()

    yiming_withB = bst.predict(PreProcess(test_data[features],False),num_iteration=bst.best_iteration)

    # fang fei guo
    feiguo = run_ffg()

    predict_result_A = pd.DataFrame(columns=['ccx_id','prob'])
    predict_result_A['ccx_id'] = info.iloc[train_A_index:train_A_index+test_A_index]['ccx_id'].unique()
    predict_result_A['prob'] = 0.2 * yiming + 0.3 * yiming_withB + 0.5 * feiguo
    predict_result_A.to_csv('./predict_result_A.csv',encoding='utf-8',index=False)

    PredictB()

def PredictB():
    # read behaivor
    train_A_index = len(train_behavior_A)
    test_A_index = len(test_behavior_A)

    behavior_info = pd.concat([train_behavior_A,test_behavior_A,test_behavior_B])
    basic_info = GetBehavior(behavior_info)
    # read consuming
    consuming_info = pd.concat([train_consumer_A,test_consumer_A,test_consumer_B])
    consuming_info = GetConsuming(consuming_info)

    info = pd.merge(basic_info,consuming_info,how='left')
    info = pd.merge(info,Y,how="outer")
    features = [col for col in info.columns if col != 'target']

    # label = info.iloc[:train_A_index]['target']
    predict_result_B = pd.DataFrame(columns=['ccx_id','prob'])
    predict_result_B['ccx_id'] = info.iloc[train_A_index+test_A_index:]['ccx_id'].unique()

    # lightgbm
    # retrain
    # param = {'num_leaves':31, 'objective':'binary','metric':'auc','boosting_type': 'gbdt'}
    features_B = [col for col in features if col != 'ccx_id']
    print(len(features_B))
    # train_data = lgb.Dataset(PreProcess(info.iloc[:train_A_index][features_B],False),label=label)
    # bst = lgb.train(param,train_data,num_boost_round=50)


    # catboost
    # features_B = [col for col in features if col != 'ccx_id']
    train_data = info.iloc[:train_A_index]
    categorical_features = np.where(train_data[features_B].dtypes != np.float)[0]

    model=CatBoostRegressor(iterations=150, depth=3, learning_rate=0.1, loss_function='RMSE',random_seed=2018)
    model.fit(train_data[features_B].fillna(-1),train_data['target'],cat_features=categorical_features)

    predict_result_B['prob'] = model.predict(info.iloc[train_A_index+test_A_index:][features_B].fillna(-1))
    predict_result_B.to_csv('./predict_result_B.csv',encoding='utf-8',index=False)

# using extra train and test data
def ValidateByExtraData():
    auc = 0
    flag = True
    for i in range(1,6):
        # train_data,test_data = ReadExtraTrainTestData(str(i),True)
        train_data,test_data = ReadExtraTrainTestData(str(i),flag)
        features = [col for col in train_data.columns if col != 'target']

        # lightgbm
        # if use B, set the weight (A:1; B:0.6)
        if flag:
            train_data['weight'] = 1
            train_data.loc[train_data.ccx_id.isin(train_behavior_B.ccx_id),'weight'] = 0.5
            weight = train_data['weight']
            train_data = lgb.Dataset(PreProcess(train_data[features],False),label=train_data['target'],weight=weight)
        else:
            train_data = lgb.Dataset(PreProcess(train_data[features],False),label=train_data['target'])
        bst = lgb.train(params,train_data,num_boost_round=150)
        pred = bst.predict(PreProcess(test_data[features],False),num_iteration=bst.best_iteration)
        # print(pred)

        predict_result = pd.DataFrame(columns=['ccx_id','prob'])
        predict_result['ccx_id'] = test_data['ccx_id'].unique()
        predict_result['prob'] = pred
        predict_result.to_csv('./predict_result_%d.csv' % i,encoding='utf-8',index=False)

        # catboost
        # categorical_features = np.where(train_data[features].dtypes != np.float)[0]
        #
        # # categorical_features = np.where(train_data.dtypes != np.float)
        #
        # model=CatBoostRegressor(iterations=150, depth=3, learning_rate=0.1, loss_function='RMSE')
        # model.fit(train_data[features].fillna(-1),train_data['target'],cat_features=categorical_features)
        # pred = model.predict(test_data[features].fillna(-1))
        temp = roc_auc_score(test_data['target'],pred)
        print("DataSet %d : %.4f" % (i,temp))
        auc += temp
    return auc / 5

# read train_data
def ReadExtraTrainTestData(valid_number,use_B=False):
    path = "./train_test/train_test/train_test_"
    # read train A
    train_behavior_A = pd.read_csv(path+valid_number+"/scene_A/train_behavior_A.csv")
    train_consumer_A = pd.read_csv(path+valid_number+"/scene_A/train_consumer_A.csv")
    train_ccx_A = pd.read_csv(path+valid_number+"/scene_A/train_ccx_A.csv")

    # read test A
    test_behavior_A = pd.read_csv(path+valid_number+"/scene_A/test_behavior_A.csv")
    test_consumer_A = pd.read_csv(path+valid_number+"/scene_A/test_consumer_A.csv")
    test_ccx_A = pd.read_csv(path+valid_number+"/scene_A/test_ccx_A.csv")

    # read target A
    train_target = pd.read_csv(path+valid_number+"/scene_A/train_target_A.csv")
    test_target = pd.read_csv(path+valid_number+"/scene_A/test_target_A.csv")

    # read B data(has read)
    train_ccx = pd.concat([train_ccx_A,test_ccx_A])

    if use_B:
        # extract features
        # train_behavior_consume,test_behavior_consume_A,test_behavior_consume_B = LabelB(train_consumer_A,train_behavior_A,train_behavior_B,train_consumer_B,pd.concat([train_target,test_target]),10,test_behavior_A,test_consumer_A,test_behavior_B,test_consumer_B)
        train_behavior_consume,test_behavior_consume_A = LabelB(train_consumer_A,train_behavior_A,train_behavior_B,train_consumer_B,pd.concat([train_target,test_target]),10,test_behavior_A,test_consumer_A)
        ccx = GetQueryFeatures(train_ccx,pd.concat([train_behavior_A.ccx_id,test_behavior_A.ccx_id]))
        train_ccx = ccx[ccx.ccx_id.isin(train_behavior_A.ccx_id)]
        test_ccx = ccx[ccx.ccx_id.isin(test_behavior_A.ccx_id)]

        # generate train and test data
        train_data = pd.merge(train_behavior_consume,train_ccx,how='left')
        test_data = pd.merge(test_behavior_consume_A,test_ccx,how='left')


    else:
        train_index = len(train_behavior_A)

        behavior = pd.concat([train_behavior_A,test_behavior_A])
        behavior = GetBehavior(behavior)

        consumer = pd.concat([train_consumer_A,test_consumer_A])
        consumer = GetConsuming(consumer)

        ccx = pd.concat([train_ccx_A,test_ccx_A])
        ccx = GetQueryFeatures(ccx,behavior.ccx_id)

        info = pd.merge(behavior,consumer,how='left')
        info = pd.merge(info,ccx,how='left')

        train_data = info.iloc[:train_index]
        train_data = pd.merge(train_data,train_target,how='left')

        test_data = info.iloc[train_index:]
        test_data = pd.merge(test_data,test_target,how='left')
    return train_data,test_data


# Test(train_consumer_A,train_behavior_A,train_ccx_A,train_consumer_B,train_behavior_B)

'''generate predict_A and predict_B'''
Run(test_consumer_A,test_behavior_A,test_ccx_A,test_consumer_B,test_behavior_B)
# PredictB()
'''test 5 files'''
# res = ValidateByExtraData()
# print("the results of 5 cv is %.4f" % res)






#
# def testfm():
#     from pyfm import pylibfm
#     from sklearn.feature_extraction import DictVectorizer
#     import numpy as np
#     train = [
#         {"user": "1", "item": "5", "age": 19},
#         {"user": "2", "item": "43", "age": 33},
#         {"user": "3", "item": "20", "age": 55},
#         {"user": "4", "item": "10", "age": 20},
#     ]
#     v = DictVectorizer()
#     X = v.fit_transform(train)
#     y = np.repeat(1.0,X.shape[0])
#     fm = pylibfm.FM()
#     fm.fit(X,y)
#     fm.predict(v.transform({"user": "1", "item": "10", "age": 24}))
#
# testfm()

=======
# 比赛之后公开代码
>>>>>>> 7ed302c4cc7e70ee1cb541a6ea6a23fa7e81d76d
