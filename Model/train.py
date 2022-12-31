"""
用不同的random_state不一樣的交叉驗證集，保留前幾個特徵再選取新特徵

input:
'../Preprocess/test.csv'
'../Preprocess/train.csv'

output:
'model1_new.joblib'
'model2_new.joblib'
"""

import numpy as np 
import pandas as pd
import lightgbm as lgb

# SequentialFeatureSelector選取最差10筆的平均挑選特徵
from sffs import SequentialFeatureSelector as SFFS

from sklearn.metrics import make_scorer
from sklearn.model_selection import StratifiedShuffleSplit
import optuna
from joblib import dump
pd.set_option('mode.chained_assignment', None)


def n_1recall(y_test, y_pred,num=1):
    """
    本次比賽將以 Recall@N-1的Precision 進行評分，意即在抓到N-1個真正報SAR案件名單下的Precision（將四捨五入計算至小數點後7位）
    """
    pred1=y_pred[np.where(y_test == 1)]
    pred1.sort()
    return (y_test.sum()-num)/((y_pred>=pred1[num]).sum())

n_1_score=make_scorer(n_1recall,greater_is_better=True,needs_proba=True)


# 比賽驗證每個月，到4月
def date_to_split(ydata,column='alert日期',start='2021-04',end='2022-05',folds=8, filter_id=None):
    """
    使用一個月做驗證，這個月之前做訓練，去掉只有一筆真正SAR的2022/02
    """
    
    fold_index=0
    tscvlist=[]
    for date in pd.date_range(start ,pd.to_datetime(end) ,freq='M').strftime('%Y-%m-%d')[::-1]:
        date_start = date[:-2]+'01'
        # 只有一筆alert跳過
        if date_start=='2022-02-01':
            continue
        if filter_id:
            train=ydata[(ydata[column]<date_start)&(ydata[filter_id]==1)]
        else:
            train=ydata[(ydata[column]<date_start)]
        test=ydata[(ydata[column]<=date)&(date_start<=ydata[column])]
        tscvlist.append( (train.index.values, test.index.values) )

        fold_index+=1
        if fold_index==folds:
            return tscvlist


test=pd.read_csv('../Preprocess/test.csv')
train=pd.read_csv('../Preprocess/train.csv')


data=train[train['ym']>='2021-05'].reset_index(drop=True)

sfeature=test.drop(columns=['alert主鍵', 'alert主鍵報SAR與否', 'alert主鍵發生日期', '顧客編號' ,'alert日期', 'ym', 'max_date','date','day_in_month','外匯交易日(帳務日)',
 '7alert主鍵發生日期_min',
 '本月預借現金金額']).columns


sss = StratifiedShuffleSplit(n_splits=30, test_size=0.1, random_state=0)

fixed_features=[
'train_id',
'信用卡刷卡_len',
'借貸_len',
'外匯_len',
'放假幾天',
'7alert主鍵發生日期_max'
]


clf = lgb.LGBMClassifier()

sfs = SFFS(clf, 
           k_features=50, 
           forward=True,
           floating=True, 
           scoring=n_1_score,
           cv=list(sss.split(data, data['alert主鍵報SAR與否'])),
           n_jobs=-1,
           fixed_features=fixed_features,
            verbose=2,
          )

sfs = sfs.fit(data[sfeature], data['alert主鍵報SAR與否'])

print('CV Score:')
print(sfs.k_score_)
print(sfs.k_feature_names_)

sfsd=pd.DataFrame(sfs.get_metric_dict()).T
sfsd['scores']=sfsd['cv_scores'].apply(lambda x:x.mean())
f_tree=list(sfsd[sfsd['scores']==sfsd['scores'].max()]['feature_names'].values[0])



param_distributions = {
    "num_leaves": optuna.distributions.IntDistribution( 40, 60),
    "min_child_samples" : optuna.distributions.IntDistribution( 10 , 30 ),
    "max_depth": optuna.distributions.IntDistribution( 5, 20 ),
    
    # 為了準
    "max_bin" : optuna.distributions.IntDistribution( 300, 500 ),
    "learning_rate": optuna.distributions.FloatDistribution(0.01, 0.1),
    "n_estimators": optuna.distributions.IntDistribution(50, 250),
    
    # 為了不overfitting
    "subsample": optuna.distributions.FloatDistribution(0,1),
    "colsample_bytree": optuna.distributions.FloatDistribution(0,1),
    "reg_alpha": optuna.distributions.FloatDistribution(1e-9, 1.0),
    "reg_lambda": optuna.distributions.FloatDistribution(1e-9, 5.0),
}

ss2 = StratifiedShuffleSplit(n_splits=30, test_size=0.1, random_state=10)

clf = lgb.LGBMClassifier(verbose=-1)
optuna.logging.set_verbosity(0)
optuna_search = optuna.integration.OptunaSearchCV(
clf, param_distributions, n_trials=None, timeout=20*60,cv=list(ss2.split(data, data['alert主鍵報SAR與否'])) ,scoring=n_1_score,verbose=-1
)

optuna_search.fit(data[f_tree], data['alert主鍵報SAR與否'])

print(optuna_search.best_score_)
best_params=optuna_search.best_params_
print(best_params)


"""
第2個模型
用不同的random_state不一樣的交叉驗證集，保留前幾個快速增加的特徵再選取新特徵
"""

sss = StratifiedShuffleSplit(n_splits=30, test_size=0.1, random_state=1)

fixed_features=['train_id',
 '放假幾天',
 '7alert主鍵發生日期_max',
 '7count_分行代碼',
 '7mode_是否為實體ATM交易',
 '借貸_len',
 '7交易金額_max',
 '外匯_len',
 '7count_消費地幣別',
 '信用卡刷卡_len',
 '0分行代碼_max',
 '0交易類別_std']


clf = lgb.LGBMClassifier()

sfs = SFFS(clf, 
           k_features=50, 
           forward=True,
           floating=True, 
           scoring=n_1_score,
           cv=list(sss.split(data, data['alert主鍵報SAR與否'])),
           n_jobs=-1,
           fixed_features=fixed_features,
            verbose=2,
          )

sfs = sfs.fit(data[sfeature], data['alert主鍵報SAR與否'])

print('CV Score:')
print(sfs.k_score_)
print(sfs.k_feature_names_)


sfsd=pd.DataFrame(sfs.get_metric_dict()).T
sfsd['scores']=sfsd['cv_scores'].apply(lambda x:x.mean())
f_tree2=list(sfsd[sfsd['scores']==sfsd['scores'].max()]['feature_names'].values[0])


ss2 = StratifiedShuffleSplit(n_splits=30, test_size=0.1, random_state=100)

clf = lgb.LGBMClassifier(verbose=-1)
optuna.logging.set_verbosity(0)
optuna_search = optuna.integration.OptunaSearchCV(
clf, param_distributions, n_trials=None, timeout=20*60,cv=list(ss2.split(data, data['alert主鍵報SAR與否'])) ,scoring=n_1_score,verbose=-1
)

optuna_search.fit(data[f_tree2], data['alert主鍵報SAR與否'])

print(optuna_search.best_score_)
best_params2=optuna_search.best_params_
print(best_params2)



print('模型1分數')
score_list=[]

for fold, (train_idx, test_idx) in enumerate(date_to_split(data,folds=10)):
    X_train, X_test = data.loc[train_idx], data.loc[test_idx]
    y_train, y_test = data['alert主鍵報SAR與否'].loc[train_idx], data['alert主鍵報SAR與否'].loc[test_idx]
    
    clf = lgb.LGBMClassifier(**best_params)
    
    clf.fit(X_train[f_tree], y_train)

    y_pred=clf.predict_proba(X_test[f_tree])[:,1]
    print('===================================================================')
    ym=X_test['ym'].iloc[0]
    print(f'{ym} 分數 {n_1recall(y_test,y_pred)}')    
    
    score_list.append(n_1recall(y_test,y_pred))
    
print(f'平均分 {np.mean(score_list)}')

print('**************************')

print('模型2分數')
score_list=[]

for fold, (train_idx, test_idx) in enumerate(date_to_split(data,folds=10)):
    X_train, X_test = data.loc[train_idx], data.loc[test_idx]
    y_train, y_test = data['alert主鍵報SAR與否'].loc[train_idx], data['alert主鍵報SAR與否'].loc[test_idx]
    
    clf = lgb.LGBMClassifier(**best_params2)
    
    clf.fit(X_train[f_tree2], y_train)

    y_pred=clf.predict_proba(X_test[f_tree2])[:,1]
    print('===================================================================')
    ym=X_test['ym'].iloc[0]
    print(f'{ym} 分數 {n_1recall(y_test,y_pred)}')    
    
    score_list.append(n_1recall(y_test,y_pred))
    
print(f'平均分 {np.mean(score_list)}')


print('**************************')
print('模型1+2分數')
score_list=[]

for fold, (train_idx, test_idx) in enumerate(date_to_split(data,folds=10)):
    X_train, X_test = data.loc[train_idx], data.loc[test_idx]
    y_train, y_test = data['alert主鍵報SAR與否'].loc[train_idx], data['alert主鍵報SAR與否'].loc[test_idx]
    
    clf = lgb.LGBMClassifier(**best_params)
    clf.fit(X_train[f_tree], y_train)
    
    clf2 = lgb.LGBMClassifier(**best_params2)
    clf2.fit(X_train[f_tree2], y_train)
    

    y_pred=clf.predict_proba(X_test[f_tree])[:,1]
    y_pred2=clf2.predict_proba(X_test[f_tree2])[:,1]
    
    y_pred=(y_pred+y_pred2)/2
    
    print('===================================================================')
    ym=X_test['ym'].iloc[0]
    print(f'{ym} 分數 {n_1recall(y_test,y_pred)}')
    
    score_list.append(n_1recall(y_test,y_pred))
print(f'平均分 {np.mean(score_list)}')


clf1 = lgb.LGBMClassifier(**best_params)
clf1.fit(data[f_tree] , data['alert主鍵報SAR與否'] )
clf2 = lgb.LGBMClassifier(**best_params2)
clf2.fit(data[f_tree2], data['alert主鍵報SAR與否'] )

dump(clf1,'model1_new.joblib')
dump(clf2,'model2_new.joblib')

