"""
執行訓練跟預測，儲存成上傳csv檔
input:
'預測的案件名單及提交檔案範例.csv'
'../data1/24_ESun_public_y_answer.csv'
'../Preprocess/test.csv'
'../Preprocess/train.csv'

output:
'test_end.csv'
"""

import pandas as pd
import lightgbm as lgb

f_tree=['風險等級',
 'train_id',
 '放假幾天',
 '7alert主鍵發生日期_max',
 '7風險等級_max',
 '眾數差_交易類別',
 '眾數差_匯率type',
 '眾數差_資訊資產代號',
 '眾數差_是否為跨行交易',
 '眾數差_是否為實體ATM交易',
 '新類數_是否為實體ATM交易',
 '7count_分行代碼',
 '7mode_是否為實體ATM交易',
 '借貸_len',
 '7交易金額_max',
 '交易日期',
 '是否為跨行交易',
 '是否為實體ATM交易',
 '眾數差_交易編號',
 '外匯_len',
 '眾數差_消費地幣別',
 '新類數_消費地幣別',
 '7count_消費地幣別',
 '信用卡刷卡_len',
 '信用卡_消費地幣別數',
 '信用卡_消費地國別',
 '消費地國別',
 '消費地幣別',
 '7mode_外匯交易日(帳務日)',
 '0交易日期_max',
 '0mode_交易類別',
 '0count_是否為實體ATM交易',
 '0借貸別_min',
 '0交易類別_min',
 '0是否為跨行交易_min',
 '0交易代碼_max',
 '0分行代碼_max',
 '0是否為跨行交易_max',
 '0交易類別_std',
 '0交易金額_max',
 '0交易金額_mean',
 '0消費地國別_min',
 '0消費地幣別_max',
 '0消費地國別_mean',
 '0消費地幣別_mean',
 '0mode_消費地國別']

best_params={'num_leaves': 52,
 'min_child_samples': 14,
 'max_depth': 10,
 'max_bin': 405,
 'learning_rate': 0.04538913114636621,
 'n_estimators': 129,
 'subsample': 0.34883458046535465,
 'colsample_bytree': 0.2854988995776602,
 'reg_alpha': 0.33165709681418964,
 'reg_lambda': 2.9666708622433826}



f_tree2=['train_id',
 '放假幾天',
 '7風險等級_min',
 '7alert主鍵發生日期_max',
 '眾數差_是否為跨行交易',
 '7交易日期_min',
 '7交易日期_std',
 '7mode_交易類別',
 '7count_分行代碼',
 '7mode_是否為跨行交易',
 '7mode_是否為實體ATM交易',
 '借貸_len',
 '7交易金額_max',
 '交易類別',
 '外匯_len',
 '眾數差_消費地幣別',
 '7count_消費地國別',
 '7count_消費地幣別',
 '信用卡刷卡_len',
 '消費地國別',
 '0交易日期_min',
 '0借貸別_min',
 '0交易類別_min',
 '0交易代碼_min',
 '0是否為跨行交易_min',
 '0是否為實體ATM交易_min',
 '0借貸別_max',
 '0分行代碼_max',
 '0是否為實體ATM交易_max',
 '0借貸別_mean',
 '0交易類別_std',
 '0資訊資產代號_std',
 '0交易代碼_std',
 '0交易金額-台幣_mean',
 '0消費地國別_min']

best_params2={'num_leaves': 47,
 'min_child_samples': 12,
 'max_depth': 14,
 'max_bin': 370,
 'learning_rate': 0.06864168413182435,
 'n_estimators': 228,
 'subsample': 0.002616515203251013,
 'colsample_bytree': 0.26174102127416643,
 'reg_alpha': 0.5206800583365043,
 'reg_lambda': 4.003145580357057}


testt=pd.read_csv('預測的案件名單及提交檔案範例.csv')
test=pd.read_csv('../Preprocess/test.csv')
train=pd.read_csv('../Preprocess/train.csv')
data=train[train['ym']>='2021-05'].reset_index(drop=True)

clf1 = lgb.LGBMClassifier(**best_params)
clf1.fit(data[f_tree] , data['alert主鍵報SAR與否'] )

clf2 = lgb.LGBMClassifier(**best_params2)
clf2.fit(data[f_tree2], data['alert主鍵報SAR與否'] )

y_prob= clf1.predict_proba(test[f_tree])
y_prob2= clf2.predict_proba(test[f_tree2])

test['probability']=(y_prob[:,1]+y_prob2[:,1])/2

test=test.rename(columns={'alert主鍵':'alert_key'})

output=test[['alert_key','probability']]

test=pd.merge(testt[['alert_key']],output,on='alert_key',how='left')


true=pd.read_csv('../data1/24_ESun_public_y_answer.csv')
true=true.rename(columns={'sar_flag':'probability'})
testo=pd.merge(test,true,on='alert_key',how='left')
testo['probability']=testo['probability_x'].fillna(0)+testo['probability_y'].fillna(0)
testo[['alert_key','probability']].to_csv('test_end.csv',index=False)



