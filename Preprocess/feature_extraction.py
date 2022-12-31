"""
生成特徵
"""
import numpy as np 
import pandas as pd


from scipy import stats

def label_fdict(info,column_name=['分行代碼','交易代碼'],date='交易日期',tag='眾數差_',s=0):
    """
    眾數差_:最近短時間類別變數眾數跟長時間眾數差異，如果沒變會是0
    新類數_:最近短時間類別數減去這之前的類別數，表示最近出現的新類別個數
    """
    try:
        info3=info[info[date]<=s]
        mode=stats.mode(info[column_name].fillna(-1))
        mode3=stats.mode(info3[column_name].fillna(-1))
        a1=dict(zip([tag+_ for _ in column_name],mode3.mode[0]-mode.mode[0]))
        
        info31=info[s<info[date]]
        a2={}
        for col in column_name:
            a2['新類數_'+col]=len(set(info3[col].fillna('').unique())-set(info31[col].fillna('').unique()))

        return a1|a2
    except:
        return {}
    

def label_fdict4(info,tag,column_name):
    """
    {天數}mode_{欄位}:眾數值
    {天數}count_{欄位}:眾數值的個數
    """
    mode=stats.mode(info[column_name])
    stats_info=pd.DataFrame({'mode':mode.mode[0],'count':mode.count[0]},index=column_name)
    stats_info=stats_info[stats_info['count']>1]
    
    df_out = stats_info.stack()
    df_out.index = df_out.index.map('{0[1]}_{0[0]}'.format).map(lambda x:tag+x)
    return df_out.to_dict()
    

def float_fdict(info,tag,column_name,f_list=['min','max','mean','std']):
    """
    {天數}{欄位}_的統計值 'min','max','mean','std'
    """
    stats_info=info[column_name].agg(f_list)
    df_out = stats_info.stack()
    df_out.index = df_out.index.map('{0[1]}_{0[0]}'.format).map(lambda x:tag+x)
    return df_out.to_dict()

    
def dp_type(x):
    """
    匯率是美元和其他
    """
    if 27<=x and x<=29:
        return 3
    if x==1:
        return 0
    if x<=4:
        return 2
    return 1


colname=pd.read_excel('../訓練資料欄位說明2.xlsx',sheet_name=[1,2,3,4,5,6,7])
datestr=pd.date_range('2021-04-01','2023-12-01').strftime('%Y-%m-%d')

ccbaa=pd.read_csv('../data1/ans/private_x_ccba_full_hashed.csv')
cdtxa=pd.read_csv('../data1/ans/private_x_cdtx0001_full_hashed.csv')
custinfoa=pd.read_csv('../data1/ans/private_x_custinfo_full_hashed.csv')
dpa=pd.read_csv('../data1/ans/private_x_dp_full_hashed.csv')
remita=pd.read_csv('../data1/ans/private_x_remit1_full_hashed.csv')
aalert_date=pd.read_csv('../data1/ans/private_x_alert_date.csv')

col_dict=colname[1].set_index('訓練資料欄位名稱')['訓練資料欄位中文說明'].to_dict()
ccbaa.columns=[col_dict.get(c) for c in ccbaa.columns]
col_dict=colname[2].set_index('訓練資料欄位名稱')['訓練資料欄位中文說明'].to_dict()
cdtxa.columns=[col_dict.get(c) for c in cdtxa.columns]
col_dict=colname[3].set_index('訓練資料欄位名稱')['訓練資料欄位中文說明'].to_dict()
custinfoa.columns=[col_dict.get(c) for c in custinfoa.columns]
col_dict=colname[4].set_index('訓練資料欄位名稱')['訓練資料欄位中文說明'].to_dict()
dpa.columns=[col_dict.get(c) for c in dpa.columns]
col_dict=colname[5].set_index('訓練資料欄位名稱')['訓練資料欄位中文說明'].to_dict()
remita.columns=[col_dict.get(c) for c in remita.columns]
col_dict=colname[6].set_index('訓練資料欄位名稱')['訓練資料欄位中文說明'].to_dict()
aalert_date.columns=[col_dict.get(c) for c in aalert_date.columns]

ydataa=pd.merge(aalert_date,custinfoa,on='alert主鍵')

dpa['借貸別']=dpa['借貸別'].apply(lambda x: {'CR':0,'DB':1}.get(x,x) )
dpa['交易金額*匯率']=dpa['交易金額']*dpa['匯率']



ccba=pd.read_csv('../data1/public_train_x_ccba_full_hashed.csv')
cdtx=pd.read_csv('../data1/public_train_x_cdtx0001_full_hashed.csv')
custinfo=pd.read_csv('../data1/public_train_x_custinfo_full_hashed.csv')
dp=pd.read_csv('../data1/public_train_x_dp_full_hashed.csv')
remit=pd.read_csv('../data1/public_train_x_remit1_full_hashed.csv')

palert_date=pd.read_csv('../data1/public_x_alert_date.csv')
talert_date=pd.read_csv('../data1/train_x_alert_date.csv')
alert_date=talert_date.append(palert_date)
alert_date=alert_date.reset_index(drop=True)

y=pd.read_csv('../data1/train_y_answer.csv')
true=pd.read_csv('../data1/24_ESun_public_y_answer.csv')
y=y.append(true)

col_dict=colname[1].set_index('訓練資料欄位名稱')['訓練資料欄位中文說明'].to_dict()
ccba.columns=[col_dict.get(c) for c in ccba.columns]
col_dict=colname[2].set_index('訓練資料欄位名稱')['訓練資料欄位中文說明'].to_dict()
cdtx.columns=[col_dict.get(c) for c in cdtx.columns]
col_dict=colname[3].set_index('訓練資料欄位名稱')['訓練資料欄位中文說明'].to_dict()
custinfo.columns=[col_dict.get(c) for c in custinfo.columns]
col_dict=colname[4].set_index('訓練資料欄位名稱')['訓練資料欄位中文說明'].to_dict()
dp.columns=[col_dict.get(c) for c in dp.columns]
col_dict=colname[5].set_index('訓練資料欄位名稱')['訓練資料欄位中文說明'].to_dict()
remit.columns=[col_dict.get(c) for c in remit.columns]
col_dict=colname[6].set_index('訓練資料欄位名稱')['訓練資料欄位中文說明'].to_dict()
alert_date.columns=[col_dict.get(c) for c in alert_date.columns]
col_dict=colname[7].set_index('訓練資料欄位名稱')['訓練資料欄位中文說明'].to_dict()
y.columns=[col_dict.get(c) for c in y.columns]

ydata=pd.merge(y,alert_date,on='alert主鍵',how='right')
ydata=pd.merge(ydata,custinfo,on='alert主鍵')

ccba=ccba.sort_values('帳務年月').reset_index(drop=True)
cdtx=cdtx.sort_values('消費日期').reset_index(drop=True)
dp=dp.sort_values('交易日期').reset_index(drop=True)
remit=remit.sort_values('外匯交易日(帳務日)').reset_index(drop=True)

dp['借貸別']=dp['借貸別'].apply(lambda x: {'CR':0,'DB':1}.get(x,x) )
dp['交易金額*匯率']=dp['交易金額']*dp['匯率']

dp=dp.sort_values('交易時間',ascending=False).sort_values('交易日期').reset_index(drop=True)


custinfo['職業']=custinfo['職業'].fillna(-1)
custinfoa['職業']=custinfoa['職業'].fillna(-1)

dp['交易代碼']=dp['交易代碼'].fillna(-1)
dpa['交易代碼']=dpa['交易代碼'].fillna(-1)

dp['分行代碼']=dp['分行代碼'].fillna(-1)
dpa['分行代碼']=dpa['分行代碼'].fillna(-1)


ydata=ydata.append(ydataa).reset_index(drop=True)

ccba=ccba.append(ccbaa).sort_values('帳務年月').reset_index(drop=True)
cdtx=cdtx.append(cdtxa).sort_values('消費日期').reset_index(drop=True)
dp=dp.append(dpa).sort_values('交易日期').reset_index(drop=True)
remit=remit.append(remita).sort_values('外匯交易日(帳務日)').reset_index(drop=True)


# 轉日期
ydata['alert日期']=ydata['alert主鍵發生日期'].apply(lambda x:datestr[x])
ydata['ym']=ydata['alert日期'].apply(lambda x:x[:7])
ydata['max_date']=ydata.groupby(['顧客編號','ym'])['alert主鍵發生日期'].transform(max)
ydata['train_id']=ydata.apply(lambda x:1 if x['max_date']==x['alert主鍵發生日期'] else 0 ,axis=1)

ydata['date']=pd.to_datetime(ydata['alert日期'])
ydata['day_in_month']=ydata['date'].apply(lambda x:x.day)
ydata['day_of_week']=ydata['date'].apply(lambda x:x.day_of_week)


holiday=pd.read_csv('110中華民國政府行政機關辦公日曆表.csv',encoding='big5')
holiday2=pd.read_csv('111年中華民國政府行政機關辦公日曆表.csv',encoding='big5')
holiday=holiday.append(holiday2)
holiday['是否放假']=holiday['是否放假'].replace(2,1)
holiday['alert日期']=pd.to_datetime(holiday['西元日期'],format='%Y%m%d')


holidays=[]
tmp=0
for k,v in holiday[['alert日期','是否放假']].iterrows():
    tmp+=v.是否放假
    if v.是否放假==0:
        holidays.append([ v.alert日期,tmp ])
        tmp=0
holidays=pd.DataFrame(holidays,columns=['alert日期','放假幾天'])
holidays['alert日期']=holidays['alert日期'].apply(lambda x:x.strftime('%Y-%m-%d'))


#有同天同個人多個alert會有一樣特徵留一個
ydata=ydata.sort_values('alert主鍵報SAR與否').drop_duplicates(['顧客編號','alert主鍵發生日期'],keep='last')


ydata=pd.merge(ydata,holidays,on='alert日期')
ydata=ydata.sort_values('alert主鍵發生日期').reset_index(drop=True)

# 美金
dp['匯率']=dp['匯率'].apply(np.round)
dp['匯率type']=dp['匯率'].apply(dp_type)



# ================長時間7天，短時間當天，如果有放假，放假發生的事情也當作當天==========


pd.options.mode.chained_assignment = None
count=0
alldata=[]

tag='7'

wday=7

for k,r in ydata[ydata['alert主鍵報SAR與否'].isnull()].iterrows():
    base=r.to_dict()
    
    fday=r['放假幾天']
    
    info=ydata[(ydata['顧客編號']==r.顧客編號)&(ydata['alert主鍵發生日期']<=r.alert主鍵發生日期)]
    info['alert主鍵發生日期']=r.alert主鍵發生日期-info['alert主鍵發生日期']
    base = base | float_fdict(info,tag,['alert主鍵發生日期','行內總資產','風險等級'])
    if len(info)>1:
        base = base | {'前次alert天數':info.iloc[-2]['alert主鍵發生日期']}
    
    
    info=ccba[(ccba['顧客編號']==r.顧客編號)&(ccba['帳務年月']<=r.alert主鍵發生日期)]
    info['帳務年月']=r.alert主鍵發生日期-info['帳務年月']
    if len(info)>0:
        base = base | info.iloc[-1].to_dict()
    

    info=cdtx[(cdtx['顧客編號']==r.顧客編號)&(cdtx['消費日期']<=r.alert主鍵發生日期)]
    info['消費日期']=r.alert主鍵發生日期-info['消費日期']
    infow=info[info['消費日期']<wday]
    
    if len(infow)>0:
        base = base | label_fdict(infow,['消費地國別', '消費地幣別'],date='消費日期',s=fday)
        base = base | float_fdict(infow,tag,['消費日期','交易金額-台幣'])
        base = base | label_fdict4(infow,tag,['消費地國別', '消費地幣別'])
        
    info=info[info['消費日期']<=fday]
    if len(info)>0:
        base = base | float_fdict(info,'0',['消費日期','交易金額-台幣'])
        base = base | label_fdict4(info,'0',['消費地國別', '消費地幣別','交易金額-台幣'])
        base = base | float_fdict(info,'0',['消費地國別', '消費地幣別'])
    
    info=info[info['消費日期']<wday]
    if len(info)>0:    
        base = base | {'信用卡刷卡_len':len(info)}
        base = base | {'信用卡_消費地幣別數':info['消費地幣別'].unique().size }
        base = base | {'信用卡_消費地國別':info['消費地國別'].unique().size }
        base = base | info[info['交易金額-台幣']==info['交易金額-台幣'].max()].iloc[0].to_dict()

    
    
    info=dp[(dp['顧客編號']==r.顧客編號)&(dp['交易日期']<=r.alert主鍵發生日期)]
    info['交易日期']=r.alert主鍵發生日期-info['交易日期']
    infow=info[info['交易日期']<wday]
    
    if len(infow)>0:
        base = base | label_fdict(infow,['借貸別','交易時間','交易類別','匯率type','資訊資產代號','交易代碼','分行代碼','是否為跨行交易','是否為實體ATM交易'],s=fday)
        base = base | float_fdict(infow,tag,['交易日期','交易金額'])
        base = base | label_fdict4(infow,tag,['借貸別','交易時間','交易類別','匯率type','資訊資產代號','交易代碼','分行代碼','是否為跨行交易','是否為實體ATM交易'])
        

    info=info[info['交易日期']<=fday]
    if len(info)>0:        
        base = base | float_fdict(info,'0',['交易日期','交易金額'])
        base = base | label_fdict4(info,'0',['交易金額','借貸別','交易時間','交易類別','匯率type','資訊資產代號','交易代碼','分行代碼','是否為跨行交易','是否為實體ATM交易'])
        base = base | float_fdict(info,'0',['借貸別','交易時間','交易類別','匯率type','資訊資產代號','交易代碼','分行代碼','是否為跨行交易','是否為實體ATM交易'])
        
        base = base | {'資訊資產代號20': len(info[info['資訊資產代號']==20])   }
        base = base | {'交易代碼20': len(info[info['交易代碼']==20]) }
        base = base | {'資訊資產代號12_借貸別1': len(info[(info['資訊資產代號']==12)&(info['借貸別']==1)])   }

        
    info=info[info['交易日期']<wday]
    if len(info)>0:    
        base = base | {'借貸_len':len(info)}
        max_v=info[info['交易金額*匯率']==info['交易金額*匯率'].max()]
        if len(max_v)>0:
            base = base | info[info['交易金額*匯率']==info['交易金額*匯率'].max()].iloc[0].to_dict()

    
    info=remit[(remit['顧客編號']==r.顧客編號)&(remit['外匯交易日(帳務日)']<=r.alert主鍵發生日期)]
    info['外匯交易日(帳務日)']=r.alert主鍵發生日期-info['外匯交易日(帳務日)']
    info=info[info['外匯交易日(帳務日)']<wday]
    
    if len(info)>0:
        base = base | label_fdict(info,['交易編號'],date='外匯交易日(帳務日)',s=fday)
        base = base | label_fdict4(info,tag,['外匯交易日(帳務日)','交易編號'])
        base = base | float_fdict(info,tag,['外匯交易日(帳務日)','交易金額(折合美金)'])

    info=info[info['外匯交易日(帳務日)']<=fday]
    if len(info)>0:
        base = base | {'外匯_len':len(info)}
        base = base | info[info['交易金額(折合美金)']==info['交易金額(折合美金)'].max()].iloc[0].to_dict()        


    alldata.append(base)

alldata=pd.DataFrame(alldata)
alldata.to_csv('test.csv',index=False)




count=0
alldata=[]

tag='7'
wday=7

for k,r in ydata[~ydata['alert主鍵報SAR與否'].isnull()].iterrows():
    base=r.to_dict()
    
    fday=r['放假幾天']
    
    info=ydata[(ydata['顧客編號']==r.顧客編號)&(ydata['alert主鍵發生日期']<=r.alert主鍵發生日期)]
    info['alert主鍵發生日期']=r.alert主鍵發生日期-info['alert主鍵發生日期']
    base = base | float_fdict(info,tag,['alert主鍵發生日期','行內總資產','風險等級'])
    if len(info)>1:
        base = base | {'前次alert天數':info.iloc[-2]['alert主鍵發生日期']}
    
    
    info=ccba[(ccba['顧客編號']==r.顧客編號)&(ccba['帳務年月']<=r.alert主鍵發生日期)]
    info['帳務年月']=r.alert主鍵發生日期-info['帳務年月']
    if len(info)>0:
        base = base | info.iloc[-1].to_dict()
    

    info=cdtx[(cdtx['顧客編號']==r.顧客編號)&(cdtx['消費日期']<=r.alert主鍵發生日期)]
    info['消費日期']=r.alert主鍵發生日期-info['消費日期']
    infow=info[info['消費日期']<wday]
    
    if len(infow)>0:
        base = base | label_fdict(infow,['消費地國別', '消費地幣別'],date='消費日期',s=fday)
        base = base | float_fdict(infow,tag,['消費日期','交易金額-台幣'])
        base = base | label_fdict4(infow,tag,['消費地國別', '消費地幣別'])
        
    info=info[info['消費日期']<=fday]
    if len(info)>0:
        base = base | float_fdict(info,'0',['消費日期','交易金額-台幣'])
        base = base | label_fdict4(info,'0',['消費地國別', '消費地幣別','交易金額-台幣'])
        base = base | float_fdict(info,'0',['消費地國別', '消費地幣別'])
    
    info=info[info['消費日期']<wday]
    if len(info)>0:    
        base = base | {'信用卡刷卡_len':len(info)}
        base = base | {'信用卡_消費地幣別數':info['消費地幣別'].unique().size }
        base = base | {'信用卡_消費地國別':info['消費地國別'].unique().size }
        base = base | info[info['交易金額-台幣']==info['交易金額-台幣'].max()].iloc[0].to_dict()

    
    
    info=dp[(dp['顧客編號']==r.顧客編號)&(dp['交易日期']<=r.alert主鍵發生日期)]
    info['交易日期']=r.alert主鍵發生日期-info['交易日期']
    infow=info[info['交易日期']<wday]
    
    if len(infow)>0:
        base = base | label_fdict(infow,['借貸別','交易時間','交易類別','匯率type','資訊資產代號','交易代碼','分行代碼','是否為跨行交易','是否為實體ATM交易'],s=fday)
        base = base | float_fdict(infow,tag,['交易日期','交易金額'])
        base = base | label_fdict4(infow,tag,['借貸別','交易時間','交易類別','匯率type','資訊資產代號','交易代碼','分行代碼','是否為跨行交易','是否為實體ATM交易'])
        

    info=info[info['交易日期']<=fday]
    if len(info)>0:        
        base = base | float_fdict(info,'0',['交易日期','交易金額'])
        base = base | label_fdict4(info,'0',['交易金額','借貸別','交易時間','交易類別','匯率type','資訊資產代號','交易代碼','分行代碼','是否為跨行交易','是否為實體ATM交易'])
        base = base | float_fdict(info,'0',['借貸別','交易時間','交易類別','匯率type','資訊資產代號','交易代碼','分行代碼','是否為跨行交易','是否為實體ATM交易'])
        
        base = base | {'資訊資產代號20': len(info[info['資訊資產代號']==20])   }
        base = base | {'交易代碼20': len(info[info['交易代碼']==20]) }
        base = base | {'資訊資產代號12_借貸別1': len(info[(info['資訊資產代號']==12)&(info['借貸別']==1)])   }

        
    info=info[info['交易日期']<wday]
    if len(info)>0:    
        base = base | {'借貸_len':len(info)}
        max_v=info[info['交易金額*匯率']==info['交易金額*匯率'].max()]
        if len(max_v)>0:
            base = base | info[info['交易金額*匯率']==info['交易金額*匯率'].max()].iloc[0].to_dict()

    
    info=remit[(remit['顧客編號']==r.顧客編號)&(remit['外匯交易日(帳務日)']<=r.alert主鍵發生日期)]
    info['外匯交易日(帳務日)']=r.alert主鍵發生日期-info['外匯交易日(帳務日)']
    info=info[info['外匯交易日(帳務日)']<wday]
    
    if len(info)>0:
        base = base | label_fdict(info,['交易編號'],date='外匯交易日(帳務日)',s=fday)
        base = base | label_fdict4(info,tag,['外匯交易日(帳務日)','交易編號'])
        base = base | float_fdict(info,tag,['外匯交易日(帳務日)','交易金額(折合美金)'])

    info=info[info['外匯交易日(帳務日)']<=fday]
    if len(info)>0:
        base = base | {'外匯_len':len(info)}
        base = base | info[info['交易金額(折合美金)']==info['交易金額(折合美金)'].max()].iloc[0].to_dict()        

    alldata.append(base)
    
    count+=1
    if count%2000==0:
        print(count)

alldata=pd.DataFrame(alldata)
alldata.to_csv('train.csv',index=False)
