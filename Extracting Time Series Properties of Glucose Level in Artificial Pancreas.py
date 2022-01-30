import numpy as np
import pandas as pd
from datetime import datetime as dt



CGM = pd.read_csv('CGMData.csv')
Insulin = pd.read_csv('InsulinData.csv') 

CGM_Data = pd.DataFrame(CGM, columns=['Date','Time','Sensor Glucose (mg/dL)'])

Insulin_Data = pd.DataFrame(Insulin, columns=['Date','Time','Alarm'])



CGM_Data['Date']=pd.to_datetime(CGM_Data['Date']).dt.date

Insulin_Data['Date']=pd.to_datetime(Insulin_Data['Date']).dt.date



CGM_Data['Time']=pd.to_datetime(CGM_Data['Time']).dt.time

Insulin_Data['Time']=pd.to_datetime(Insulin_Data['Time']).dt.time



CGM_Data['Sensor Glucose (mg/dL)'] = pd.to_numeric(CGM_Data['Sensor Glucose (mg/dL)'])



Manual_Automode=Insulin_Data.loc[Insulin_Data['Alarm']=='AUTO MODE ACTIVE PLGM OFF']



startdate_auto=Manual_Automode.iloc[1,0] 

starttime_auto=Manual_Automode.iloc[1,1] 




cgm_M=CGM_Data.loc[(CGM_Data['Date']<startdate_auto) | ((CGM_Data['Date']==startdate_auto) & (CGM_Data['Time']<starttime_auto) )]

cgm_M=cgm_M.dropna() 



cgm_A=CGM_Data.loc[(CGM_Data['Date']>startdate_auto) | ((CGM_Data['Date']==startdate_auto) & (CGM_Data['Time']>=starttime_auto) )]

cgm_A=cgm_A.dropna()




def func_calc(subset_data, typeVal = -1, lessthan = 1, greaterthan = 1):
    if (typeVal == 1):
        val_c=subset_data.loc[subset_data.iloc[:,2]<lessthan]
    if (typeVal == 2):
        val_c=subset_data.loc[subset_data.iloc[:,2]>greaterthan]
    if (typeVal == 3):
        val_c=subset_data.loc[(subset_data.iloc[:,2]>=greaterthan) & (subset_data.iloc[:,2]<=lessthan)]
    
    cgm_date=subset_data.groupby('Date')# grouping every unique date
    c=val_c.groupby('Date').size().reset_index()# used to convert groupby to DataFrame
    c=c.set_index('Date')
    c.columns=['Value']
    p=c['Value']/288
    p=p.to_frame()
    p['Value']=p['Value']*100
    avg_val=(p['Value'].sum())/len(cgm_date)
    return avg_val
    
M_fullDay180 = func_calc(cgm_M, 2, 0, 180)

M_fullDay250 = func_calc(cgm_M, 2, 0, 250)

M_fullDay70_180 = func_calc(cgm_M, 3, 180, 70)

M_fullDay70_150 = func_calc(cgm_M, 3, 150, 70)

M_fullDay70 = func_calc(cgm_M, 1, 70, 0)

M_fullDay54 = func_calc(cgm_M, 1, 54, 0)


A_fullDay180 = func_calc(cgm_A, 2, 0, 180)

A_fullDay250 = func_calc(cgm_A, 2, 0, 250)

A_fullDay70_180 = func_calc(cgm_A, 3, 180, 70)

A_fullDay70_150 = func_calc(cgm_A, 3, 150, 70)

A_fullDay70 = func_calc(cgm_A, 1, 70, 0)

A_fullDay54 = func_calc(cgm_A, 1, 54, 0)


start=dt.strptime('00:00:00','%H:%M:%S').time()

end=dt.strptime('06:00:00','%H:%M:%S').time()

cgm_M_overnight=cgm_M.loc[(cgm_M['Time']>=start) & (cgm_M['Time']<end)]

cgm_A_overnight=cgm_A.loc[(cgm_A['Time']>=start) & (cgm_A['Time']<end)]



M_overnight_180 = func_calc(cgm_M_overnight, 2, 0, 180)

M_overnight_250 = func_calc(cgm_M_overnight, 2, 0, 250)

M_overnight_70_180 = func_calc(cgm_M_overnight, 3, 180, 70)

M_overnight_70_150 = func_calc(cgm_M_overnight, 3, 150, 70)

M_overnight_70 = func_calc(cgm_M_overnight, 1, 70, 0)

M_overnight_54 = func_calc(cgm_M_overnight, 1, 54, 0)


A_overnight_180 = func_calc(cgm_A_overnight, 2, 0, 180)

A_overnight_250 = func_calc(cgm_A_overnight, 2, 0, 250)

A_overnight_70_180 = func_calc(cgm_A_overnight, 3, 180, 70)

A_overnight_70_150 = func_calc(cgm_A_overnight, 3, 150, 70)

A_overnight_70 = func_calc(cgm_A_overnight, 1, 70, 0)

A_overnight_54 = func_calc(cgm_A_overnight, 1, 54, 0)


start=dt.strptime('06:00:00','%H:%M:%S').time()

end=dt.strptime('23:59:00','%H:%M:%S').time()

cgm_M_daytime=cgm_M.loc[(cgm_M['Time']>=start) & (cgm_M['Time']<=end)]

cgm_A_daytime=cgm_A.loc[(cgm_A['Time']>=start) & (cgm_A['Time']<end)]



M_daytime_180 = func_calc(cgm_M_daytime, 2, 0, 180)

M_daytime_250 = func_calc(cgm_M_daytime, 2, 0, 250)

M_daytime_70_180 = func_calc(cgm_M_daytime, 3, 180, 70)

M_daytime_70_150 = func_calc(cgm_M_daytime, 3, 150, 70)

M_daytime_70 = func_calc(cgm_M_daytime, 1, 70, 0)

M_daytime_54 = func_calc(cgm_M_daytime, 1, 54, 0)


A_daytime180 = func_calc(cgm_A_daytime, 2, 0, 180)

A_daytime250 = func_calc(cgm_A_daytime, 2, 0, 250)

A_daytime70_180 = func_calc(cgm_A_daytime, 3, 180, 70)

A_daytime70_150 = func_calc(cgm_A_daytime, 3, 150, 70)

A_daytime70 = func_calc(cgm_A_daytime, 1, 70, 0)

A_daytime54 = func_calc(cgm_A_daytime, 1, 54, 0)



data = [[M_overnight_180,M_overnight_250,M_overnight_70_180,M_overnight_70_150,M_overnight_70,M_overnight_54,M_daytime_180,M_daytime_250,M_daytime_70_180,M_daytime_70_150,M_daytime_70,M_daytime_54,M_fullDay180,M_fullDay250,M_fullDay70_180,M_fullDay70_150,M_fullDay70,M_fullDay54, 1.1], [A_overnight_180,A_overnight_250,A_overnight_70_180,A_overnight_70_150,A_overnight_70,A_overnight_54,A_daytime180,A_daytime250,A_daytime70_180,A_daytime70_150,A_daytime70,A_daytime54,A_fullDay180,A_fullDay250,A_fullDay70_180,A_fullDay70_150,A_fullDay70,A_fullDay54, 1.1]]

Results = pd.DataFrame(data)
Results.to_csv('Results.csv', header=False, index=False)
Results[0].append(1.1)
Results.append(1.1)