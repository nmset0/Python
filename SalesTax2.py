import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.vector_ar.vecm import coint_johansen 
from statsmodels.tsa.vector_ar.vecm import VECM
# import prophet as pr

FinalData = pd.read_excel('Data/FinalData.xlsx')
TrafficVolume = pd.read_excel('Data/TrafficVolume.xlsx')

# Cleaning

# Convert datetime column 
FinalData['Month'] = pd.to_datetime(FinalData['Month'], format='%b-%y')
FinalData['Month'] = FinalData['Month'].dt.month

# Necessary adjustments
FinalData.loc[FinalData['year'] == 2024, 'Lead_STF_Real'] *= (3.85 / 4.35)
FinalData.iloc[:12, FinalData.columns.get_loc('SG')] += 5   # ".column.get_loc"
FinalData.iloc[:12, FinalData.columns.get_loc('G')] += 5  
FinalData.iloc[:12, FinalData.columns.get_loc('EDUHS')] -= 5   


Traffic_long = TrafficVolume.melt(id_vars=['Year'],  var_name='Month',  value_name='Value' ) # melt for wide to long

Traffic_long['Month'] = pd.to_datetime( Traffic_long['Month'], format='%b').dt.month
Traffic_long['Month'] = pd.to_datetime(Traffic_long['Year'].astype(str) + '-' + Traffic_long['Month'].astype(str) + '-01')

Traffic_long = Traffic_long[['Month', 'Year', 'Value']]
Traffic_long['Month'] = Traffic_long['Month'].dt.month
Traffic_long = Traffic_long.sort_values(by=['Year', 'Month'], ascending=True)
Traffic_long.rename(columns={'Year': 'year'}, inplace=True)
Traffic_long.rename(columns={'Value': 'traffic_frequency'}, inplace=True)

FinalData_clean = pd.merge(FinalData, Traffic_long, on = ['year', 'Month'], how = 'left')

df = FinalData_clean[['LH', 'EDUHS', 'MAN', 'AFS', 'RT', 'traffic_frequency', 'Lead_STF_Real']]
ldf = np.log(df) # Log all variables

# Conduct the Johansen procedure
johansen = coint_johansen(ldf, det_order = 2, k_ar_diff = 12)
print(johansen.lr1)
for i, trace_stat in enumerate(johansen.lr1):
    print(f"Hypothesis r <= {i}: Trace Statistic = {trace_stat}, Critical Values = {johansen.cvt[i]}")

# Cointegration coefficients
BETA = np.round(johansen.evec[:, :6], 3)  

# Short-run adjustment coefficients
vecm = VECM(ldf, k_ar_diff = 12, coint_rank = 6, deterministic = "co") 
vecm_fit = vecm.fit(method = "ml") 


