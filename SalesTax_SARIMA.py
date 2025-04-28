import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.vector_ar.vecm import coint_johansen 
from statsmodels.tsa.vector_ar.vecm import VECM
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller

FinalData = pd.read_excel('Data/FinalData.xlsx')
TrafficVolume = pd.read_excel('Data/TrafficVolume.xlsx')

# Cleaning

# Convert datetime column 
FinalData['Month'] = pd.to_datetime(FinalData['Month'], format='%b-%y')
FinalData['Month'] = FinalData['Month'].dt.month

# Necessary adjustments
FinalData.loc[FinalData['year'] == 2024, 'Lead_STF_Real'] *= (3.85 / 4.35)
FinalData.iloc[:11, FinalData.columns.get_loc('SG')] += 5   # ".column.get_loc"
FinalData.iloc[:11, FinalData.columns.get_loc('G')] += 5  
FinalData.iloc[:11, FinalData.columns.get_loc('EDUHS')] -= 5   


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

# Dickey-Fuller unit root test
def dickeyfuller_test(timeseries):
    # Perform the Dickey-Fuller test
    result = adfuller(timeseries, autolag='AIC')
    p_value = result[1]
    print(f'ADF Statistic: {result[0]}')
    print(f'p-value: {p_value}')
    if p_value < 0.05:
        print('The time series is stationary.')
    else:
        print('The time series is not stationary.')

for column in ldf.columns:
    print(f"\nDickey-Fuller Test for {column}:")
    dickeyfuller_test(ldf[column])

# ldf = ldf.diff().dropna()



# Variables
p, d, q = 1, 1, 1
P, D, Q = 1, 1, 1
s = 12
nhor = 12

# Evaluate model on 2024 forecast
ldf2023 = ldf[FinalData_clean['year'] <= 2023]
sarima_model_2023 = SARIMAX(ldf2023['Lead_STF_Real'], order=(p, d, q), seasonal_order=(P, D, Q, s))
sarima_results_2023 = sarima_model_2023.fit()
print(sarima_results_2023.summary())

# forecasting 2024
forecast2024 = sarima_results_2023.get_forecast(steps = nhor)
forecast_mean_2024 = forecast2024.predicted_mean
forecast_ci_2024 = forecast2024.conf_int()

# Comparing observed vs. predicted
Actual2024 = ldf['Lead_STF_Real'][FinalData_clean['year'] == 2024]
AE = abs(Actual2024 - forecast_mean_2024)
MAE = np.mean(AE)
print(f'MAE: {round(MAE, 3)}')


# Forecast 2025
sarima_model_2025 = SARIMAX(ldf['Lead_STF_Real'], order=(p, d, q), seasonal_order=(P, D, Q, s))
sarima_results_2025 = sarima_model_2025.fit()
forecast2025 = sarima_results_2025.get_forecast(steps = nhor)
forecast_mean_2025 = forecast2025.predicted_mean
forecast_ci = forecast2025.conf_int()
 # plot forecast
plt.figure(figsize=(12, 6))
plt.plot(ldf['Lead_STF_Real'], label='Observed')
plt.plot(forecast_mean_2025, label='Forecast', color='red')
plt.fill_between(forecast_ci.index, forecast_ci.iloc[:, 0], forecast_ci.iloc[:, 1], color='pink')
plt.title("Sales Tax Forecast")
plt.xlabel("Date")
plt.ylabel("Sales Tax")
plt.legend()
plt.show()

forecast_mean_2025 *= (4.35 / 3.85)
print(np.exp(forecast_mean_2025)*1000000)
