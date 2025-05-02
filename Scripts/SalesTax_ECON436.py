# Packages
import pandas as pd
import numpy as np
from statsmodels.tsa.vector_ar.vecm import coint_johansen, VECM
from statsmodels.tsa.api import VAR
import matplotlib.pyplot as plt

FD = pd.read_excel("Data/FinalData.xlsx")
TRAF_raw = pd.read_excel("Data/TrafficVolume.xlsx")
# slightly edited from original data sets

# Cleaning 

# Necessary Adjustments
FD.loc[FD['year'] == 2024, 'Lead_STF_Real'] *= (3.85 / 4.35)
FD.loc[:11, 'SG'] += 5
FD.loc[:11, 'G'] += 5
FD.loc[:11, 'EDUHS'] -= 5
FD['Month'] = pd.to_datetime(FD['Month'], format='%b-%y')
FD['Month'] = FD['Month'].dt.month

Traffic_long = TRAF_raw.melt(id_vars=['Year'], var_name='Month', value_name='Value')
Traffic_long['Month'] = pd.to_datetime(Traffic_long['Month'], format='%b').dt.month
TRAF_long = Traffic_long.sort_values(by=['Year', 'Month'], ascending=True).rename(columns={'Year': 'year'})

# Bind Traffic Data and Final Data
DF = pd.merge(FD, TRAF_long, on = ['year', 'Month'], how = 'left')
DF.rename(columns={'Value': 'traffic_frequency'}, inplace=True)

# Selecting predictors + Lead_STF_Real
DF_NEW = DF[['LH', 'INFO', 'EDUHS', 'MAN', 'AFS', 'RT', 'traffic_frequency', 'Lead_STF_Real']]

# Log all variables
LDF = DF_NEW.apply(np.log)

# Cointegrating Vectors (Johansen Test)
johansen_result = coint_johansen(LDF, det_order=0, k_ar_diff=12)
BETA = np.round(johansen_result.evec[:, :6], 3)  # r = 6
print(BETA)

# Short-run Adjustment Coefficients (VECM)
vecm_model = VECM(LDF, k_ar_diff=11, coint_rank=6, deterministic='co')
vecm_result = vecm_model.fit()
print(vecm_result.summary())

# Evaluate Model Performance
LDF_2023 = LDF.iloc[:-12, :]
johansen_2023 = coint_johansen(LDF_2023, det_order=0, k_ar_diff=12)

vec2var_model = VAR(LDF_2023)
vec2var_fitted = vec2var_model.fit(maxlags=11)
forecast = vec2var_fitted.forecast(LDF_2023.values[-vec2var_fitted.k_ar:], steps=12)
Forecast2024 = np.exp(forecast[:, LDF.columns.get_loc('Lead_STF_Real')])
Actual2024 = np.exp(LDF.loc[DF['year'] == 2024, 'Lead_STF_Real'])

# Calculate MAE
AE = np.abs(Forecast2024 - Actual2024.values)
MAE = np.mean(AE)
print(round(MAE, 4))

# Transforming VECM to VAR and Forecasting
vec2var_full_model = VAR(LDF)
vec2var_full_fitted = vec2var_full_model.fit(maxlags=11)
forecast_full = vec2var_full_fitted.forecast(LDF.values[-vec2var_full_fitted.k_ar:], steps=11)

# Predicted Values
forecast_full[:, LDF.columns.get_loc('Lead_STF_Real')] = np.exp(forecast_full[:, LDF.columns.get_loc('Lead_STF_Real')])
forecast_full[:, LDF.columns.get_loc('Lead_STF_Real')] *= (4.35 / 3.85)
print(f"2025 Forecast: {forecast_full[:, LDF.columns.get_loc('Lead_STF_Real')]}")

# Graphical Representation
plt.plot(forecast_full[:, LDF.columns.get_loc('Lead_STF_Real')], label='Forecast')
plt.show()