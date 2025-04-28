import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# import statsmodels as st
# import prophet as pr

FinalData = pd.read_excel('Data/FinalData.xlsx')
TrafficVolume = pd.read_excel('Data/TrafficVolume.xlsx')

# Convert datetime column 
FinalData['Month'] = pd.to_datetime(FinalData['Month'], format='%b-%y')
FinalData['Month'] = FinalData['Month'].dt.month

# Necessary adjustments
FinalData.loc[FinalData['year'] == 2024, 'Lead_STF_Real'] *= (3.85 / 4.35)
FinalData.iloc[:12, FinalData.columns.get_loc('SG')] += 5   # ".column.get_loc"
FinalData.iloc[:12, FinalData.columns.get_loc('G')] += 5  
FinalData.iloc[:12, FinalData.columns.get_loc('EDUHS')] -= 5   

