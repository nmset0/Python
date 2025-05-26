import numpy as np
import pandas as pd

FMAP = pd.read_csv('Data/FMAP-Data.csv')

FMAP_wide = pd.pivot_table(FMAP, index = ['Year', 'Month', 'EFPG_code'], columns = ['Attribute'], values = 'Value')

FMAP_wide.iloc[1:] = FMAP_wide.iloc[1:].apply(pd.to_numeric, errors='coerce')

for col in FMAP_wide.select_dtypes(include=['float', 'object']):
    FMAP_wide[col] = FMAP_wide[col].astype(str).str.lstrip("'")

# FMAP_wide.to_csv('Data/FMAP_wide.csv', index = True)