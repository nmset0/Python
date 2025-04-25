# .venv\Scripts\activate

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

price = pd.read_csv('Data/FMAP-Data.csv')
PR = pd.DataFrame(price)
print(PR)