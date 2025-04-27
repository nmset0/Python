import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

SalesTax = pd.read_excel('Data/SalesTax.xlsx')
#print(SalesTax.describe())

SalesTax.loc[:, 'SalesTax'] = SalesTax.loc[:, 'SalesTax']/1000000

x = SalesTax.loc[:, 'Year'].values.reshape(-1,1)
y = SalesTax.loc[:, 'SalesTax'].values

model = LinearRegression()
model.fit(x, y)

print(f"Intercept: {round(model.intercept_, 3)}") 
print(f"Slope: {round(model.coef_[0], 3)}")  

plt.scatter(x, y, color='blue', label='Actual Data') 
y_pred = model.predict(x)  
plt.plot(x, y_pred, color='green', label='Regression Line')
plt.xlabel('Year')
plt.ylabel('Sales Tax')
plt.title('F.C. Sales Tax')
plt.legend()
plt.show()

