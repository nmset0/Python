from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

FRAME = pd.read_csv('Data/CombinedData_H2AWorkers_FEMA_AgrOutput_Healthcare.csv')
FRAME = FRAME.drop(FRAME.columns[0], axis=1)

Y = FRAME['MigrantHealthCenters']
X = FRAME.drop(columns=['MigrantHealthCenters'] + FRAME.columns[351:368].tolist())  # Drop response and healthcare facility columns
X = X._get_numeric_data()

forestModel = RandomForestClassifier(random_state=335, criterion='gini')
forestModel.fit(X, Y)

importances = forestModel.feature_importances_
feature_importances = pd.Series(importances, index=X.columns)
feature_importances_sorted = feature_importances.sort_values(ascending=False)

Z = feature_importances_sorted.iloc[0:10]
rowNamesZ = Z.index

plt.figure(figsize=(10, 6))
plt.bar(feature_importances_sorted.index, feature_importances_sorted.values)
plt.xticks(rotation=90, fontsize=4.5)
plt.xlabel('Features')
plt.ylabel('Importance')
plt.title('Random Forest Feature Importances')
plt.show()

pred = FRAME[rowNamesZ]
pred = pred.dropna()  
Y = Y.loc[pred.index]  

pred = sm.add_constant(pred)

model = sm.OLS(Y, pred).fit()
print(model.summary())



