import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Provided by https://www.kaggle.com/datasets/aldol07/socioeconomic-factors-and-income-dataset
sgdata = pd.read_csv('Data/sgdata.csv')
print(sgdata.info())

educ = sgdata['Education'].value_counts()
educ = pd.concat([educ[educ.index != 'other / unknown'], educ[educ.index == 'other / unknown']])

# matplotlib
# plt.bar(educ.index, educ.values)
# plt.title('Education')
# plt.xlabel('Education Level')
# plt.ylabel('Frequency')
# plt.show()

# seaborn
educ = educ.reset_index()
educ.columns = ['Education', 'Frequency']
sns.barplot(data=educ, x='Education', y='Frequency', palette='inferno')
plt.title('Education')
plt.xlabel('Education Level')
plt.ylabel('Frequency')
plt.show()

sx = sgdata['Sex'].value_counts()
plt.bar(sx.index, sx.values, color = ['#ff69b4', '#0055b3'])
plt.xticks([0, 1], labels = ['Female', 'Male'])
plt.yticks(range(0, 1201, 100))
plt.title('Gender Distribution')
plt.xlabel('Sex')
plt.ylabel('Frequency')
plt.show()

