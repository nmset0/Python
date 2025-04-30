import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import seaborn as sns

# Provided by https://www.kaggle.com/datasets/aldol07/socioeconomic-factors-and-income-dataset
sgdata = pd.read_csv('Data/sgdata.csv')
print(sgdata.info())

# Education distr
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
plt.ylabel('Count')
plt.show()

# sex distr
sx = sgdata['Sex'].value_counts()
plt.bar(sx.index, sx.values, color = ['#ff69b4', '#0055b3'])
plt.xticks([0, 1], labels = ['Female', 'Male'])
plt.yticks(range(0, 1201, 100))
plt.title('Gender Distribution')
plt.xlabel('Sex')
plt.ylabel('Count')
plt.show()

# age distr
counts, bins, patches = plt.hist(sgdata['Age'], bins=20, align='mid', edgecolor='black')
norm = mcolors.Normalize(vmin=min(counts), vmax=max(counts))
colormap = cm.BuGn

for count, patch in zip(counts, patches):
    color = colormap(norm(count))
    patch.set_facecolor(color)

plt.title('Age Distribution')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()

# marital status
Mawwiage = sgdata['Marital status'].value_counts()
Mawwiage = Mawwiage.reset_index()
Mawwiage.columns = ['Marital status', 'Frequency']
sns.barplot(data = Mawwiage, x='Marital status', y='Frequency', palette = 'RdPu')
plt.title('Marriage')
plt.xlabel('Marital Status')
plt.ylabel('Count')
plt.show()

print(sgdata.drop('ID', axis = 1).describe().loc[['mean', 'std']])
