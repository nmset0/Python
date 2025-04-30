# import numpy as np
import statsmodels.api as stm
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# import pingouin as pg

# Provided by https://www.kaggle.com/datasets/aldol07/socioeconomic-factors-and-income-dataset
sgdata = pd.read_csv('Data/sgdata.csv')

sgdata = sgdata.sort_values(by='Age', ascending=True)
sgdata = sgdata.reset_index(drop=True)

print(sgdata.info())

# #Education distr
# educ = sgdata['Education'].value_counts()
# educ = pd.concat([educ[educ.index != 'other / unknown'], educ[educ.index == 'other / unknown']])

# #matplotlib
# plt.bar(educ.index, educ.values)
# plt.title('Education')
# plt.xlabel('Education Level')
# plt.ylabel('Frequency')
# plt.show()

# #seaborn
# educ = educ.reset_index()
# educ.columns = ['Education', 'Frequency']
# sns.barplot(data=educ, x='Education', y='Frequency', palette='inferno')
# plt.title('Education')
# plt.xlabel('Education Level')
# plt.ylabel('Count')
# plt.show()

# #sex distr
# sx = sgdata['Sex'].value_counts()
# plt.bar(sx.index, sx.values, color = ['#ff69b4', '#0055b3'])
# plt.xticks([0, 1], labels = ['Female', 'Male'])
# plt.yticks(range(0, 1201, 100))
# plt.title('Gender Distribution')
# plt.xlabel('Sex')
# plt.ylabel('Count')
# plt.show()

# #age distr
# counts, bins, patches = plt.hist(sgdata['Age'], bins=20, align='mid', edgecolor='black')
# norm = mcolors.Normalize(vmin=min(counts), vmax=max(counts))
# colormap = cm.BuGn

# for count, patch in zip(counts, patches):
#     color = colormap(norm(count))
#     patch.set_facecolor(color)

# plt.title('Age Distribution')
# plt.xlabel('Age')
# plt.ylabel('Frequency')
# plt.show()

# #marital status
# Mawwiage = sgdata['Marital status'].value_counts()
# Mawwiage = Mawwiage.reset_index()
# Mawwiage.columns = ['Marital status', 'Frequency']
# sns.barplot(data = Mawwiage, x='Marital status', y='Frequency', palette = 'RdPu')
# plt.title('Marriage')
# plt.xlabel('Marital Status')
# plt.ylabel('Count')
# plt.show()


# Linear Models
print(sgdata.drop('ID', axis = 1).describe().loc[['mean', 'std']])


# Getting response and predictor
Y = sgdata['Income']
X = sgdata[['Age']]
X = stm.add_constant(X)

# Making OLS model
model = stm.OLS(Y, X) # Income ~ Age
result = model.fit()
print(result.summary())

# change points' color based on settlement size
colorMap = {0: '#bf0a30', 
            1: '#2e6f40', 
            2: '#6c3baa'
            }
colors = sgdata['Settlement size'].map(colorMap)

# change points' shapes based on occupation
markerMap = {'skilled employee / official': 'o', 
             'unemployed / unskilled': 's', 
             'management / self-employed / highly qualified employee / officer': '^'
             } 
markers = sgdata['Occupation'].map(markerMap)

# scatterplot
for employment_status, marker in markerMap.items():
    subset = sgdata[sgdata['Occupation'] == employment_status]
    plt.scatter(subset['Age'], subset['Income'], c=subset['Settlement size'].map(colorMap),
                alpha=0.7, label=employment_status, marker=marker)

# settlement size legend
color_handles = [
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10, label=f'Settlement size {i}')
    for i, color in colorMap.items()
]
# occupation legend
marker_handles = [
    plt.Line2D([0], [0], marker=marker, color='w', markerfacecolor='black', markersize=10, label=label)
    for label, marker in markerMap.items()
]

plt.legend(
    handles=color_handles + marker_handles,
    title='Legend',
    loc='upper left',
    bbox_to_anchor=(1.05, 1)  # Position the legend outside the plot
)
plt.title('Age vs. Income by Settlement Size')
plt.xlabel('Age')
plt.ylabel('Income')
plt.show()