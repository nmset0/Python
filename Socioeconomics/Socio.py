import pandas as pd
import statsmodels.api as stm
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
from matplotlib.patches import Patch
import seaborn as sns
import pingouin as pg
from scipy.stats import chi2_contingency

# Provided by https://www.kaggle.com/datasets/aldol07/socioeconomic-factors-and-income-dataset
sgdata = pd.read_csv('Data/sgdata.csv')

sgdata = sgdata.sort_values(by='Age', ascending=True)
sgdata = sgdata.reset_index(drop=True)

sgdata['Settlement size'] = sgdata['Settlement size'].astype('category')
sgdata['Sex'] = sgdata['Sex'].astype('category')

print(sgdata.info())

settlementSizePercent = sgdata['Settlement size'].value_counts(normalize=True) * 100
print("Percentage of each Settlement size:")
print(settlementSizePercent)

#Education distr
educ = sgdata['Education'].value_counts()
educ = pd.concat([educ[educ.index != 'other / unknown'], educ[educ.index == 'other / unknown']])

#matplotlib
plt.bar(educ.index, educ.values)
plt.title('Education')
plt.xlabel('Education Level')
plt.ylabel('Frequency')
plt.show()

#seaborn
educ = educ.reset_index()
educ.columns = ['Education', 'Frequency']
sns.barplot(data=educ, x='Education', y='Frequency', palette='inferno')
plt.title('Education')
plt.xlabel('Education Level')
plt.ylabel('Count')
plt.show()

#sex distr
sx = sgdata['Sex'].value_counts()
plt.bar(sx.index, sx.values, color = ['#ff69b4', '#0055b3'])
plt.xticks([0, 1], labels = ['Female', 'Male'])
plt.yticks(range(0, 1201, 100))
plt.title('Gender Distribution')
plt.xlabel('Sex')
plt.ylabel('Count')
plt.show()

#age distr
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

#marital status
Mawwiage = sgdata['Marital status'].value_counts()
Mawwiage = Mawwiage.reset_index()
Mawwiage.columns = ['Marital status', 'Frequency']
sns.barplot(data = Mawwiage, x='Marital status', y='Frequency', palette = 'RdPu')
plt.title('Marriage')
plt.xlabel('Marital Status')
plt.ylabel('Count')
plt.show()


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
# lt.show()

# Correlation / ANOVA

# correlation between age and income
incomeAge_corr = pg.corr(sgdata['Age'], 
                         sgdata['Income'],
                         alternative = 'two-sided',
                         method = 'pearson')
print(f"Age and Income Correlation (r): {round(incomeAge_corr['r'], 3)}")
print(f"P-value: {incomeAge_corr['p-val']}")

# anova between income between settlement sizes
incomeSettlement = pg.welch_anova(dv = 'Income', 
                                  between = 'Settlement size', 
                                  data = sgdata)
print(f"Income and Settlement Size F-Statistic: {round(incomeSettlement['F'], 3)}")
print(f"Income and Settlement Size P-Value: {incomeSettlement['p-unc']}")


# Chi-Square test for Settlement size and Occupation
contingency_table_occupation = pd.crosstab(sgdata['Settlement size'], sgdata['Occupation'])
chi2_occupation, p_occupation, dof_occupation, expected_occupation = chi2_contingency(contingency_table_occupation)
print("Chi-Square Test for Settlement size and Occupation")
print(f"Chi-Square Statistic: {chi2_occupation:.3f}")
print(f"P-Value: {p_occupation}")
print("Expected Frequencies:")
print(expected_occupation)

# Chi-Square test for Settlement size and Education
contingency_table_education = pd.crosstab(sgdata['Settlement size'], sgdata['Education'])
chi2_education, p_education, dof_education, expected_education = chi2_contingency(contingency_table_education)
print("\nChi-Square Test for Settlement size and Education")
print(f"Chi-Square Statistic: {chi2_education:.3f}")
print(f"P-Value: {p_education}")
print("Expected Frequencies:")
print(expected_education)


# anova between income and education level
incomeEducation = pg.welch_anova(dv = 'Income',
                                  between = 'Education', 
                                  data = sgdata)
print(f"Income and Education F-Statistic: {round(incomeEducation['F'], 3)}")
print(f"Income and Education P-Value: {incomeEducation['p-unc']}")

# anova between income and occupation
incomeOccupation = pg.welch_anova(dv = 'Income', 
                                  between = 'Occupation', 
                                  data = sgdata)
print(f"Income and Occupation F-Statistic: {round(incomeOccupation['F'], 3)}")
print(f"Income and Occupation P-Value: {incomeOccupation['p-unc']}")

# anova between income and sex
incomeSex = pg.welch_anova(dv = 'Income', 
                                  between = 'Sex', 
                                  data = sgdata)
print(f"Income and Sex F-Statistic: {round(incomeSex['F'], 3)}")
print(f"Income and Sex P-Value: {incomeSex['p-unc']}")


# income by occupation
income_by_occupation = sgdata.groupby('Occupation')['Income'].mean()
print(f"Average Income by Occupation: {income_by_occupation}")


# distribution of income for each education level 
sns.boxplot(data=sgdata, x='Education', y='Income', palette='coolwarm')
plt.title('Income Distribution by Education Level')
plt.xlabel('Education Level')
plt.ylabel('Income')
plt.xticks(rotation=45)
plt.show()

# Age distribution for each settlement size
sns.histplot(data=sgdata, x='Age', hue='Settlement size', multiple='stack', palette='Set2', bins=20)

settlement_sizes = sgdata['Settlement size'].cat.categories  # Get unique categories
palette = sns.color_palette('Set2', len(settlement_sizes))  # Match the palette used in the plot
legend_handles = [Patch(color=palette[i], label=f'Settlement size {settlement_sizes[i]}') for i in range(len(settlement_sizes))]

plt.title('Age Distribution by Settlement Size')
plt.xlabel('Age')
plt.ylabel('Count')
plt.legend(handles=legend_handles, title='Settlement Size', loc='upper right')
plt.show()

