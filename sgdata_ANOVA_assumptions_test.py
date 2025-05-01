import pandas as pd
import scipy.stats as stats
from scipy.stats import shapiro, levene, bartlett

sgdata = pd.read_csv('Data/sgdata.csv')


# normality
print("Normality Test (Shapiro-Wilk):")
for column in sgdata.select_dtypes(include=['float64', 'int64']).columns:
    stat, p = shapiro(sgdata[column].dropna())
    print(f"{column}: W={stat:.4f}, p={p:.4f} {'(Normal)' if p > 0.05 else '(Not Normal)'}")

# equal variances
if 'Settlement size' in sgdata.columns:
    print("\nHomogeneity of Variances Test (Levene):")
    groups = [sgdata[sgdata['Settlement size'] == group]['Income'].dropna() for group in sgdata['Settlement size'].unique()]
    stat, p = levene(*groups)
    print(f"Levene Test: W={stat:.4f}, p={p:.4f} {'(Equal Variances)' if p > 0.05 else '(Unequal Variances)'}")

    print("\nHomogeneity of Variances Test (Bartlett):")
    stat, p = bartlett(*groups)
    print(f"Bartlett Test: W={stat:.4f}, p={p:.4f} {'(Equal Variances)' if p > 0.05 else '(Unequal Variances)'}")

# independence
print("\nIndependence Assumption: Ensure independence by study design.")