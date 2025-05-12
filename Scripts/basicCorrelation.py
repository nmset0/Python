import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
from scipy.stats import spearmanr, kendalltau

X = np.random.normal(random.uniform(0, .75), random.uniform(0.1, 5), 100)
Y = np.random.normal(random.uniform(0, .75), random.uniform(0.1, 5), 100)
Z = np.random.poisson(lam = random.randint(0, 5), size = random.randint(1, 10))
empty = pd.DataFrame()

plt.scatter(X, Y)
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Rsample X vs. Rsample Y')
plt.show()

def correlation(x, y, method):
    if len(x) == 0 or len(y) == 0:
        c = 'NA'
        print('One or more variables is empty. Correlation cannot be calculated.')
    elif len(x) != len(y):
        print('X and Y are not the same length. Correlation cannot be calculated.')
    else:
        if method == 'pearson':
            c = round(np.corrcoef(x, y)[0, 1], 3)
        elif method == 'spearman':
            c = spearmanr(x, y)
        elif method == 'kendall':
            c = kendalltau(x, y)
        else:
            print('Not An Applicable Method.')
        print(f'Correlation Coefficient: {c}')

correlation(X, Y, 'pearson')
correlation(Y, X, 'spearman')
correlation(X, empty, 'pearson')
correlation(X, Z, 'kendall')


def is_subset(empty_set, universal_set):
    for element in empty_set:
        if element not in universal_set:
            return False
    return True  

empty_set = set()  # Empty set
set_a = {1, 2, 3}
set_b = {"apple", "banana", "cherry"}
set_c = set()  # Another empty set

print("Is the empty set a subset of set_a?", is_subset(empty_set, set_a))
print("Is the empty set a subset of set_b?", is_subset(empty_set, set_b))
print("Is the empty set a subset of set_c?", is_subset(empty_set, set_c))
print("Is set_b set a subset of set_a?", is_subset(set_b, set_a))
print("Is set_a set a subset of set_b?", is_subset(set_a, set_b))
