# Python 3 Essentials for Statistics, Data Analysis, and Data Science

# --- 1. NUMPY ---
# NumPy is used for numerical computing and array operations
import numpy as np

# Creating an array
arr = np.array([1, 2, 3, 4, 5])
print("Array:", arr)

# Basic statistics
print("Mean:", np.mean(arr))
print("Standard Deviation:", np.std(arr))

# --- 2. PANDAS ---
# Pandas is used for data manipulation and analysis
import pandas as pd

# Reading CSV and Excel files
df_csv = pd.read_csv('data.csv')
df_xlsx = pd.read_excel('data.xlsx')

# Display first few rows
print(df_csv.head())

# Pivoting and joining
pivoted = df_csv.pivot_table(index='Category', values='Sales', aggfunc='sum')
joined = pd.merge(df_csv, df_xlsx, on='ID', how='inner')

# --- 3. MATPLOTLIB ---
# Matplotlib is used for plotting and visualization
import matplotlib.pyplot as plt

# Line plot
plt.plot(arr)
plt.title("Line Plot")
plt.xlabel("Index")
plt.ylabel("Value")
plt.show()

# --- 4. SEABORN ---
# Seaborn is used for statistical data visualization
import seaborn as sns

# Histogram
sns.histplot(df_csv['Sales'], kde=True)
plt.title("Sales Distribution")
plt.show()

# --- 5. BASIC STATISTICAL METHODS ---
# Mean, median, mode, correlation, etc.
print("Median:", df_csv['Sales'].median())
print("Correlation matrix:")
print(df_csv.corr())

# --- 6. SCIKIT-LEARN ---
# Scikit-learn is used for machine learning
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Preparing data
X = df_csv[['Advertising']]
y = df_csv['Sales']

# Splitting data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Training a linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions and evaluation
y_pred = model.predict(X_test)
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))

# --- 7. TENSORFLOW ---
# TensorFlow is used for deep learning
import tensorflow as tf

# Creating a simple neural network model
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(1)
])
model.compile(optimizer='adam', loss='mean_squared_error')

# Training the model
model.fit(X_train, y_train, epochs=10)

# --- 8. PYTORCH ---
# PyTorch is another deep learning library
import torch
import torch.nn as nn
import torch.optim as optim

# Convert data to tensors
X_tensor = torch.tensor(X_train.values, dtype=torch.float32)
y_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)

# Define a simple model
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc = nn.Linear(X_train.shape[1], 1)

    def forward(self, x):
        return self.fc(x)

model = SimpleNN()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Training loop
for epoch in range(10):
    optimizer.zero_grad()
    outputs = model(X_tensor)
    loss = criterion(outputs, y_tensor)
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item()}")

# --- 9. OTHER USEFUL TOOLS ---
# SciPy for scientific computations
from scipy import stats
print("T-test result:", stats.ttest_1samp(df_csv['Sales'], popmean=100))

# Statsmodels for statistical modeling
import statsmodels.api as sm
X_with_const = sm.add_constant(X)
model_sm = sm.OLS(y, X_with_const).fit()
print(model_sm.summary())
