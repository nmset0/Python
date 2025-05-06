import pandas as pd

# Raw Data
agriculture = pd.read_csv('Data/AgrCensus_Output_raw.csv')

# Rename "data.item" to "ID" for intuitiveness
agriculture.rename(columns = {'data.item': 'ID'}, inplace = True)

# Remove unwanted columns
agriculture_1 = agriculture.drop(columns = ['domain.category', 'state.fips', 'commodity', 'county.code'])

print(agriculture_1[0:10]) # view data

# Changing column names to lowercase
agriculture_1 = agriculture_1.map(lambda x: x.lower() if isinstance(x, str) else x)

# Pivoting
agriculture_wide = agriculture_1.pivot_table(index=['state', 'county'], columns = 'ID', values = 'value', aggfunc = 'first').reset_index()

# Alter punctuation
agriculture_wide.columns = agriculture_wide.columns.str.replace(r"[:\-\,/ ]", "_", regex = True).str.replace("__", "_", regex = False)

# Replace cells that contain "(d)" 
agriculture_wide = agriculture_wide.replace(r'\(d\)', 0, regex = True)
# Fill NaN
agriculture_wide = agriculture_wide.fillna(0)

pd.set_option('display.max_columns', 5)
print(agriculture_wide[0:10]) # view new data frame

# Save clean data as new .csv file
# agriculture_wide.to_csv('Data/AgrCensus_Output_clean.csv', index = False)