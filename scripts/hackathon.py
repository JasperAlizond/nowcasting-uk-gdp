'''
    Initialise Script
'''
# Import packages
import sweetviz as sv
import pandas as pd
import numpy as np

# Define useful one-liner functions
flatten_list = lambda list_of_lists: [x for xs in list_of_lists for x in xs]



'''
    Import dataset
'''
# Download data and create EDA report
df_train = pd.read_excel('raw_data/gdp_base.xlsx',sheet_name='df_current_train')
my_report = sv.analyze(df_train)
my_report.show_html()

# Split dataframe to inputs and response variables
df_response = df_train[['date','y']]
df_input = df_train[['date']+df_train.columns.tolist()[3:]]

# Add lag to input variables
for i in range(1,4):
    df_input_lag = df_input.assign(
        date = lambda df: [x + pd.DateOffset(months=i) for x in df.date]
    ).rename(columns={x:f'{x}_{i}' for x in [col for col in df_input.columns if col != 'date']})
    df_input = df_input.merge(df_input_lag,on='date',how='left')
# Reorder columns and save
col_order = ['date']+flatten_list([[col]+[f"{col}_{i}" for i in range(1,4)] for col in df_train.columns.tolist()[3:]])
df_input[col_order].to_csv('outputs\input_data.csv',index=False)