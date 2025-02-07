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
df_train = pd.read_excel('data_processing/raw_data/gdp_base.xlsx',sheet_name='df_current_train')
my_report = sv.analyze(df_train)
my_report.show_html()

# Split dataframe to inputs and response variables
cols_train = df_train.columns.tolist()[3:]
df_response = df_train[['date','y']]
df_input = df_train[['date']+cols_train]

# Add lag to input variables
for i in range(1,4):
    df_input_lag = df_input.assign(
        date = lambda df: [x + pd.DateOffset(months=i) for x in df.date]
    ).rename(columns={x:f'{x}_{i}' for x in [col for col in df_input.columns if col != 'date']})
    df_input = df_input.merge(df_input_lag,on='date',how='left')
# Reorder columns and save
col_order = ['date']+flatten_list([[col]+[f"{col}_{i}" for i in range(1,4)] for col in cols_train])
df_response.merge(df_input[col_order],on='date').to_csv('outputs\input_data.csv',index=False)

my_report = sv.analyze(df_response.merge(df_input[col_order],on='date'),pairwise_analysis='on')
my_report.show_html()