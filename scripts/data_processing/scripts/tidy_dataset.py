'''
    Initialise Script
'''
# Import packages
import sweetviz as sv
import pandas as pd
import numpy as np

# Define useful one-liner functions
flatten_list = lambda list_of_lists: [x for xs in list_of_lists for x in xs]
unique_list = lambda list_with_duplicates: list(set(list(list_with_duplicates)))


'''
    Tidy Data
'''
# Group columns
cols_drop = ['net_lend','visa']
cols_index = ['date','data_type']
cols_resp = ['y']
cols_monthly = ['cpi','m4','m4_lend','loans','br','gscpi','GBP/US','GBP/EU','unemply','awe','oil_pl','A_MoM','B_MoM','BE_MoM','C_MoM','D_MoM','E_MoM','F_MoM','G_MoM','GT_MoM','GVA_MoM','H_MoM','I_MoM','IOS_MoM','IOP_MoM','J_MoM','K_MoM','L_MoM','M_MoM','O_MoM','N_MoM','P_MoM','Q_MoM','R_MoM','S_MoM','T_MoM']
cols_quarter = ['us_gdp','eu_gdp','sr','imp','exp','expp','pop','A_QoQ','BE_QoQ','C_QoQ','F_QoQ','GT_QoQ','P3G_QoQ','P3H_QoQ','P3N_QoQ','P51_QoQ','P51S_QoQ','P6_QoQ','P7_QoQ']
cols_monthly_level = ['cpi', 'br', 'unemply', 'awe', 'GBP/US', 'GBP/EU', 'oil_pl']
cols_quarter_level = ['us_gdp', 'eu_gdp', 'sr', 'pop']

# Download data & drop column
df_train = pd.read_excel('data_processing/raw_data/gdp_base.xlsx',sheet_name='df_current_train')
df_test = pd.read_excel('data_processing/raw_data/gdp_base.xlsx',sheet_name='df_current_test')
df = pd.concat([
    df_train.assign(data_type = 'train'),
    df_test[[False if x in df_train.date.tolist() else True for x in df_test.date]].assign(data_type = 'test')
])
df.drop(inplace=True,columns=cols_drop)

# Add lag columns
lags = 12
df_inputs = df[cols_index+cols_resp+cols_monthly+cols_quarter]
df_final = df.copy(deep=True)
for i in range(1,lags+1):
    df_input_lag = df_inputs.assign(
        date = lambda df: [x + pd.DateOffset(months=i) for x in df.date]
    ).rename(columns={x:f'{x}_{i}' for x in [col for col in cols_resp+cols_monthly+cols_quarter]})
    df_final = df_final.merge(df_input_lag.drop(columns='data_type'),on='date',how='left')

# Keep only columns we want
cols = unique_list(flatten_list([
    cols_index,
    ['y','y_3'],
    [f"A_QoQ_{i}" for i in [2,5,8,11]],
    [f"cpi_{i}" for i in range(1,6)],
    flatten_list([[f"{col}_{i}" for i in [1,2,3,4]] for col in ['I_MoM','T_MoM']]),
    flatten_list([[f"{col}_{i}" for i in [1,2,3]] for col in cols_monthly_level]),
    flatten_list([[f"{col}_{i}" for i in [1,2]] for col in cols_monthly if col not in cols_monthly_level]),
    flatten_list([[f"{col}_{i}" for i in [2,5]] for col in cols_quarter_level + ['P3H_QoQ']]),
    flatten_list([[f"{col}_{i}" for i in [2]] for col in cols_quarter if col not in cols_quarter_level]),
]))
# Filter for date that's suitable for all variables
df_final = df_final[df_final['y'].notnull()][cols]
df_final = df_final[df_final['date'] > '2000-08-01'].reset_index(drop=True)
df_final.isnull().sum().sum()

# cpi 5
# I_MOM 4
# A_QoQ 4
# T_MoM 4
# P3H_QoQ 2

'''
    Rate change
'''
# Calculate the monthly percentage change
for i in range(1,3):
    df_final = df_final.assign(**{
        f'{col}_pc_{i}':lambda df: (df[f"{col}_{i}"] - df[f"{col}_{i+1}"]) / df[f"{col}_{i+1}"] for col in cols_monthly_level
    })
# Calculate the quarterly percentage change
df_final = df_final.assign(**{
    f'{col}_pc_1':lambda df: (df[f"{col}_2"] - df[f"{col}_5"]) / df[f"{col}_5"] for col in cols_quarter_level
})

#
df_final.to_csv('data_processing/outputs/df_tidy.csv',index=False)

'''
    EDA
'''
my_report = sv.analyze(df_final)
my_report.show_html()