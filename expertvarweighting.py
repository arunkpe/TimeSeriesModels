"""
This code creates an optimizer thats weights inputs from VAR model and WSJ
"""
import pandas as pd


wsj_data = pd.read_csv('wsj_mev.csv', index_col=0, parse_dates=['date', 'forecast_date'])

df_mean_mev = wsj_data[['mev_type', 'forecast_date', 'date', 'values']]\
    .groupby(['date', 'forecast_date', 'mev_type']).mean().reset_index()

df_median_mev = wsj_data[['mev_type', 'forecast_date', 'date', 'values']]\
    .groupby(['date', 'forecast_date', 'mev_type']).median().reset_index()

df_median_mev.sort_values(by=['date', 'forecast_date'], inplace=True)
