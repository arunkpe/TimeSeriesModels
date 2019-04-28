"""
This code creates an optimizer thats weights inputs from VAR model and WSJ
"""

import pandas as pd

wsj_data = pd.read_csv('/Users/vibhor/Desktop/Models/TimeSeriesModels/wsj_mev.csv', index_col=0, parse_dates=True)
df_mean_mev = wsj_data[['mev_type','forecast_date','date','values']].groupby(['date','forecast_date','mev_type']).mean().unstack()
df_median_mev = wsj_data[['mev_type','forecast_date','date','values']].groupby(['date','forecast_date','mev_type']).median().unstack()
#wsj_data_wide = wsj_data.pivot(values='values', columns=['forecast_date'])

df_median_mev.sort_values(by=['date','forecast_date'],inplace=True)