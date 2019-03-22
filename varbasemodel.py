from TimeSeriesModels.transformFred import getFredData
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import kpss
from statsmodels.tsa.stattools import adfuller
import numpy as np
import pandas as pd
from statsmodels.tsa.api import VAR
from datetime import datetime
from dateutil.relativedelta import relativedelta
import statsmodels.api as smodels
from sklearn.externals import joblib


#Initiate Fred Object
FredData = getFredData()


ac = FredData.available_codes().reset_index().set_index('Code')
print(ac.loc['CPI'])#printing sample variable location


dict_macrovars = dict()

list_macrovars = ['HPI', 'GDP', 'MORTGAGE30US', 'UE']
startDate = '1981-01-01'
endDate   = '2018-12-31'
for var in list_macrovars:
    macro_df = FredData.fetch_data(var)
    macro_df =  macro_df[(macro_df.Date >= startDate) & (macro_df.Date<=endDate)]
    macro_df = macro_df.set_index('Date')
    kpss_stats = kpss(macro_df.iloc[:,0])
    adf_stats = adfuller(macro_df.iloc[:,0])

    dict_macrovars[var] = {'df':macro_df,'kpss_stats':kpss_stats, 'adf_stats':adf_stats}

fig, ax = plt.subplots(nrows=int(len(list_macrovars)/2), ncols=int(len(list_macrovars)/2))
for row_index,row in enumerate(ax):
    for col_index,col in enumerate(row):
        col.plot(dict_macrovars[list_macrovars[(row_index*2)+(col_index)]]['df'])
        col.set_title(list_macrovars[(row_index*2)+(col_index)])

plt.show()

for var in list_macrovars:
    significance_threshold_kpss = dict_macrovars[var]['kpss_stats'][3]['1%']
    significance_threshold_adf  = dict_macrovars[var]['adf_stats'][4]['1%']
    pval_kpss = dict_macrovars[var]['kpss_stats'][1]
    pval_adf = dict_macrovars[var]['adf_stats'][1]

    kpss_stationarity = 0
    adf_stationarity = 0
    if pval_kpss < significance_threshold_kpss:
        kpss_stationarity = 1
    if pval_adf > significance_threshold_adf:
        adf_stationarity = 1

    print("Var:",var,"KPSS Stationarity",kpss_stationarity,"ADF Stationarity",adf_stationarity,pval_kpss,pval_adf)


for var in list_macrovars:
    if var in ('HPI','GDP'):
        dict_macrovars[f'{var}_logDiff'] = {'df':np.log(dict_macrovars[var]['df']).diff().dropna()}
    elif var in ('MORTGAGE30US','UE'):
        dict_macrovars[f'{var}_Diff'] = {'df':(dict_macrovars[var]['df']).diff().dropna()}
    else:
        pass

list_vars = ['HPI_logDiff', 'GDP_logDiff', 'MORTGAGE30US_Diff','UE_Diff']
for var in list_vars:
    kpss_stats = kpss(dict_macrovars[var]['df'].iloc[:,0])
    adf_stats = adfuller(dict_macrovars[var]['df'].iloc[:,0])
    dict_macrovars[var] = {'df': dict_macrovars[var]['df'], 'kpss_stats':kpss_stats, 'adf_stats':adf_stats}


for var in list_vars:
    significance_threshold_kpss = dict_macrovars[var]['kpss_stats'][3]['1%']
    significance_threshold_adf  = dict_macrovars[var]['adf_stats'][4]['1%']
    pval_kpss = dict_macrovars[var]['kpss_stats'][1]
    pval_adf = dict_macrovars[var]['adf_stats'][1]

    kpss_stationarity = 0
    adf_stationarity = 0
    if pval_kpss < significance_threshold_kpss:
        kpss_stationarity = 1
    if pval_adf > significance_threshold_adf:
        adf_stationarity = 1

    print("Var:",var,"KPSS Stationarity",kpss_stationarity,"ADF Stationarity",adf_stationarity,pval_kpss,pval_adf)

df_all = pd.DataFrame(columns=['MEV_Type', 'Value'], index=['Date'])
df_all.dropna(inplace=True)

for mev_name, mev_dict in dict_macrovars.items():
    df = mev_dict['df'].copy()
    df['MEV_Type'] = mev_name
    df_all = df_all.append(df.rename(columns={df.columns[0]: "Value"}),sort=True)

df_wide = df_all.pivot(columns='MEV_Type', values='Value')
df_wide.index = pd.to_datetime(df_wide.index, errors='coerce')
mort_ser = df_wide.MORTGAGE30US.resample('1D').interpolate(method='linear')

gdp_ser = df_wide['GDP']
df_wide = df_wide.loc[gdp_ser[~gdp_ser.isna()].index, :]
mort_ser = mort_ser.loc[gdp_ser[~gdp_ser.isna()].index]
df_wide['MORTGAGE30US'] = mort_ser

df_wide['MORTGAGE30US_Diff'] = df_wide['MORTGAGE30US'].diff()


var_model_list = list_vars
#var_model_list.append('UE')
varModelDF = df_wide[var_model_list]
varModelDF.dropna(inplace=True)

def tsplot(y, x, lags=None, figsize=(10, 8)):
    fig = plt.figure(figsize=figsize)
    layout = (2, 2)
    ts_ax = plt.subplot2grid(layout, (0, 0), colspan=2)
    acf_ax = plt.subplot2grid(layout, (1, 0))
    pacf_ax = plt.subplot2grid(layout, (1, 1))
    ts_ax.set_title(x)

    y.plot(ax=ts_ax)
    smodels.graphics.tsa.plot_acf(y, alpha = 0.05, use_vlines= True,lags=10, ax=acf_ax)
    smodels.graphics.tsa.plot_pacf(y, alpha = 0.05, use_vlines= True, lags=10, ax=pacf_ax)
    #[ax.set_xlim(0.5) for ax in [acf_ax, pacf_ax]]
    sns.despine()
    plt.tight_layout()
    return ts_ax, acf_ax, pacf_ax

for var in list_vars:
    tsplot(varModelDF[var],var)


#tes-train split etc.
startDate_train = datetime.strptime('1981-07-01', '%Y-%m-%d')
endDate_train = datetime.strptime('2012-07-01', '%Y-%m-%d')

dates_list = varModelDF.index

train_dates = [date for date in dates_list if (date >= startDate_train) & (date <= endDate_train)]
test_dates = list(set(dates_list) - set(train_dates))


train_varModelDF = varModelDF.loc[train_dates]
test_varModelDF = varModelDF.loc[test_dates]

#Run Model on test-train
var_model_list = list_vars
#var_model_list.append('UE')
varModelDF = df_wide[var_model_list]
varModelDF.dropna(inplace=True)

model = VAR(train_varModelDF)
results = model.fit(ic='bic',maxlags =4)
results.plot()
results.plot_acorr()

plt.show()

results.summary()

# save the model to disk
#filename = '/Users/vibhor/Desktop/Models/lag4model.sav'
#joblib.dump(model, filename)

# load the model from disk
#loaded_model = joblib.load(filename)
#result = loaded_model.score(X_test, Y_test)
#print(result)

#test performance
startDt = test_dates[0]
endDt = test_dates[-1]
results.predict(train_varModelDF,startDate,endDt)

lag_order = results.k_ar
results.forecast(test_varModelDF.values[-lag_order:], 5)



################
dict_varmodels = dict()
var_model_list = list_vars

last_forecast_date = datetime.strptime('2018-01-01', '%Y-%m-%d')
model_build_date = datetime.strptime('2012-07-01', '%Y-%m-%d')

while (model_build_date <= last_forecast_date):
    train_dates = [date for date in dates_list if (date < model_build_date)]
    train_data = varModelDF.loc[train_dates]
    # Run Model on test-train
    varModelDF = df_wide[var_model_list]
    varModelDF.dropna(inplace=True)

    model = VAR(train_varModelDF)
    results = model.fit(ic='bic', maxlags=4)

    for fcst_period in range(1,4):

    forecast = results.predict(train_varModelDF, startDate, endDt)

    dict_varmodels[model_build_date] = {'model':model,'results':results, 'adf_stats':adf_stats}
    model_build_date = model_build_date + relativedelta(months=3)