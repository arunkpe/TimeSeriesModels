"""
This program creates and tests Vector AutoRegressive models on HPI, GDP, 30Yr Mortgage rates and Unemployment rate
Data starts from Jan-1981.
"""
#Import all the required libraries
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
import functools


#Initiate Fred Object
FredData = getFredData() #this is a class defined in transformFred.py


ac = FredData.available_codes().reset_index().set_index('Code')
print(ac.loc['CPI'])#printing sample variable location


dict_macrovars = dict()

list_macrovars = ['HPI', 'GDP', 'MORTGAGE30US', 'UE'] #names of macroeconomic variables (MEVs)
startDate = '1981-01-01'
endDate   = '2018-12-31'

#This for loop imports MEVs into macro_df dataframe and bounds them between the dates above
#Stationarity tests (kpss and adf) are run on MEVs and MEV data and test stats are stored in dict_macrovars dictionary
for var in list_macrovars:
    macro_df = FredData.fetch_data(var)
    macro_df =  macro_df[(macro_df.Date >= startDate) & (macro_df.Date<=endDate)]
    macro_df = macro_df.set_index('Date') #Date column now becomes the index
    kpss_stats = kpss(macro_df.iloc[:,0])
    adf_stats = adfuller(macro_df.iloc[:,0])

    dict_macrovars[var] = {'df':macro_df,'kpss_stats':kpss_stats, 'adf_stats':adf_stats}

#The below loop plots the MEV data
fig, ax = plt.subplots(nrows=int(len(list_macrovars)/2), ncols=int(len(list_macrovars)/2))
for row_index,row in enumerate(ax):
    for col_index,col in enumerate(row):
        col.plot(dict_macrovars[list_macrovars[(row_index*2)+(col_index)]]['df'])
        col.set_title(list_macrovars[(row_index*2)+(col_index)])

plt.show()

#The below loop informs which MEV is stationary
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

#The below loop transforms the MEV that is non-stationary (UE is the exception)...
for var in list_macrovars:
    if var in ('HPI','GDP'):
        dict_macrovars[f'{var}_logDiff'] = {'df':np.log(dict_macrovars[var]['df']).diff().dropna()}
    elif var in ('MORTGAGE30US','UE'):
        dict_macrovars[f'{var}_Diff'] = {'df':(dict_macrovars[var]['df']).diff().dropna()}
    else:
        pass

#...and this one runs the stationarity tests on the new transformed variables, dict_macrovars gets bigger
list_vars = ['HPI_logDiff', 'GDP_logDiff', 'MORTGAGE30US_Diff','UE_Diff']
for var in list_vars:
    kpss_stats = kpss(dict_macrovars[var]['df'].iloc[:,0])
    adf_stats = adfuller(dict_macrovars[var]['df'].iloc[:,0])
    dict_macrovars[var] = {'df': dict_macrovars[var]['df'], 'kpss_stats':kpss_stats, 'adf_stats':adf_stats}

#Transformed variables' stationaroty status is displayed
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

df_all = pd.DataFrame(columns=['MEV_Type', 'Value'], index=['Date'])#dataframe of MEVs is now constructed
df_all.dropna(inplace=True)

#This for loop populates the dataframe from the dictionary dict_macrovars
for mev_name, mev_dict in dict_macrovars.items():
    df = mev_dict['df'].copy()
    df['MEV_Type'] = mev_name
    df_all = df_all.append(df.rename(columns={df.columns[0]: "Value"}),sort=True)

df_wide = df_all.pivot(columns='MEV_Type', values='Value')#reshaping from long to wide format
#df_wide.index = pd.to_datetime(df_wide.index, errors='coerce')
mort_ser = df_wide.MORTGAGE30US.resample('1D').interpolate(method='linear') #linearly interpolate mortgage rates
#after resampling available data to daily frequency. This does the job of a weighted interpolation for NaNs

gdp_ser = df_wide['GDP']
df_wide = df_wide.loc[gdp_ser[~gdp_ser.isna()].index, :]#retain df_wide only where GDP is non NaN
mort_ser = mort_ser.loc[gdp_ser[~gdp_ser.isna()].index]#do the same for the interpolated mortgage rates
df_wide['MORTGAGE30US'] = mort_ser#push the rates into the dataframe

df_wide['MORTGAGE30US_Diff'] = df_wide['MORTGAGE30US'].diff()#take first difference of mortgage rates


var_model_list = list_vars #new list containing only stationary variable....not sure if this needed.list_vars wd suffice
#var_model_list.append('UE')
varModelDF = df_wide[var_model_list]#limit dataframe to only those stationary variables that will be use VAR model
varModelDF.dropna(inplace=True)#and drop na that came about due to diff

#This function plots the PAC, ACF
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

#ACF,PACF plotting function is called in a loop run on all model variables 
for var in list_vars:
    tsplot(varModelDF[var],var)
#Result: UE - 4 lags, Mortgage - 6!, GDP -2, HPI -3.

#tes-train split etc.
startDate_train = datetime.strptime('1981-07-01', '%Y-%m-%d') #data start date parsed to required format
endDate_train = datetime.strptime('2012-07-01', '%Y-%m-%d')#train end date parsed to required format

dates_list = varModelDF.index #contains all dates

train_dates = [date for date in dates_list if (date >= startDate_train) & (date <= endDate_train)]
#all dates btwn start and end train are selected
test_dates = list(set(dates_list) - set(train_dates)) #dates selected from train are removed from total & alloc to test


train_varModelDF = varModelDF.loc[train_dates]#form train dataframe using only train dates
test_varModelDF = varModelDF.loc[test_dates]#form test dataframe using only test dates

#Run Model on test-train
var_model_list = list_vars #repeated
#var_model_list.append('UE')
varModelDF = df_wide[var_model_list]
varModelDF.dropna(inplace=True)

model = VAR(train_varModelDF)
results = model.fit(ic='bic',maxlags =4)
results.plot()
results.plot_acorr()

plt.show()

results.summary() #for all variables, BIC ended up selecting L1 models

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

#lag_order = results.k_ar
lag_order = 1
results.forecast(test_varModelDF.values[-lag_order:], 5)

fcst = results.forecast(train_varModelDF.values[-lag_order:], 5)

################
#The idea here is to slide the train end date, build the model and check how the errors behave.
#Errors are compared across models for different lengths of forecasting timeframe

dict_varmodels = dict() #This dictionary will contain all info on models built on sliding train end date
var_model_list = list_vars

last_forecast_date = datetime.strptime('2018-01-01', '%Y-%m-%d')#No model is built beyond this time. Ensures min 4Q fcst
model_build_date = datetime.strptime('2012-07-01', '%Y-%m-%d')#This is the stopping time for train data.
modelID = 'A'

#Define MAE
def mae(ypred, ytrue):
    """ returns the mean absolute percentage error """
    return np.mean(np.abs(ypred-ytrue))

#This while loop slides the train data
while (model_build_date < last_forecast_date):
    train_dates = [date for date in dates_list if (date <= model_build_date)]
    train_data = varModelDF.loc[train_dates]
    actual_TestDates = list(set(dates_list) - set(train_dates))
    actual_TestDates.sort()
    actual_TestData = varModelDF.loc[actual_TestDates]#the ordering of data is getting screwed
    actual_TestData.reset_index(inplace=True)

    fcstErrPeriod = ['1Yr','2Yr','3Yr','4Yr','5Yr','6Yr','7Yr','8Yr']
    fcstErr = pd.DataFrame(0,  index=range(len(fcstErrPeriod)),columns = range(len(list_vars)))
    fcstErr.columns = list_vars

    forecast_period = len(dates_list) - len(train_dates)

    # Run Model on test-train
    train_varModelDF = train_data[var_model_list]
    train_varModelDF.dropna(inplace=True)

    model = VAR(train_varModelDF)
    results = model.fit(ic='bic', maxlags=4)
    lag_order = results.k_ar

    forecast = results.forecast(train_varModelDF.values[-lag_order:], forecast_period)

    fcstDF = pd.DataFrame(forecast)
    fcstDF.columns = list_vars
    fcstErrMod = np.mod(len(actual_TestData), 4)
    fcstErrQtrs = len(actual_TestData) - fcstErrMod
    if(fcstErrQtrs ==4):
        fcstQtrRange = range(4,fcstErrQtrs+4,4)
    else:
        fcstQtrRange = range(4,fcstErrQtrs+4,4)

    zeroList = [0] * (fcstErr.shape[0] - int(fcstErrQtrs / 4))

    for var in list_vars:
        err = []
        for Qtrs in fcstQtrRange:
            err.append(mae(fcstDF[var][0:Qtrs], actual_TestData[var][0:Qtrs]))
        err.extend(zeroList)
        fcstErr[var] = err

    modelID = chr(ord(modelID) + 1)
    dict_varmodels[modelID] = {'buildDate':model_build_date,'model':model,'results':results, 'fcst':forecast,'fcstErr':fcstErr}
    model_build_date = model_build_date + relativedelta(months=3)


firstPeriodError = pd.DataFrame()
secondPeriodError = pd.DataFrame()
thirdPeriodError = pd.DataFrame()
fourthPeriodError = pd.DataFrame()
fifthPeriodError = pd.DataFrame()
sixthPeriodError = pd.DataFrame()

#Extract the errors
for modelName,modelAttribs in dict_varmodels.items():
    fcstErr = modelAttribs['fcstErr'].copy()

    fcstVal = fcstErr.iloc[0]
    fcstVal.name = modelName
    firstPeriodError = firstPeriodError.append(fcstVal)

    fcstVal = fcstErr.iloc[1]
    fcstVal.name = modelName
    secondPeriodError = secondPeriodError.append(fcstVal)

    fcstVal = fcstErr.iloc[2]
    fcstVal.name = modelName
    thirdPeriodError = thirdPeriodError.append(fcstVal)

    fcstVal = fcstErr.iloc[3]
    fcstVal.name = modelName
    fourthPeriodError = fourthPeriodError.append(fcstVal)

    fcstVal = fcstErr.iloc[4]
    fcstVal.name = modelName
    fifthPeriodError = fifthPeriodError.append(fcstVal)

    fcstVal = fcstErr.iloc[5]
    fcstVal.name = modelName
    sixthPeriodError = sixthPeriodError.append(fcstVal)




errorDFs = [firstPeriodError,secondPeriodError,thirdPeriodError,fourthPeriodError,fifthPeriodError,sixthPeriodError]
for num, errorDF in enumerate(errorDFs, start=1):
    # print(errorDF.columns)
    errorDF.columns = [f'{col}_period{num}' for col in list_vars]

merge = functools.partial(pd.merge, left_index=True, right_index=True)
errorDFs = functools.reduce(merge, errorDFs)
print(errorDFs.head())


#FacetPlot

g = sns.FacetGrid(errorDFs, col="origin")
g.map(sns.distplot, "mpg")


sns.set(style="ticks")

# Create a dataset with many short random walks
rs = np.random.RandomState(4)
pos = rs.randint(-1, 2, (20, 5)).cumsum(axis=1)
pos -= pos[:, 0, np.newaxis]
step = np.tile(range(5), 20)
walk = np.repeat(range(20), 5)
df = pd.DataFrame(np.c_[pos.flat, step, walk],
                  columns=["position", "step", "walk"])

# Initialize a grid of plots with an Axes for each walk
grid = sns.FacetGrid(df, col="walk", hue="walk", palette="tab20c",
                     col_wrap=4, height=1.5)

# Draw a horizontal line to show the starting point
grid.map(plt.axhline, y=0, ls=":", c=".5")

# Draw a line plot to show the trajectory of each random walk
grid.map(plt.plot, "step", "position", marker="o")

# Adjust the tick positions and labels
grid.set(xticks=np.arange(5), yticks=[-3, 3],
         xlim=(-.5, 4.5), ylim=(-3.5, 3.5))

# Adjust the arrangement of the plots
grid.fig.tight_layout(w_pad=1)