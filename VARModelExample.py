from transformFred import getFredData
from statsmodels.tsa.api import VAR
import pandas as pd
import numpy as np
import matplotlib as plt
import seaborn as sns
from fbprophet import Prophet


dataSet = getFredData()

hpi = dataSet.fetch_data('HPI')
ue  = dataSet.fetch_data('UR')
gdp = dataSet.fetch_data('GDP')

startDate = '1980-01-01'
endDate   = '2018-12-31'

hpi = hpi[(hpi['Date'] > startDate) & (hpi['Date'] < endDate)]
ue = ue[(ue['Date'] > startDate) & (ue['Date'] < endDate)]
gdp = gdp[(gdp['Date'] > startDate) & (gdp['Date'] < endDate)]

#Log transforms on GDP and HPI - why?!
hpi['USSTHPI'] = np.log(hpi['USSTHPI'])
gdp['GDP'] = np.log(gdp['GDP'])

#Does Mr Khar notice the subtle difference below?
varDataSet = pd.merge(hpi, ue, how='left', on='Date')
varDataSet = pd.merge(gdp, varDataSet, how='right', on='Date')

varDataSet.set_index('Date',inplace = True)
varDataSet = varDataSet.diff().dropna()

model = VAR(varDataSet)
results = model.fit(2)
results.plot()
results.plot_acorr()


#Forecast GDP using the Facebook Prophet Package
fbgdp = gdp
fbgdp.columns = ['ds','y']
m = Prophet(seasonality_mode='multiplicative').fit(fbgdp)
future = m.make_future_dataframe(periods=60,freq='Q')
fcst = m.predict(future)
fig = m.plot(fcst)
fig = m.plot_components(fcst)


m = Prophet(seasonality_mode='multiplicative', mcmc_samples=300).fit(fbgdp)
fcst = m.predict(future)
fig = m.plot_components(fcst)