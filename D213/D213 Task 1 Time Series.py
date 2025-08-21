import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy.signal import periodogram
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA

# connect to the csv file and convert to dataframe
data_train = pd.read_csv('C:\\Users\\LorraineFerrusi\\OneDrive - The Information Lab\\Documents\\Personal\\WGU\\D213\\Task 1\\time_series_train.csv')
data_test = pd.read_csv('C:\\Users\\LorraineFerrusi\\OneDrive - The Information Lab\\Documents\\Personal\\WGU\\D213\\Task 1\\time_series_test.csv')

data_prepped = pd.concat([data_train, data_test], axis=0)
data_prepped = data_prepped.drop(data_prepped.columns[0], axis=1)

# plot the autocorrelation function (ACF)
fig, ax = plt.subplots(figsize=(10, 6))
sm.graphics.tsa.plot_acf(data_prepped['Revenue'], lags=40, ax=ax)
plt.title('Autocorrelation Function (ACF)')
plt.xlabel('Lags')
plt.ylabel('Autocorrelation')
plt.grid(True)
plt.show()

# plot the spectral density
frequencies, power = periodogram(data_prepped['Revenue'])
plt.figure(figsize=(10, 6))
plt.semilogy(frequencies, power)
plt.title('Spectral Density')
plt.xlabel('Frequency')
plt.ylabel('Power')
plt.grid(True)
plt.show()

# decompose the time series
decomposition = seasonal_decompose(data_prepped['Revenue'], model='additive', period=30)

# plot the decomposed components
plt.figure(figsize=(12, 8))
plt.subplot(411)
plt.plot(data_prepped['Revenue'], label='Original')
plt.legend(loc='upper left')
plt.subplot(412)
plt.plot(decomposition.trend, label='Trend')
plt.legend(loc='upper left')
plt.subplot(413)
plt.plot(decomposition.seasonal, label='Seasonal')
plt.legend(loc='upper left')
plt.subplot(414)
plt.plot(decomposition.resid, label='Residual')
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()

# identify the best ARIMA model
model = ARIMA(data_train['Revenue'], order=(5,1,0), seasonal_order=(1,1,1,30))
model_fit = model.fit()

# print the summary of the model
print(model_fit.summary())

# perform forecast
forecast_steps = len(data_test)
forecast = model_fit.forecast(steps=forecast_steps)

# plot the forecast
plt.figure(figsize=(10, 6))
plt.plot(data_train['Day'], data_train['Revenue'], label='Train')
plt.plot(data_test['Day'], data_test['Revenue'], label='Test')
plt.plot(data_test['Day'], forecast, label='Forecast')
plt.title('Revenue Forecast')
plt.xlabel('Day')
plt.ylabel('Revenue')
plt.legend()
plt.grid(True)
plt.show()

# create a table with the forecasted values
forecast_df = pd.DataFrame({'Day': test['Day'], 'Actual Revenue': test['Revenue'], 'Forecasted Revenue': forecast})
forecast_df.head(20)

# plot original data and forecasted data
plt.figure(figsize=(12, 6))

# plot original data
plt.plot(data_prepped.index, data_prepped['Revenue'], label='Train Data', color='blue')

# plot forecasted data
plt.plot(forecast_df.index, forecast_df['Forecasted Revenue'] , label='Forecasted Data', color='red')

# add labels and title
plt.xlabel('Day')
plt.ylabel('Revenue')
plt.title('Original Data vs Forecasted Data')
plt.legend()

# show plot
plt.show()
