import pandas as pd
import pandas_datareader
import datetime
import matplotlib.pylab as plt
import seaborn as sns
from matplotlib.pylab import style
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

style.use('ggplot')
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

stockFile = 'T10yr.csv'
stock = pd.read_csv(stockFile, index_col=0, parse_dates=[0])
# print(stock.head(10))
# print(stock.tail(10))

stock_week = stock['Close'].resample('W-MON').mean()
stock_train = stock_week['2000':'2015']

stock_train.plot(figsize=(12,8))
plt.legend(bbox_to_anchor=(1.25, 0.5))
plt.title("Stock Close")
sns.despine()
# plt.show()

stock_diff = stock_train.diff()
stock_diff = stock_diff.dropna()

plt.figure()
plt.plot(stock_diff)
plt.title('一阶差分')
plt.show()

acf = plot_acf(stock_diff, lags=20)
plt.title("ACF")
acf.show()

pacf = plot_pacf(stock_diff, lags=20)
plt.title("PACF")
pacf.show()

model = ARIMA(stock_train, order=(1, 1, 1),freq='W-MON')

result = model.fit()
print(result.summary())

pred = result.predict('20140609', '20180731',dynamic=True, typ='levels')
print (pred)

plt.figure(figsize=(6, 6))
plt.xticks(rotation=45)
plt.plot(pred)
plt.plot(stock_train)
plt.show()
