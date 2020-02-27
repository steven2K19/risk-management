import yfinance as yf
import numpy as np
import pandas as pd
from datetime import date, timedelta
from scipy.stats import kurtosis, skew

symbol = "MSFT"    
start = date(2010,1,1)
end = date(2020,1,1)

# load data
stock = yf.Ticker(symbol)
ohlc = stock.history(start=start, end=end)

# return daily, monthly, annually, holding period return
ret_day = ohlc['Close'].pct_change().dropna()
ret_month = ohlc['Close'].asfreq('M',method='ffill').pct_change().dropna()
ret_year = ohlc['Close'].asfreq('A',method='ffill').pct_change().dropna()
divid = ohlc[ohlc.Dividends>0]
divid['repurchase_price']= (divid.Close + divid.Open + divid.Low + divid.Close)/4
repurchase_share = (divid.Dividends/divid.repurchase_price).sum()
hpr = (1+repurchase_share)*ohlc.Close.iloc[-1]/ohlc.Close.iloc[0]-1

# standard deviation daily, monthly, annualy; skew, kurtosis 
std_day = np.std(ret_day)
std_month= np.std(ret_month)
std_year = std_month*np.sqrt(12)
skw = skew(ret_month)
exkurt = kurtosis(ret_month)

# max drawdown 252 days
rollmax = ohlc.Close.rolling(window=252, min_periods=1).max()
rollmin = ohlc.Close.rolling(window=252, min_periods=1).min()
mdd = (rollmin - rollmax)/rollmax
mdd = mdd.drop_duplicates(keep='last').min()

# downside standard deviation  std_dn
mu = np.mean(ret_day)
ret_dn = ret_day.loc[ret_day<mu]
std_dn = np.std(ret_dn)
std_dnm = std_dn*np.sqrt(21)
std_dny = std_dn*np.sqrt(252)

# 5 year monthly beta
def get_beta(end, tick):
    start = end -timedelta(days=365*5) - pd.tseries.offsets.BusinessDay(n=2)
    spy= yf.Ticker('SPY')
    spy= spy.history(start=start, end=end)
    tick= yf.Ticker(tick)
    tick = tick.history(start=start, end=end)
    df = df ['Close'].resample('BMS').first()
    df['ret'] = df['MSFT'].pct_change()
    df['retx'] = df['SPY'].pct_change()
    df = df.dropna()
    cov = df[['ret','retx']].cov()
    cov = cov.iloc[0,1]
    spy_var = df['retx'].var()
    beta = cov/spy_var
    return beta
