import yfinance as yf
import numpy as np
from datetime import date
from scipy.stats import kurtosis, skew
from scipy import stats

symbol = "MSFT"    
start = date(2010,1,1)
end = date(2020,1,1)
stock = yf.Ticker(symbol)
ohlc = stock.history(start=start, end=end)
ret = ohlc['Close'].pct_change().dropna()

mu = np.mean(ret)
sig = np.std(ret)
skw = skew(ret)
exkurt = kurtosis(ret)        
kurt= exkurt + 3   
sigm = sig*np.sqrt(21)
siga = sig*np.sqrt(252)
norm_pvalue = stats.shapiro(ret)[1]
ret_d = ret.loc[ret<mu]
sigd = np.std(ret_d)
sigdm = sigd*np.sqrt(21)
sigda = sigd*np.sqrt(252)
rollmax = ohlc.Close.rolling(window=252, min_periods=1).max()
rollmin = ohlc.Close.rolling(window=252, min_periods=1).min()
mdd = (rollmin - rollmax)/rollmax
mdd = mdd.drop_duplicates(keep='last').min()
