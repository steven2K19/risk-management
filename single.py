import pandas_datareader as pdr
import numpy as np
from datetime import date, timedelta 
from scipy.stats import kurtosis, skew
from scipy import stats

def get_beta(tickers):
    df = pdr.DataReader(tickers, 'yahoo',start,end)
    df = df ['Adj Close'].resample('BMS').first()
    df['ret'] = df['MSFT'].pct_change()
    df['retx'] = df['SPY'].pct_change()
    df = df.dropna()
    cov = df[['ret','retx']].cov()
    cov = cov.iloc[0,1]
    spy_var = df['retx'].var()
    beta = cov/spy_var
    return beta

symbol = "MSFT"
today = date.today()      
start = today-timedelta(days=365)  
end = today
ohlc = pdr.DataReader(symbol, 'yahoo',start,end) 
ret = ohlc['Adj Close'].pct_change().dropna()

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
beta=get_beta(sym)
