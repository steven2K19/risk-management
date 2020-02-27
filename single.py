import yfinance as yf
import numpy as np
import pandas as pd
from datetime import date, timedelta
from scipy.stats import kurtosis, skew, norm
import pandas_datareader as pdr
import statsmodels.formula.api as sm

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
divid['repurchase_price']= np.mean([divid.Close, divid.Open, divid.Low, divid.Close])
repurchase_share = (divid.Dividends/divid.repurchase_price).sum()
hpr = (1+repurchase_share)*ohlc.Close.iloc[-1]/ohlc.Close.iloc[0]-1
mu_day = np.mean(ret_day)
mu_month = np.mean(ret_month)
mu_year = np.mean(ret_year)

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
ret_dn = ret_day.loc[ret_day<mu_day]
std_dn = np.std(ret_dn)
std_dnm = std_dn*np.sqrt(21)
std_dny = std_dn*np.sqrt(252)

# monthly beta
def get_beta_month(sym,end):
    start = end -timedelta(days=365*10) - pd.tseries.offsets.BusinessDay(n=2)
    spy= yf.Ticker('SPY')
    spy= spy.history(start=start, end=end)
    tick= yf.Ticker(symbol)
    tick = tick.history(start=start, end=end)
    spy = spy['Close'].resample('BM').first()
    tick = tick['Close'].resample('BM').first()
    spy = spy.pct_change().dropna()
    spy_var = spy.var()
    tick = tick.pct_change().dropna()
    df = pd.concat([spy,tick], axis=1)
    df.columns = ['spy','tick']
    cov = df[['spy','tick']].cov()
    cov = cov.iloc[0,1]
    beta = cov/spy_var
    return beta
beta = get_beta_month(symbol, end)


cnf = 0.95
cna= 1-cnf
# Parametric VaR      VaR= share*price*VaR*np.sqrt(days)
mu = mu_day
sig = std_day       
kurt= exkurt + 3  
var_para = abs(norm.ppf(cna, mu, sig)) 

# History VaR
cnf = 0.95
cna = 1-cnf
ret2 = np.sort(ret_day)
n = np.size(ret2)
lefttail = int(n*cna)
var_hist = abs(ret2[lefttail])

# Modified VaR 
z = norm.ppf(cna)
mod = z+1/6.*(z**2-1)*skw+1/24.*(z**3-3*z)*kurt-1/36.*(2*z**3-5*z)*skw**2
var_mod= abs(mu + mod*sig)

# CVaR  average 5% loss 
var_con = abs(ret2[ret2<= ret2[int(n*cna)]].mean())

# Monte Carlos VaR
simu=1000
ret3=np.random.normal(mu,sig,simu)
ret4=np.sort(ret3)
var_mc=abs(ret4[int(simu*cna)])

# performance metrics 
rf = pdr.DataReader('TB3MS','fred',start,end)
rf = (rf['TB3MS'][-1]*0.01+1)**(1/365)-1
spy = pdr.DataReader('SPY', 'yahoo',start,end)
retspy = spy['Adj Close'].pct_change().dropna()
muspy = np.mean(retspy)
stdspy = np.std(retspy)

sharpe = (mu_day-rf)/std_day
treynor = (mu_day-rf)/beta
rovar = mu_day/var_para
sortino = (mu_day-rf)/std_dn  
jenalpha = mu_day - (rf + beta*(muspy-rf))  
msquare = (mu_day-rf)* stdspy/std_day - (muspy-rf)

# Fama French
def get_ff5(symbol,end):
    end = end
    start = end -timedelta(days=365*10) - pd.tseries.offsets.BusinessDay(n=2)
    stock = yf.Ticker(symbol)
    stock = stock.history(start=start, end=end)
    dss = stock.Close.pct_change()*100
    ds = pdr.DataReader('F-F_Research_Data_5_Factors_2x3_daily', 'famafrench',start=start, end=end)
    ds = ds[0]
    df = pd.concat([dss, ds], axis=1).dropna()
    df['excess'] = df['Close'] - df['RF']
    df['MKRF'] = df['Mkt-RF']
    ff5 = sm.ols(formula='excess~ MKRF+SMB+HML+RMW+CMA', data=df).fit()
    rsquare = ff5.rsquared_adj          # ff5.summary(), ff5.pvalues
    alpha = ff5.params['Intercept']
    mkrf = ff5.params['MKRF']
    smb = ff5.params['SMB']
    hml = ff5.params['HML']
    rmw = ff5.params['RMW']
    cma = ff5.params['CMA']
    metric = [rsquare, alpha, mkrf, smb, hml, rmw, cma]
    return metric

metric = get_ff5(symbol,end)
inf = [hpr, mu_day, mu_month, mu_year, std_day, std_month, std_year, skw, exkurt, mdd, std_dn, std_dnm, std_dny, beta, var_para, var_hist, var_mod, var_con, var_mc, sharpe, treynor, rovar, sortino, jenalpha, msquare*100]
inf= inf + metric
inf = [round(item,4) for item in inf]
inf.append(symbol)

names = [hpr, mu_day, mu_month, mu_year, std_day, std_month, std_year, skw, exkurt, mdd, std_dn, std_dnm, std_dny, beta, var_para, var_hist, var_mod, var_con, var_mc, sharpe, treynor, rovar, sortino, jenalpha, msquare, ff5rsquare, alpha, mkrf, smb, hml, rmw, cma]
