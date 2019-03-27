'''
Created on 14 Mar 2019

@author: francescoferrari, Sebastian Hälg
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import arch
import os
from utils import __get_df_multi, __get_df_daily
from scipy import stats as st
from numpy import disp

# ------------------------------------------------------------------------------
# HELPERS
# ------------------------------------------------------------------------------

def __check_missing(ticker_):
    flag = not os.path.isfile('./db/{}.csv'.format(ticker_))
    return flag
pass

# ------------------------------------------------------------------------------
# CALCULATORS
# ------------------------------------------------------------------------------

def __calc_log_returns(df_):
    df_ret=np.log(df_.shift(1)/df_)
    return df_ret
pass

def __calc_simple_returns(df_):
    df_ret=df_.shift(1)/df_-1
    return df_ret
pass

def __calc_ann_hist_vol(df_,date_,lookback_=180):
    assert not np.nan in np.array(df_.loc[date_]), \
        "One of the stock was not listed as of {}".format(date_)
    idx_to=np.where((df_.index==date_))[0][0]
    idx_from=idx_to-lookback_
    df_ret=__calc_log_returns(df_.iloc[idx_from:idx_to])
    ann_vol=np.std(df_ret.iloc[1:])*np.sqrt(252)
    return ann_vol
pass

def __calc_ann_ewma_vol(df_,date_,lookback_=180,lambda_=0.94):
    # We choose 0.94 as default following RiskMetrics' approach
    assert not np.nan in np.array(df_.loc[date_]), \
        "One of the stock was not listed as of {}".format(date_)
    idx_to=np.where((df_.index==date_))[0][0]
    idx_from=idx_to-lookback_
    df_ret=__calc_log_returns(df_.iloc[idx_from:idx_to])
    df_ret_sqrd=np.power(df_ret,2.0)
    df_ret_sqrd=df_ret_sqrd.reindex(index=df_ret_sqrd.index[::-1])
    weights=[(1-lambda_)*np.power(lambda_,i) for i in range(len(df_ret_sqrd)-1)]
    ann_vol=np.sqrt(np.dot(weights,df_ret_sqrd.iloc[0:-1]))*np.sqrt(252)
    return ann_vol
pass

def __calc_ann_garch11_vol(df_,date_,lookback_=180):
    # Needs to be reviewed carefully
    assert not np.nan in np.array(df_.loc[date_]), \
        "One of the stock was not listed as of {}".format(date_)
    idx_to=np.where((df_.index==date_))[0][0]
    idx_from=idx_to-lookback_
    df_ret=__calc_log_returns(df_.iloc[idx_from:idx_to])
    if len(df_.shape)==1:
        ann_vol=np.sqrt(__get_garch11_forcast(df_ret.iloc[1:]))*np.sqrt(252)
    else:
        ann_vol=np.array([0]*df_.shape[1])
        for i in range(df_.shape[1]):
            ann_vol[i]=np.sqrt(__get_garch11_forcast(df_ret.iloc[1:,i]))*np.sqrt(252)
    return ann_vol
pass

def __calc_corr_mat(df_,date_,lookback_=180):
    assert not np.nan in np.array(df_.loc[date_]), \
        "One of the stock was not listed as of {}".format(date_)
    idx_to=np.where((df_.index==date_))[0][0]
    idx_from=idx_to-lookback_
    df_ret=__calc_log_returns(df_.iloc[idx_from:idx_to])
    return df_ret.iloc[1:].corr()
pass

def __calc_lending_value(tickers_,date_,holdings_,alpha_=0.25,prob_=0.01,
                         closeout_days_=10,lookback_=180,
                         method_=__calc_ann_hist_vol):
    if len(tickers_)==1:
        df=__get_single_df_field(tickers_[0],'close')
        ann_vol = method_(df,date_,lookback_)
        lending_value = (1-alpha_)*np.exp(ann_vol*np.sqrt(closeout_days_/252)* \
                        st.norm.ppf(prob_))/(1-alpha_*np.exp(ann_vol* \
                        np.sqrt(closeout_days_/252)*st.norm.ppf(prob_)))
        return lending_value
    df_sim=__simulate_prices(tickers_,date_,closeout_days_,lookback_,method_)
    S_tau=df_sim[0,0,:]
    V_tau=np.dot(np.array(holdings_),S_tau)
    V_tau_delta=np.dot(np.array(holdings_),np.transpose(df_sim[-1,:,:]))
    V_alpha_q=np.quantile(V_tau_delta,prob_)
    lending_value = (V_alpha_q*alpha_-V_alpha_q)/(V_alpha_q*alpha_-V_tau)
    return lending_value
pass

# ------------------------------------------------------------------------------
# GETTERS
# ------------------------------------------------------------------------------

def __get_missing(tickers_):
    flags = [__check_missing(ticker) for ticker in tickers_]
    __get_df_multi(__get_df_daily,np.array(tickers_)[flags],'full',True)   
pass

def __get_df_from_to(ticker_,from_=None,to_=None):
    if __check_missing(ticker_):
        __get_df_daily(ticker_,'full',store_=True)
    df=pd.read_csv('db/{}.csv'.format(ticker_),parse_dates=['date'])
    df=df.set_index('date')
    if (from_==None and to_==None):
        return df
    elif from_==None:
        return df.loc[0:to_]
    else:
        return df.loc[from_:to_]
pass

def __get_single_df_field(ticker_,field_):
    df=__get_df_from_to(ticker_)
    return df[field_]
pass

def __get_multiple_df_field(tickers_,field_,from_=None,to_=None):
    df_all=__get_single_df_field(tickers_[0],field_)
    for ticker in tickers_[1:]:
        df_all=pd.concat([df_all,__get_single_df_field(ticker,field_)],axis=1)
    df_all.columns=tickers_
    if (from_==None and to_==None):
        return df_all
    elif from_==None:
        return df_all.loc[0:to_]
    else:
        return df_all.loc[from_:to_]
pass

def __get_garch11_forcast(df_ret_):
    model=arch.arch_model(df_ret_.iloc[1:],p=1,q=1)
    model_fit=model.fit(update_freq=10)
    daily_var_forcast=model_fit.forecast(horizon=1).variance.values[-1]
    return daily_var_forcast
pass

# ------------------------------------------------------------------------------
# SIMULATORS
# ------------------------------------------------------------------------------

def __simulate_prices(tickers_,date_,n_days_,lookback_,method_,n_sim_=10000):
    dt=1/252
    df=__get_multiple_df_field(tickers_,'close')
    S_tau=np.diag(df.loc[date_])
    ann_vol=method_(df,date_,lookback_)
    corr_mat=__calc_corr_mat(df,date_,lookback_)
    chol=np.transpose(np.linalg.cholesky(corr_mat))
    mu=np.power(ann_vol,2.0)/2
    nu=mu-np.power(ann_vol,2.0)/2
    n_assets=len(tickers)
    df_sim=np.zeros(shape=(n_days_+1,n_sim_,n_assets))
    for i in range(n_sim_):
        rand_normal_seq=np.random.normal(size=(n_days_,
                                               np.shape(corr_mat)[0]))
        corr_seq=np.dot(rand_normal_seq,chol)
        df_sim[:,i,:] = np.dot(np.r_[np.ones(shape=(1,n_assets)),
                                     np.cumprod(np.exp(np.tile(np.array(nu*dt),
                                                               (n_days_,1))+np.dot(corr_seq,
                                                                                   np.diag(ann_vol))*np.sqrt(dt)),
                                                               axis=1)],
                               S_tau)    
    return df_sim
pass

# ------------------------------------------------------------------------------
# MAIN
# ------------------------------------------------------------------------------

tickers = ['AAPL','FB']
holdings=np.array([1,1])
date = pd.to_datetime('2017-03-13')
# lending_value=__calc_lending_value(tickers,date,holdings)
lending_value=0.85
alpha=0.25
beta=(1-(1-lending_value)*alpha)

df=__get_multiple_df_field(tickers,'close',date)
if len(holdings)==1:
    V=df*holdings
else:
    V=pd.Series(np.dot(holdings,np.transpose(df)),index=df.index)
X=pd.Series(np.zeros(len(V)),index=df.index)
X.iloc[0]=V.iloc[0]*lending_value
margin_call=pd.Series(np.zeros(len(V),dtype=bool),index=df.index)
is_closeout=pd.Series(np.zeros(len(V),dtype=bool),index=df.index)
tau=[0]
eta_odd=[0]
a =df.index.to_series() 
b =a[1].month
for i in range(len(V)-1):
#    X.iloc[i+1]=lending_value*np.max(V.iloc[tau[-1]:i+2])
    if abs(a[i+1].month-a[i].month)==1: #check if month changes
#    if abs(a[i+1].month-b)==2:    
        X.iloc[i+1]=lending_value*np.max(V.iloc[tau[-1]:i+2])
#        b = a[i+1].month
    else:
        X.iloc[i+1]=X[i]
        
    if X.iloc[i+1]/V.iloc[i+1]>lending_value/beta:
        is_closeout.iloc[i+1]=True
        if is_closeout.iloc[i]==False or i==tau[-1]:
            eta_odd.append(i+1)
            margin_call.iloc[i+1]=True
    if not False in list(is_closeout.iloc[eta_odd[-1]:i+1]) and len(is_closeout.iloc[eta_odd[-1]:i+1])==10:
        tau.append(i+1)
        X.iloc[i+1]=lending_value*V.iloc[tau[-1]]

plt.plot(V.index,np.array(V),lw=0.5)
plt.plot(V.index,np.array(X),lw=0.5)
margin_call_dates=margin_call.index[eta_odd[1:]]
critical_dates=margin_call.index[tau[1:]]
for margin_call_date in margin_call_dates:
    plt.axvline(margin_call_date, color='k', linestyle='-',lw=0.2)
for critical_date in critical_dates:
    plt.axvline(critical_date, color='r', linestyle='-',lw=0.2)
print(lending_value)
