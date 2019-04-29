'''
Created on 14 Mar 2019

@author: francescoferrari
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import tickers
import arch
import os
from utils import __get_df_multi, __get_df_daily, __get_df_intraday
from scipy import stats as st
from pandas.plotting import register_matplotlib_converters
from numpy import logical_and

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

def __calc_adtv(df_):
    adtv=np.mean(df_.volume)
    return adtv
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
    # Needs to be reviewed carefully
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
    assert not np.nan in np.array(df_.iloc[idx_from]), \
        "One of the stock was not listed as of {}".format(df_.index[idx_from])
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
    lending_value = ((1-alpha_)*V_alpha_q)/(V_tau-V_alpha_q*alpha_)
    return lending_value
pass

def __calc_liquidity_adjustment(date_,df_=None,ticker_=None):
    if df_ is None:
        df_=__get_df_intraday(ticker_,'full','1min')
    idxs= np.logical_and(df.index>=date_ ,
                         df.index<np.datetime64(date_)+np.timedelta64(1, 'D'))
    df_req = df.iloc[idxs]
    close = np.array(df_req.close)
    volume = np.array(df_req.volume)
    for i in range(len(close)-1):
        if close[i]>close[i+1]:
            volume[i+1]=volume[i+1]*(-1)
        elif close[i]<close[i+1]:
            pass
        elif close[i]==close[i+1]:
            volume[i+1]=volume[i+1]*np.sign(volume[i])
    time_diff = (df_req.index[2:]-df_req.index[1:-1]).seconds/31557600
    log_ret = np.log(close[2:]/close[1:-1])
    y = np.array(log_ret/np.sqrt(time_diff))
    w = np.array((volume[2:]-volume[1:-1])/np.sqrt(time_diff))
    z = np.array(np.sqrt(time_diff))
    x = np.column_stack((w,z))
    results = smf.OLS(y,x).fit()
#     print(results.summary())
    gamma = results.params[0]
    p_value = results.pvalues[0]
    return gamma, p_value
pass

def __calc_liquidity_adjustment_est(df_, date_, lookback_=21):
    idx_to=np.where((df_.index==date_))[0][0]
    idx_from=idx_to-lookback_
    adtv=__calc_adtv(df_.iloc[idx_from:idx_to])
    gamma_hat = np.power(10.0,-1.87096)*np.power(adtv,-0.794554)
    return gamma_hat
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
    n_assets=len(tickers_)
    df_sim=np.zeros(shape=(n_days_+1,n_sim_,n_assets))
    for i in range(n_sim_):
        rand_normal_seq=np.random.normal(size=(n_days_,
                                               np.shape(corr_mat)[0]))
        corr_seq=np.dot(rand_normal_seq,chol)
        arg_exp=np.tile(np.array(nu*dt),(n_days_,1))+np.dot(corr_seq,np.diag(ann_vol))*np.sqrt(dt)
        cum_prod=np.cumprod(np.exp(arg_exp),axis=0)
        df_sim[:,i,:] = np.dot(np.concatenate((np.array([np.ones(n_assets)]),cum_prod),axis=0),S_tau)
    return df_sim
pass

# ------------------------------------------------------------------------------
# MAIN
# ------------------------------------------------------------------------------

# ticker='UBSG'
# df = pd.read_csv('db/{}.csv'.format(ticker),parse_dates=['date'])
# df=df.set_index('date')
# dates=np.unique(df.index.strftime('%Y-%m-%d'))
# results = [__calc_liquidity_adjustment(date, df) for date in dates]
# gammas = [result[0] for result in results]
# pvalues = [result[1] for result in results]
# gammasMA30 = [];
# for i in range(30, 63):
#     gammasMA30.append(np.mean(gammas[i-30:i]))
# gammasMA10 = [];
# for i in range(10, 63):
#     gammasMA10.append(np.mean(gammas[i-10:i]))
# plt.plot(pd.to_datetime(dates),gammas)
# plt.plot(pd.to_datetime(dates[30:63]),np.array(gammasMA30))
# plt.plot(pd.to_datetime(dates[10:63]),np.array(gammasMA10))
# plt.xlabel('date')
# plt.ylabel(r'$\gamma$')
# plt.title(ticker)
# plt.legend([r'$\gamma$ time series','MA(30)','MA(10)'])
# plt.xticks(rotation='vertical')
# plt.subplots_adjust(bottom=0.25)
# print(np.mean(gammas))
# print(np.sum(np.array(pvalues)<0.05)/len(pvalues))
# print(np.sum(np.array(gammas)>0)/len(pvalues))
# print(np.sum(df.volume)/63)

# ------------------------------------------------------------------------------

# gamma,_ =__calc_liquidity_adjustment('2019-04-16',ticker_='UBSG.SWI')
# df=__get_df_from_to('UBSG.SWI')
# gamma_hat=__calc_liquidity_adjustment_est(df, '2019-04-16')

# ------------------------------------------------------------------------------

# tickers = [ticker[0] for ticker in tickers.smi]
# df = __get_multiple_df_field(tickers,'close')
# vola = __calc_ann_hist_vol(df,'2018-04-16',lookback_=180).sort_values(ascending=False)
# tickers = np.array(vola.index)
# 
# tickers_list = [tickers[0:i+1] for i in range(len(tickers))]
# holdings_list = [np.ones(tickers.shape) for tickers in tickers_list]
# date = pd.to_datetime('2018-04-16')
# lending_value_list = [__calc_lending_value(list(tickers_),date,holdings_) for tickers_,holdings_ in zip(tickers_list,holdings_list)]

# ------------------------------------------------------------------------------

# tickers=tickers_list[-1]
# holdings=holdings_list[-1]
# lending_value=lending_value_list[-1]
# beta=(1-(1-lending_value)*0.25)
# df=__get_multiple_df_field(tickers,'close',date)
# if len(holdings)==1:
#     V=df*holdings
# else:
#     V=pd.Series(np.dot(holdings,np.transpose(df)),index=df.index)
# X=pd.Series(np.zeros(len(V)),index=df.index)
# X.iloc[0]=V.iloc[0]*lending_value
# margin_call=pd.Series(np.zeros(len(V),dtype=bool),index=df.index)
# is_closeout=pd.Series(np.zeros(len(V),dtype=bool),index=df.index)
# tau=[0]
# eta_odd=[0]
# for i in range(len(V)-1):
#     X.iloc[i+1]=lending_value*np.max(V.iloc[tau[-1]:i+2])
#     if X.iloc[i+1]/V.iloc[i+1]>lending_value/beta:
#         is_closeout.iloc[i+1]=True
#         if is_closeout.iloc[i]==False or i==tau[-1]:
#             eta_odd.append(i+1)
#             margin_call.iloc[i+1]=True
#     if not False in list(is_closeout.iloc[eta_odd[-1]:i+1]) and len(is_closeout.iloc[eta_odd[-1]:i+1])==10:
#         tau.append(i+1)
#         X.iloc[i+1]=lending_value*V.iloc[tau[-1]]
# 
# register_matplotlib_converters()
# plt.plot(V.index,np.array(V),lw=0.5)
# plt.plot(V.index,np.array(X),lw=0.5)
# margin_call_dates=margin_call.index[eta_odd[1:]]
# critical_dates=margin_call.index[tau[1:]]
# for margin_call_date in margin_call_dates:
#     plt.axvline(margin_call_date, color='k', linestyle='-',lw=0.5)
# for critical_date in critical_dates:
#     plt.axvline(critical_date, color='r', linestyle='-',lw=0.5)
# plt.xlabel('date')
# plt.ylabel(r'$V_t$')
# plt.title(r'SMI Stocks - $\lambda={}$'.format(lending_value))
# plt.legend([r'$V_t$',r'$X_t$',r'$\eta$'])

# ------------------------------------------------------------------------------

df_vsmi = __get_df_from_to('VSMI')
dates_vsmi = df_vsmi.index
idxs = np.logical_and(dates_vsmi>='2001-04-16',dates_vsmi<='2019-04-16')
dates_vsmi = dates_vsmi[idxs]
close_vsmi = df_vsmi.close[idxs]
df_ubs = __get_df_from_to('UBSG.SWI')
dates = df_ubs.index
dates = dates[np.logical_and(dates>='2001-04-16',dates<='2019-04-16')]

lending_value_hv = [__calc_lending_value(['UBSG.SWI'],date,[1],method_=__calc_ann_hist_vol) for date in dates]
lending_value_ewma = [__calc_lending_value(['UBSG.SWI'],date,[1],method_=__calc_ann_ewma_vol) for date in dates]
lending_value_garch = [__calc_lending_value(['UBSG.SWI'],date,[1],method_=__calc_ann_garch11_vol) for date in dates]

fig, ax1 = plt.subplots()
color = 'tab:red'
ax1.set_xlabel('date')
ax1.set_ylabel(r'$VSMI_t$', color=color)
ax1.plot(dates_vsmi, close_vsmi, color=color, lw=0.5)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()
# ax2.plot(dates,lending_value_hv, lw=0.5, color='tab:blue')
ax2.plot(dates,lending_value_ewma, lw=0.5,color='tab:green')
# ax2.plot(dates,lending_value_garch, lw=0.5,color='tab:blue')
ax2.set_ylabel(r'$\lambda_t$', color='tab:green')
ax2.tick_params(axis='y', labelcolor='tab:green')
plt.title('UBSG - EWMA volatility')

fig.tight_layout()

plt.plot(df_ubs.close)


