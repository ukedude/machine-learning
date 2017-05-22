# -*- coding: utf-8 -*-
"""
Utilities for reading and pre-processing
"""
#from yahoo_finance import Share

import pandas as pd 

def compute_daily_returns(df):
	'''Compute and Return the daily return values'''
	# daily_return = (price[t] / price[t - 1]) - 1
	return (df/df.shift(1)) - 1

def compute_cumulative_returns(df):
	return (df/df[0]) - 1

def get_rolling_std(df, window=20):
	return pd.Series.rolling(df, center=False, window=window).std()

def get_rolling_mean(df, window=20):
	return pd.Series.rolling(df, center=False,window=window).mean()

def get_bollinger_bands(rm, rstd):
	upper_band = rm + (rstd * 2)
	lower_band = rm - (rstd * 2)
	return upper_band, lower_band
 
# If empty in the middle forward fill
# If empty at the beginning backfill
def fill_missing_values(df):
    # Forward fill... Then back fill
    df.fillna(method="ffill", inplace=True)
    df.fillna(method="bfill", inplace=True)
    
def get_data(symbols, start_date,end_date, drop_na=True):
    from pandas_datareader import data as dreader   
 
    dates = pd.date_range(start_date, end_date)
    df = pd.DataFrame(index=dates)

    if 'SPY' not in symbols:
        symbols.insert(0, 'SPY')

    for symbol in symbols:
#        shr = Share(symbol)
#        df_temp =  pd.DataFrame(shr.get_historical(start_date, end_date))
        df_temp = dreader.DataReader(symbol,'yahoo',start_date,end_date)
#        df_temp.drop(['Close','Open','High','Low','Symbol','Volume'], inplace=True, axis=1)
        df_temp.drop(['Close','Open','High','Low','Volume'], inplace=True, axis=1)
        df_temp['Date'] = pd.to_datetime(df_temp['Date'])
        df_temp = df_temp.set_index(['Date'])
#        df_temp = df_temp.rename(columns={'Adj_Close': symbol})
        df_temp = df_temp.rename(columns={'Adj Close': symbol})
        df_temp[symbol] = pd.to_numeric(df_temp[symbol])
        df = df.join(df_temp)
    if drop_na:
        df = df.dropna(subset=["SPY"])
        fill_missing_values(df)
    return df
    

    
def compute_features(df):
    
    stocks = df.columns
    stockdata = dict()
    for symbol in stocks:
        
        cdf = pd.DataFrame()
        cdf['Adj_Close'] = df[symbol]
        cdf['Daily_Return'] = compute_daily_returns(cdf['Adj_Close'])
        for window in (20,50,100,200):
            ma = 'MA('+str(window)+")"
            cdf[ma] = get_rolling_mean(cdf['Adj_Close'],window=window)
            ma_returns = ma + '_Return'
            cdf[ma_returns] = compute_daily_returns(cdf[ma])
            std = 'STD('+str(window)+")"
            cdf[std] = get_rolling_std(cdf['Adj_Close'],window=window)
            ub = 'UB('+str(window)+")"
            lb = 'LB('+str(window)+")"
            cdf[ub], cdf[lb] = get_bollinger_bands(cdf[ma], cdf[std])
    
        #cut off first 200 days
        cdf = cdf[200:]
        #print cdf.describe()
        #print cdf[0:10]
        stockdata[symbol] = cdf

    return stockdata
