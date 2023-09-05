import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
import plotly.graph_objects as go
from scipy import stats
from datetime import datetime, time as datetime_time
import time

from mercury_Bot import send_message
from mercury_Bot import send_html



def fetch_data(ticker, time_interval):
    end_date = datetime.today().date()
    start_date = end_date - timedelta(days=50)
    data = yf.download(ticker, start=start_date, end=end_date, interval=time_interval)
    return data

def prepare_data(df):
    df = pd.DataFrame(df)
    date_values = df.index.date
    df['Date'] = date_values
    new_columns = ['Date', 'Open', 'High', 'Low', 'Close']
    df = df[new_columns]
    df.reset_index(drop=True, inplace=True)
    return df

def isPivot(candle, window, df):
    if candle-window < 0 or candle+window >= len(df):
        return 0

    pivotHigh = 1
    pivotLow = 2
    for i in range(candle-window, candle+window+1):
        if df.iloc[candle].Low > df.iloc[i].Low:
            pivotLow=0
        if df.iloc[candle].High < df.iloc[i].High:
            pivotHigh=0
    if (pivotHigh and pivotLow):
        return 3
    elif pivotHigh:
        return pivotHigh
    elif pivotLow:
        return pivotLow
    else:
        return 0

def pointpos(x):
    if x['isPivot']==2:
        return x['Low']-1e-3
    elif x['isPivot']==1:
        return x['High']+1e-3
    else:
        return np.nan

def plot_candlestick(dfpl):
    fig = go.Figure(data=[go.Candlestick(x=dfpl.index,
                    open=dfpl['Open'],
                    high=dfpl['High'],
                    low=dfpl['Low'],
                    close=dfpl['Close'])])

    fig.add_scatter(x=dfpl.index, y=dfpl['pointpos'], mode="markers",
                    marker=dict(size=5, color="MediumPurple"),
                    name="pivot")
    #fig.update_layout(xaxis_rangeslider_visible=False)
    fig.show()

def collect_channel(candle, backcandles, window, df):
    localdf = df[candle-backcandles-window:candle-window]
    localdf['isPivot'] = localdf.apply(lambda x: isPivot(x.name,window,df), axis=1)
    highs = localdf[localdf['isPivot']==1].High.values
    idxhighs = localdf[localdf['isPivot']==1].High.index
    lows = localdf[localdf['isPivot']==2].Low.values
    idxlows = localdf[localdf['isPivot']==2].Low.index

    if len(lows)>=2 and len(highs)>=2:
        sl_lows, interc_lows, r_value_l, _, _ = stats.linregress(idxlows,lows)
        sl_highs, interc_highs, r_value_h, _, _ = stats.linregress(idxhighs,highs)

        return(sl_lows, interc_lows, sl_highs, interc_highs, r_value_l**2, r_value_h**2)
    else:
        return(0,0,0,0,0,0)

def plot_channel(dfpl, sl_lows, interc_lows, sl_highs, interc_highs):
    candle = 200
    backcandles = 30
    window = 2

    #dfpl = df[candle-backcandles-window-5:candle+200]

    fig = go.Figure(data=[go.Candlestick(x=dfpl.index,
                    open=dfpl['Open'],
                    high=dfpl['High'],
                    low=dfpl['Low'],
                    close=dfpl['Close'])])

    fig.add_scatter(x=dfpl.index, y=dfpl['pointpos'], mode="markers",
                    marker=dict(size=5, color="MediumPurple"),
                    name="pivot")

    sl_lows, interc_lows, sl_highs, interc_highs, r_sq_l, r_sq_h = collect_channel(candle, backcandles, window, dfpl)
    print(r_sq_l, r_sq_h)
    x = np.array(range(candle-backcandles-window, candle+1))
    fig.add_trace(go.Scatter(x=x, y=sl_lows*x + interc_lows, mode='lines', name='lower slope'))
    fig.add_trace(go.Scatter(x=x, y=sl_highs*x + interc_highs, mode='lines', name='max slope'))
    #fig.update_layout(xaxis_rangeslider_visible=False)
    fig.show()

def isBreakOut(candle, backcandles, window, df):
    if (candle-backcandles-window)<0:
        return 0

    sl_lows, interc_lows, sl_highs, interc_highs, r_sq_l, r_sq_h = collect_channel(candle,
                                                                                   backcandles,
                                                                                   window,df)

    prev_idx = candle-1
    prev_high = df.iloc[candle-1].High
    prev_low = df.iloc[candle-1].Low
    prev_close = df.iloc[candle-1].Close

    curr_idx = candle
    curr_high = df.iloc[candle].High
    curr_low = df.iloc[candle].Low
    curr_close = df.iloc[candle].Close
    curr_open = df.iloc[candle].Open

    if ( prev_high > (sl_lows*prev_idx + interc_lows) and
        prev_close < (sl_lows*prev_idx + interc_lows) and
        curr_open < (sl_lows*curr_idx + interc_lows) and
        curr_close < (sl_lows*prev_idx + interc_lows)): #and r_sq_l > 0.9
        return 1

    elif ( prev_low < (sl_highs*prev_idx + interc_highs) and
        prev_close > (sl_highs*prev_idx + interc_highs) and
        curr_open > (sl_highs*curr_idx + interc_highs) and
        curr_close > (sl_highs*prev_idx + interc_highs)): #and r_sq_h > 0.9
        return 2

    else:
        return 0

def breakpointpos(x):
    if x['isBreakOut']==2:
        return x['Low']-3e-3
    elif x['isBreakOut']==1:
        return x['High']+3e-3
    else:
        return np.nan

def plot_breakout(dfpl, backcandles, window):
    candle = len(dfpl)
    df = dfpl
    
    dfpl = df[candle-backcandles-window-5:candle+20]
    dfpl["isBreakOut"] = [isBreakOut(candle, backcandles, window, df) for candle in dfpl.index]
    dfpl['breakpointpos'] = dfpl.apply(lambda row: breakpointpos(row), axis=1)

    fig = go.Figure(data=[go.Candlestick(x=dfpl.index,
                    open=dfpl['Open'],
                    high=dfpl['High'],
                    low=dfpl['Low'],
                    close=dfpl['Close'])])

    fig.add_scatter(x=dfpl.index, y=dfpl['pointpos'], mode="markers",
                    marker=dict(size=5, color="MediumPurple"),
                    name="pivot")

    fig.add_scatter(x=dfpl.index, y=dfpl['breakpointpos'], mode="markers",
                    marker=dict(size=8, color="Black"), marker_symbol="hexagram",
                    name="pivot")

    sl_lows, interc_lows, sl_highs, interc_highs, r_sq_l, r_sq_h = collect_channel(candle, backcandles, window, df)
    
    x = np.array(range(candle-backcandles-window, candle+1))
    fig.add_trace(go.Scatter(x=x, y=sl_lows*x + interc_lows, mode='lines', name='lower slope'))
    fig.add_trace(go.Scatter(x=x, y=sl_highs*x + interc_highs, mode='lines', name='max slope'))
    #fig.update_layout(xaxis_rangeslider_visible=False)
    fig.show()
    fig.write_html('breakout.html')
    return dfpl, r_sq_l, r_sq_h
    
    

def main():
    
    ticker = "^NSEBANK"
    time_interval = "15m"
    
    data = fetch_data(ticker, time_interval)
    df = prepare_data(data)
    
    backcandles = 30
    window = 2
    
    df['isPivot'] = df.apply(lambda x: isPivot(x.name, window, df), axis=1)
    df['pointpos'] = df.apply(lambda row: pointpos(row), axis=1)
    
    dfpl, r_sq_l, r_sq_h = plot_breakout(df, backcandles, window)
    
    for index, row in dfpl.iterrows():
        if row['isBreakOut'] == 1 or row['isBreakOut'] == 2:
            if index + 1 == len(data):
                print("Today Breakout Detected\n")
                print("Confidence level : " + str(round((r_sq_l + r_sq_h)*100/2, 2))+"%")
                text = "*Asset : ^NSEBANK*\n" + "*Breakout Detected @ [15min]*\n" + "*Confidence level : " + str(round((r_sq_l + r_sq_h)*100/2, 2))+"%*"
                
                send_html("breakout.html")
                send_message(text)
                
                break
            
            

# Looper
def check_and_call_function():
    current_time = datetime.now().time()
    
    print("\r" +"Time: "+ str(current_time), end='', flush=True)
    
    hour = 9
    
    start_timek = datetime_time(hour, 15)
    end_timek = datetime_time(hour, 17)
            
    if start_timek <= current_time <= end_timek:
        print(" ")
        main()
        print("Re-Run at " + str(current_time))
        time.sleep(2 * 60)


while True:
    check_and_call_function()
    time.sleep(60)  # 60-second wait        

            
            


