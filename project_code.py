#-----------------------------------------------------------------------------------------------------------------------------
#Algorithmic Trading
#MiBDS, 2nd Year, Part-Time
#Academic Year: 2024/2025
#Jan Galiński (40867)
#Individual Work Project
#"Finding signals in financial markets"
#-----------------------------------------------------------------------------------------------------------------------------

# %%
#-----------------------------------------------------------------------------------------------------------------------------
# 1) Importing libraries
#-----------------------------------------------------------------------------------------------------------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# %%
#-----------------------------------------------------------------------------------------------------------------------------
# 2) Downloading data
#-----------------------------------------------------------------------------------------------------------------------------

#The datasets contain daily data of selected assets for 5 years in various timeframes

#BND stands for Vanguard Total Bond Market ETF (bonds ETF)
BND = pd.read_csv("BND ETF Stock Price History.csv", parse_dates=['Date'], index_col='Date')
#GBP_USD stands for British Pound Sterling / US Dollar (FX)
GBP_USD = pd.read_csv("GBP_USD Historical Data.csv", parse_dates=['Date'], index_col='Date')
#MSFT stands for Microsoft Corporation (stock)
MSFT = pd.read_csv("MSFT Historical Data.csv", parse_dates=['Date'], index_col='Date')
#ND100 stands for NASDAQ 100 Index (index)
ND100 = pd.read_csv("Nasdaq 100 Historical Data.csv", parse_dates=['Date'], index_col='Date')
#NG stands for Natural Gas Futures (future)
NG = pd.read_csv("Natural Gas Futures Historical Data.csv", parse_dates=['Date'], index_col='Date')


# %%
#-----------------------------------------------------------------------------------------------------------------------------
# 3) Visualizing price of each asset across time
#-----------------------------------------------------------------------------------------------------------------------------

#plot BND
plt.figure(figsize=(12, 6))
plt.plot(BND.index, BND['Price'], label='Price', color='blue')
plt.title('BND Price Over Time')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.grid(True)
plt.show()

#plot GBP_USD
plt.figure(figsize=(12, 6))
plt.plot(GBP_USD.index, GBP_USD['Price'], label='Price', color='green')
plt.title('GBP_USD Price Over Time')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.grid(True)
plt.show()

#plot MSFT
plt.figure(figsize=(12, 6))
plt.plot(MSFT.index, MSFT['Price'], label='Price', color='pink')
plt.title('MSFT Price Over Time')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.grid(True)
plt.show()

#plot ND100
plt.figure(figsize=(12, 6))
plt.plot(ND100.index, ND100['Price'], label='Price', color='blue')
plt.title('ND100 Price Over Time')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.grid(True)
plt.show()

#plot NG
plt.figure(figsize=(12, 6))
plt.plot(NG.index, NG['Price'], label='Price', color='green')
plt.title('NG Price Over Time')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.grid(True)
plt.show()


# %%
#-----------------------------------------------------------------------------------------------------------------------------
# 4) Basic statistics of price of each asset across time
#-----------------------------------------------------------------------------------------------------------------------------

#BND statistics
print("BND statistics")
print(BND['Price'].describe())
#GBP_USD statistics
print("GBP_USD statistics")
print(GBP_USD['Price'].describe())
#MSFT statistics
print("MSFT statistics")
print(MSFT['Price'].describe())
#ND100 statistics
print("ND100 statistics")
print(ND100['Price'].describe())
#NG statistics
print("NG statistics")
print(NG['Price'].describe())

# %%
#-----------------------------------------------------------------------------------------------------------------------------
# 5) Defining strategy
#-----------------------------------------------------------------------------------------------------------------------------

#Strategy
#The selected strategy for signal is trend-following using moving average crossover
#It is quite simple and popular strategy among investors 
#The objective is to identify the direction of market trend and use it to advantage
#There will be defined two moving averages (MA): one short term and one long term
#Buy signal will be defined as when short term MA crosses above long term MA
#Sell signal will be defined as when short term MA crosses below long term MA
#After buy signal position will be held until a sell signal
#Similarly after sell signal position will be held until a buy signal

#Economic reasoning
#The key assumption of this strategy arguing that it should work is basing on price momentum
#As a result trends tend to persist for given time period and allow to take profit by leveraging them
#What is more taking into account long term MA, allows to filter out only short term noise/fluctuations
#On the other hand this strategy can fail due to only basing on historical data (lagging) and failing to quickly respond
#Lastly this strategy can have limitations in markets where there are no clear trends present

#define strategy function
def trend_following_strategy(data, price_column, short_window=10, long_window=50):
    
    #data load
    df = data
    
    #moving averages (MA) calculation
    df['Short_MA'] = df[price_column].rolling(window=short_window, min_periods=1).mean()
    df['Long_MA'] = df[price_column].rolling(window=long_window, min_periods=1).mean()
    
    #trading signals
    df['Signal'] = 0
    df['Signal'][short_window:] = \
        (df['Short_MA'][short_window:] > df['Long_MA'][short_window:]).astype(int)
    
    #buy/sell signals
    df['Position'] = df['Signal'].diff()
    
    #results
    plt.figure(figsize=(14, 8))
    plt.plot(df.index, df[price_column], label=f'{price_column}', alpha=0.5)
    plt.plot(df.index, df['Short_MA'], label=f'Short {short_window}-Day MA', alpha=0.75)
    plt.plot(df.index, df['Long_MA'], label=f'Long {long_window}-Day MA', alpha=0.75)
    
    #visualize buy signals
    plt.plot(df[df['Position'] == 1].index, 
             df['Short_MA'][df['Position'] == 1], 
             '^', markersize=10, color='g', label='Buy Signal')
    
    #visualize sell signals
    plt.plot(df[df['Position'] == -1].index, 
             df['Short_MA'][df['Position'] == -1], 
             'v', markersize=10, color='r', label='Sell Signal')
    
    plt.title('Trend Following Strategy using Moving Averages')
    plt.xlabel('Date')
    plt.ylabel(f'{price_column} Price')
    plt.legend()
    plt.grid()
    plt.show()
    
    return df

#results per each asset
results_BND = trend_following_strategy(data=BND, price_column='Price')
results_GBP_USD = trend_following_strategy(data=GBP_USD, price_column='Price')
results_MSFT = trend_following_strategy(data=MSFT, price_column='Price')
results_ND100 = trend_following_strategy(data=ND100, price_column='Price')
results_NG = trend_following_strategy(data=NG, price_column='Price')

# %%
#-----------------------------------------------------------------------------------------------------------------------------
# 6) Testing performance
#-----------------------------------------------------------------------------------------------------------------------------

#define testing performance function
def calculate_performance(df, price_column='Price'):

    #daily returns
    df['Return'] = df[price_column].pct_change()
    
    #variables for strategy performance
    position = 0  # 1 for holding position and 0 for not holding
    entry_price = 0
    df['Strategy_Return'] = 0

    for i in range(1, len(df)):
        if df['Position'].iloc[i] == 1:  #Buy signal
            position = 1
            entry_price = df[price_column].iloc[i]
        elif df['Position'].iloc[i] == -1 and position == 1:  #Sell signal
            position = 0
            #return for the holding period
            df['Strategy_Return'].iloc[i] = (df[price_column].iloc[i] - entry_price) / entry_price

    #cumulative returns
    df['Cumulative_Strategy_Return'] = (1 + df['Strategy_Return']).cumprod() - 1

    #Sharpe Ratio (taking the assumption of risk-free rate of 0%)
    sharpe_ratio = np.sqrt(252) * df['Strategy_Return'].mean() / df['Strategy_Return'].std()

    #Maximum Drawdown
    cumulative_return = (1 + df['Strategy_Return']).cumprod()
    drawdown = (cumulative_return / cumulative_return.cummax()) - 1
    max_drawdown = drawdown.min()
    
    #Win Rate
    trades = df[df['Strategy_Return'] != 0]
    win_rate = len(trades[trades['Return'] > 0]) / len(trades) if len(trades) > 0 else 0

    #save performance metrics
    performance_metrics = {
        'Sharpe Ratio': sharpe_ratio,
        'Maximum Drawdown': max_drawdown,
        'Win Rate': win_rate
    }
    metrics_df = pd.DataFrame([performance_metrics])

    #results
    print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
    print(f"Maximum Drawdown: {max_drawdown:.2%}")
    print(f"Win Rate: {win_rate:.2%}")
    
    #visualize cumulative returns
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df['Cumulative_Strategy_Return'], label='Cumulative Strategy Return')
    plt.title('Cumulative Returns of Moving Average Strategy')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Return')
    plt.grid(True)
    plt.legend()
    plt.show()
    
    return df, metrics_df


#results with performance per each asset
results_BND = trend_following_strategy(data=BND, price_column='Price')
performance_BND, metrics_BND = calculate_performance(results_BND, price_column='Price')

results_GBP_USD = trend_following_strategy(data=GBP_USD, price_column='Price')
performance_GBP_USD, metrics_GBP_USD = calculate_performance(results_GBP_USD, price_column='Price')

results_MSFT = trend_following_strategy(data=MSFT, price_column='Price')
performance_MSFT, metrics_MSFT = calculate_performance(results_MSFT, price_column='Price')

results_ND100 = trend_following_strategy(data=ND100, price_column='Price')
performance_ND100, metrics_ND100 = calculate_performance(results_ND100, price_column='Price')

results_NG = trend_following_strategy(data=NG, price_column='Price')
performance_NG, metrics_NG = calculate_performance(results_NG, price_column='Price')

# %%
