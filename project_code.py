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
import seaborn as sns

# %%
#-----------------------------------------------------------------------------------------------------------------------------
# 2) Downloading data
#-----------------------------------------------------------------------------------------------------------------------------

BND = pd.read_csv("BND ETF Stock Price History.csv", parse_dates=['Date'], index_col='Date')
GBP_USD = pd.read_csv("GBP_USD Historical Data.csv", parse_dates=['Date'], index_col='Date')
MSFT = pd.read_csv("MSFT Historical Data.csv", parse_dates=['Date'], index_col='Date')
ND100 = pd.read_csv("Nasdaq 100 Historical Data.csv", parse_dates=['Date'], index_col='Date')
NG = pd.read_csv("Natural Gas Futures Historical Data.csv", parse_dates=['Date'], index_col='Date')


# %%
#-----------------------------------------------------------------------------------------------------------------------------
# 3) Visualizing price across time
#-----------------------------------------------------------------------------------------------------------------------------

plt.figure(figsize=(12, 6))
plt.plot(BND.index, BND['Price'], label='Price', color='blue')
plt.title('BND Price Over Time')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(12, 6))
plt.plot(GBP_USD.index, GBP_USD['Price'], label='Price', color='green')
plt.title('GBP_USD Price Over Time')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(12, 6))
plt.plot(MSFT.index, MSFT['Price'], label='Price', color='pink')
plt.title('MSFT Price Over Time')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(12, 6))
plt.plot(ND100.index, ND100['Price'], label='Price', color='blue')
plt.title('ND100 Price Over Time')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.grid(True)
plt.show()

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
# 4) Basic statistics of price across time
#-----------------------------------------------------------------------------------------------------------------------------

print(BND['Price'].describe())
print(GBP_USD['Price'].describe())
print(MSFT['Price'].describe())
print(ND100['Price'].describe())
print(NG['Price'].describe())

# %%
#-----------------------------------------------------------------------------------------------------------------------------
# 5) Defining strategy
#-----------------------------------------------------------------------------------------------------------------------------


def trend_following_strategy(data, price_column, short_window=20, long_window=50):
    
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

result_df = trend_following_strategy(data=NG, price_column='Price')


# %%
