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
