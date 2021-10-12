### The DataOracle class reads real historical trade data (not price or quote)
### from a given date in history to be resimulated.  It stores these trades
### in a time-sorted array at maximum resolution.  It can be called by
### certain "background" agents to obtain noisy observations about the "real"
### price of a stock at a current time.  It is intended to provide some realistic
### behavior and "price gravity" to the simulated market -- i.e. to make the
### market behave something like historical reality in the absence of whatever
### experiment we are running with more active agent types.

import datetime as dt
import numpy as np
import pandas as pd
import os, sys
from datetime import datetime, timedelta

from math import sqrt
from util.util import print, log_print

from joblib import Memory
mem = Memory(cachedir='./cache', verbose=0)


#@mem.cache
def read_trades(trade_file, symbols):
  log_print ("Data not cached.  This will take a minute...")

  df = pd.read_pickle(trade_file, compression='bz2')

  # Filter to requested symbols.
  df = df.loc[symbols]

  # Filter duplicate indices (trades on two exchanges at the PRECISE same time).  Rare.
  df = df[~df.index.duplicated(keep='first')]

  # Ensure resulting index is sorted for best performance later on.
  df = df.sort_index()

  return (df)


class DataOracle:

  def __init__(self, historical_date = None, symbols = None, data_dir = None):
    self.historical_date = historical_date
    self.symbols = symbols

    self.mkt_open = None

    # Read historical trades here...
    h = historical_date
    # pre = 'ct' if h.year < 2015 else 'ctm'
    # trade_file = os.path.join(data_dir, 'trades', 'trades_{}'.format(h.year),
    #                           '{}_{}{:02d}{:02d}.bgz'.format(pre, h.year, h.month, h.day))

    # bars_1m_file = os.path.join(data_dir, '1m_ohlc', '1m_ohlc_{}'.format(h.year),
    #                           '{}{:02d}{:02d}_ohlc_1m.bgz'.format(h.year, h.month, h.day))
    def convertDate(date_str):
            try:
                return datetime.fromtimestamp(date_str) + timedelta( days=17885, hours=14, minutes=30)
            except ValueError:
                return convertDate(date_str[:-1])
    file_path = os.path.join(data_dir, 'lobster', 'AAPL_order.csv')
    orders_df = pd.read_csv(file_path)
    orders_df = orders_df[orders_df['TYPE'].astype(int)==4]
    orders_df['TIMESTAMP'] = orders_df['TIMESTAMP'].astype(float).apply(convertDate).dt.floor('Min')
    orders_df['SIZE'] = orders_df['SIZE'].astype(int)
    orders_df['PRICE'] = orders_df['PRICE'].astype(int)

    orders_df = orders_df.set_index('TIMESTAMP')
    self.df_bars_1m =orders_df.drop_duplicates(['TIMESTAMP'], keep='first')
    # log_print ("DataOracle initializing trades from file {}", trade_file)
    # log_print ("DataOracle initializing 1m bars from file {}", bars_1m_file)

    # then = dt.datetime.now()
    # self.df_trades = read_trades(trade_file, symbols)
    # self.df_bars_1m = read_trades(bars_1m_file, symbols)
    # now = dt.datetime.now()

    # log_print ("DataOracle initialized for {} with symbols {}", historical_date, symbols)
    # log_print ("DataOracle initialization took {}", now - then)



  # Return the daily open price for the symbol given.  The processing to create the 1m OHLC
  # files does propagate the earliest trade backwards, which helps.  The exchange should
  # pass its opening time.
  def getDailyOpenPrice (self, symbol, mkt_open, cents=True):
    # Remember market open time.
    self.mkt_open = mkt_open

    log_print ("Oracle: client requested {} at market open: {}", symbol, mkt_open)

    # Find the opening historical price in the 1m OHLC bars for this symbol.
    # open = self.df_bars_1m.loc[(symbol,mkt_open.time()),'open']
    open = self.df_bars_1m['PRICE'].iloc[0]
    log_print ("Oracle: market open price was was {}", open)

    return int(round(open * 100)) if cents else open

  def getDailyClosePrice (self, symbol, mkt_close, cents=True):
    # Remember market open time.
    self.mkt_close = mkt_close

    log_print ("Oracle: client requested {} at market open: {}", symbol, mkt_close)

    # Find the opening historical price in the 1m OHLC bars for this symbol.
    # open = self.df_bars_1m.loc[(symbol,mkt_open.time()),'open']
    close = self.df_bars_1m['PRICE'].iloc[-1]
    log_print ("Oracle: market close price was was {}", close)

    return int(round(close * 100)) if cents else close
  # Return the latest trade price for the symbol at or prior to the given currentTime,
  # which must be of type pd.Timestamp.
  def getLatestTrade (self, symbol, currentTime):

    log_print ("Oracle: client requested {} as of {}", symbol, currentTime)

    # See when the last historical trade was, prior to simulated currentTime.
    # dt_last_trade = self.df_trades.loc[symbol].index.asof(currentTime)
    dt_last_trade = self.df_trades.index.asof(currentTime)
    if pd.notnull(dt_last_trade):
      # last_trade = self.df_trades.loc[(symbol,dt_last_trade)]
      price  = self.df_trades['PRICE'].loc[dt_last_trade]

      # price = last_trade['PRICE']
      time = dt_last_trade

    # If we know the market open time, and the last historical trade was before it, use
    # the market open price instead.  If there were no trades before the requested time,
    # also use the market open price.
    if pd.isnull(dt_last_trade) or (self.mkt_open and time < self.mkt_open):
      price = self.getDailyOpenPrice(symbol, self.mkt_open, cents=False)
      time = self.mkt_open

    log_print ("Oracle: latest historical trade was {} at {}", price, time)

    return price


  # Return a noisy observed historical price for agents which have that ability.
  # currentTime must be of type pd.Timestamp.  Only the Exchange or other privileged
  # agents should use noisy=False.
  #
  # NOTE: sigma_n is the observation variance, NOT STANDARD DEVIATION.
  #
  # Each agent must pass its own np.random.RandomState object to the oracle.
  # This helps to preserve the consistency of multiple simulations with experimental
  # changes (if the oracle used a global Random object, simply adding one new agent
  # would change everyone's "noise" on all subsequent observations).
  def observePrice(self, symbol, currentTime, sigma_n = 0.0001, random_state = None):
    last_trade_price = self.getLatestTrade(symbol, currentTime)

    # Noisy belief is a normal distribution around 1% the last trade price with variance
    # as requested by the agent.
    if sigma_n == 0:
      belief = float(last_trade_price)
    else:
      belief = random_state.normal(loc=last_trade_price, scale=last_trade_price * sqrt(sigma_n))

    log_print ("Oracle: giving client value observation {:0.2f}", belief)

    # All simulator prices are specified in integer cents.
    return int(round(belief * 100))

