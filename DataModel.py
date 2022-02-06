import ccxt
from datetime import datetime, timedelta
import pandas as pd



class DataModel():
    
    def __init__(self,market,periods):
        self.exchange = ccxt.binance()
        self.market = market
        self.periods = periods
        
        self.get_data()

        
    def get_data(self):
        self.ohlcvs = self.exchange.fetch_ohlcv(self.market, self.periods)
        #일자, 시가, 고가, 저가, 종가, 거래량
        self.df = pd.DataFrame(self.ohlcvs, columns=['datetime', 'open', 'high', 'low', 'close', 'volume'])
        self.df['datetime'] = pd.to_datetime(self.df['datetime'], unit='ms')
        self.df['datetime'] = pd.DatetimeIndex(self.df['datetime']) + timedelta(hours=9)
        self.df.set_index('datetime', inplace=True)
        self.df.reset_index(inplace=True)
        return self.df

    