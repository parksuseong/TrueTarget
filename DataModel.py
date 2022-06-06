
import ccxt
from datetime import datetime, timedelta
import pandas as pd


"""
일봉 = 4시간봉 * 6 = 1시간봉 * 96 = 30분봉 * 192 = 15분봉 * 384 = 5분봉 * 1152
"""

class DataModel():
    
    def __init__(self,market,periods,from_date,to_date):
        self.exchange = ccxt.binance()
        self.market = market
        self.periods = periods
        self.df = pd.DataFrame([], columns=['datetime', 'open', 'high', 'low', 'close', 'volume'])
        self.get_data(from_date=from_date,to_date=to_date)
        self.calc_rsi()
        
    def get_data(self,from_date=None,to_date=None):
        to_date = int(pd.to_datetime(to_date).timestamp()*1000)
        from_date = int(pd.to_datetime(from_date).timestamp()*1000)
        
        while(from_date <= to_date):
            ohlcvs = self.exchange.fetch_ohlcv(self.market, self.periods, since=from_date, limit=1000)
            tmp_df = pd.DataFrame(ohlcvs, columns=['datetime', 'open', 'high', 'low', 'close', 'volume'])   
            self.df = pd.concat([self.df,tmp_df])
            from_date = int(tmp_df['datetime'][-1:])+1

        
        del_idx = self.df[ self.df['datetime'] >= to_date].index
        self.df.drop(del_idx,inplace=True)
        
        self.df['datetime'] = pd.to_datetime(self.df['datetime'], unit='ms')
        self.df['datetime'] = pd.DatetimeIndex(self.df['datetime']) #+ timedelta(hours=9)
        self.df.set_index('datetime', inplace=True)
        self.df.reset_index(inplace=True)  
        return self.df


        

    #calculate rsi
    def calc_rsi(self, periods=14, ema=True):
        close_delta=self.df['close'].diff()
        up=close_delta.clip(lower=0)
        down=-1*close_delta.clip(upper=0)

        if ema: #exponentail moving avg
            ma_up=up.ewm(com=periods-1, adjust=True, min_periods=periods).mean()
            ma_down=down.ewm(com=periods-1, adjust=True, min_periods=periods).mean()
        else: #simple moving avg
            ma_up=up.rolling(window.periods).mean()
            ma_down=up.rolling(window.periods).mean()

        rsi=ma_up/ma_down
        rsi=100-(100/(1+rsi))
        rsi.fillna(0,inplace=True)
        rsi = rsi.to_frame()
        rsi.columns = ['rsi']
        
        self.df = pd.concat([self.df,rsi], axis=1, join='inner')
        
        #result = pd.DataFrame(result, columns=['datetime', 'open','high','low','close','vol','rsi'])
        self.df.columns = ['datetime','open','high','low','close','volume','rsi']

        del_idx = self.df[ self.df['rsi'] == 0].index
        self.df.drop(del_idx,inplace=True)

        return self.df

    def get_df(self):
        return self.df