import ccxt
from datetime import datetime, timedelta
import pandas as pd
import TelegramModel


exchange = ccxt.binance()
periods = '1d'
market = 'BTC/USDT'

ohlcvs = exchange.fetch_ohlcv(market, periods)
#일자, 시가, 고가, 저가, 종가, 거래량
df = pd.DataFrame(ohlcvs, columns=['datetime', 'open', 'high', 'low', 'close', 'volume'])
df['datetime'] = pd.to_datetime(df['datetime'], unit='ms')
df['datetime'] = pd.DatetimeIndex(df['datetime']) + timedelta(hours=9)
df.set_index('datetime', inplace=True)
df.reset_index(inplace=True)


telegram_bot = TelegramModel.TelegramModel()
telegram_bot.send_message("안녕")