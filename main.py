#import TelegramModel
import DataModel
import matplotlib as mpl
import numpy as np
import matplotlib.pyplot as plt


model = DataModel.DataModel('BTC/USDT','1d')
df = model.get_data()
print(df)


#df[["close"]].plot()
#plt.show()






