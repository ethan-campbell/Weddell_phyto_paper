from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
import matplotlib.dates as mdates

dates = [datetime(2018,1,4),
         datetime(2018,1,5),
         datetime(2018,1,6),
         datetime(2018,1,7),
         datetime(2018,1,8),
         datetime(2018,1,9),
         datetime(2018,1,10),
         datetime(2018,1,11),
         datetime(2018,1,12),
         datetime(2018,1,14),
         datetime(2018,1,15),
         datetime(2018,1,16),
         datetime(2018,1,17),
         datetime(2018,1,18),
         datetime(2018,1,19),
         datetime(2018,1,20)]

weights = [8 + 15/16,
           8 + 15/16,
           9,
           9 + 1.5/16,
           9 + 2.5/16,
           9 + 3.5/16,
           9 + 3.5/16,
           9 + 6.5/16,
           9 + 5/16,
           9 + 4/16,
           9 + 6/16,
           9 + 7/16,
           9 + 8.5/16,
           9 + 6.5/16,
           9 + 7.5/16,
           9 + 11/16]

data = pd.Series(index=dates,data=weights)

[slope,intercept,_,p] = stats.linregress(mdates.date2num(data.index.date),data.values)[0:4]

fig = plt.figure(figsize=(7,5))
plt.plot(dates,mdates.date2num(dates)*slope + intercept,'k-',
         label='p = {0:.02f} (trend is significant\nat 100% confidence level)'.format(p))
plt.scatter(dates,weights,c='purple',s=150)
plt.legend()
fig.autofmt_xdate()
plt.ylabel('Weight (lbs)')
plt.title("Mister's weight")
plt.tight_layout()