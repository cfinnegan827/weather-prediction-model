from datetime import datetime
from meteostat import Daily, Point
import pandas as pd
import ssl

# Disable SSL verification
ssl._create_default_https_context = ssl._create_unverified_context

# Set time period
start = datetime(2014, 10, 27)
end = datetime(2024, 10, 27)
Boston = Point(42.361145, -71.057083)
data = Daily(Boston, start, end)
data = data.fetch()
data=data.reset_index().iloc[:,[0,1,2,3,4,6,7,9]]

data.to_csv('test.csv',index=False)