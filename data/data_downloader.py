"""
data_downloader.py

ludicrously simple script to parse the file of urls and download them to 
the specified filename

author: alex shenfield
date:   25/04/2020
"""

import urllib.request
import pandas as pd

# get the urls
df = pd.read_csv('./48k_drive_end_fault_urls.csv', header=None)

# for every line download the file to the corresponding filename
for f in df.iterrows():
    urllib.request.urlretrieve(f[1][0], f[1][1])