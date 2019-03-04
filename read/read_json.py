import json
import pandas as pd
import numpy as np

# function to map dict to new keys
def map_dict(dict,key):
    # if key exists in the curent dict
    try:
        return dict[key]
    # if key does not exist return nan
    except:
        return np.nan

fl_nm = '2018-09-30.json'
lines = (line for line in open(fl_nm))

ln = json.loads(next(lines))
header=ln.keys()

#reading the whole file line by line and finding the true header
h=0
for line in lines:
    ln=json.loads(line)
    hdr = ln.keys()
    if len(hdr)>len(header):
        header = hdr

    h=h+1
    if h%10000==0:
        print h

# reading the file again line by line and transforming the dict as per the true header
fl_nm = '2018-09-30.json'
lines = (line for line in open(fl_nm))

lst=[]
i=0
for line in lines:
    ln=json.loads(line)
    ln_new = {k:map_dict(ln,k) for k in header}
    lst.append(ln_new.values())
    i=i+1
    if i%1000==0:
        print i

#convert to dataframe
df = pd.DataFrame(lst,columns=header)
