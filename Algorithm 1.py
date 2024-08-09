import decimal
import pandas as pd
import os
import re
import numpy as np
import collections
from sklearn.metrics import confusion_matrix
from sklearn import mixture
from scipy.spatial import distance
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt

os.chdir('\raw extracted data')
filegroup = ['119_1_denoised.txt','120_1_denoised.txt','120_2_denoised.txt','121_2_denoised.txt','122_1_denoised.txt','123_1_denoised.txt','125_1_denoised.txt','127_1_denoised.txt','128_1_denoised.txt','128_2_denoised.txt','128_3_denoised.txt','129_1_denoised.txt','129_2_denoised.txt','130_1_denoised.txt']
file_len= [0.2,0.2,0.4,0.3,0.3,0.3,0.435,0.2,0.3,0.3,0.4,0.4,0.3,0.3]
l=[]
j=1

smooth_lines_total=[]
smooth_length_total=[]

ind=0
leng=[]
z1=[]
for filename in filegroup:
    data = pd.read_csv(filename,  sep="\t", header = None, index_col=False, float_precision= "round_trip", names=["north", "east", "height", "time", "date", "lines", "length", "green", "status"])

    z2= data.groupby(["north", "east"], sort=False)["lines","length"].mean()
    leng.append(len(z2))

    if int(7.5/file_len[ind]) % 2 == 0:
        window_size = int(7.5/file_len[ind])+1
    else:
        window_size = int(7.5/file_len[ind])    
    poly_order = 1    
    print(window_size)
    smooth_lines = savgol_filter(z2.lines, window_size, poly_order)
    smooth_lines_total.extend(smooth_lines)

    smooth_length = savgol_filter(z2.length, window_size, poly_order)
    smooth_length_total.extend(smooth_length)
      
    j=j+1
    ind=ind+1
    
    
smooth_lines_total=np.asarray(smooth_lines_total)
smooth_length_total=np.asarray(smooth_length_total)
    
smooth_lines_total[smooth_lines_total < 0] = 0
smooth_length_total[smooth_length_total < 0] = 0


lower_quantile_lines = np.percentile(smooth_lines_total,7)
upper_quantile_lines = np.percentile(smooth_lines_total,99)

lower_quantile_length = np.percentile(smooth_length_total,7)
upper_quantile_length = np.percentile(smooth_length_total,99)

## generate Individual files based on the upper and lower quantiles

l=[]
j=1
smooth_lines_total=[]
smooth_length_total=[]
ind=0
leng=[]
z1=[]
for filename in filegroup:
    data = pd.read_csv(filename,  sep="\t", header = None, index_col=False, float_precision= "round_trip", names=["north", "east", "height", "time", "date", "lines", "length", "green", "status"])

    z2= data.groupby(["north", "east"], sort=False)["lines","length"].mean()
    leng.append(len(z2))

    if int(7.5/file_len[ind]) % 2 == 0:
        window_size = int(7.5/file_len[ind])+1
    else:
        window_size = int(7.5/file_len[ind])    
    poly_order = 1    
    
    smooth_lines = savgol_filter(z2.lines, window_size, poly_order)
    smooth_length = savgol_filter(z2.length, window_size, poly_order)
    smooth_lines_new = (smooth_lines -  lower_quantile_lines)/upper_quantile_lines
    smooth_length_new = (smooth_length - lower_quantile_length )/upper_quantile_length 
    smooth_lines_new[smooth_lines_new < 0] = 0
    smooth_lines_new[smooth_lines_new > 1] = 1
    smooth_length_new[smooth_length_new > 1] = 1
    smooth_length_new[smooth_length_new < 0] = 0
    
    combination= ((smooth_lines_new)*0.62 + (smooth_length_new)*0.38 )*100
    
    fig=plt.figure(num=None, figsize=(10, 8), dpi=150, facecolor='w', edgecolor='k')
    plt.plot(combination)
    plt.title(filename)
    plt.xlabel('Unique locations')
    plt.ylabel('Percentage coverage')
    #plt.plot([0, len(data.index)], [30, 30], 'r--', lw=2)
    plt.plot([0, len(combination)], [10, 10], 'k--', lw=2)

    plt.savefig('filename+'.jpg')
    
    j=j+1
    ind=ind+1
